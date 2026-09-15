"""Offline coverage regressions for the primary constituent source parser."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import certifi
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch.dict(os.environ, {"SSL_CERT_FILE": certifi.where()}):
    import sp500_breakout as screener


def source_page(rows, last_page=None):
    body = "".join(
        f'<tr><td class="ctg"><a href="/item/main.naver?code={ticker}" '
        f'target="_parent">{name}</a></td></tr>' for ticker, name in rows)
    pagination = (f'<td class="pgRR"><a href="/sise/entryJongmok.naver?'
                  f'&amp;page={last_page}&amp;type=KPI200">맨뒤</a></td>'
                  if last_page is not None else "")
    return f"<html><body><table>{body}</table><table>{pagination}</table></body></html>"


def kosdaq150_source(stock_count=150, *, duplicate=False, product_id="2ETF54"):
    stocks = [{"itmNo": f"{i:06d}", "secNm": f"코스닥 종목 {i}"}
              for i in range(1, stock_count + 1)]
    if stocks:
        stocks[0] = {"itmNo": "0126Z0", "secNm": "영문코드 종목"}
    if duplicate and len(stocks) > 1:
        stocks[-1] = dict(stocks[0])
    holdings = [{"itmNo": "KRD010010001", "secNm": "원화예금"}, *stocks]
    return json.dumps({
        "info": {"product": {
            "fId": product_id, "fNm": "KODEX 코스닥 150", "bmIdx": "코스닥 150 지수",
        }},
        "pdf": {
            "gijunYMD": "20260915", "totalCnt": str(len(holdings)),
            "nowCnt": str(len(holdings)), "list": holdings,
        },
    }, ensure_ascii=False)


class ConstituentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="constituents-regression-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legacy = self.root / "kospi200_constituents.csv"
        self.cache = self.root / "kospi200_constituents_v2.csv"
        self.kosdaq_cache = self.root / "kosdaq150_constituents.csv"
        self.patch_cache = patch.object(screener, "CACHE_DIR", self.root)
        self.patch_path = patch.object(screener, "KOSPI200_CSV", self.legacy)
        self.patch_kosdaq_path = patch.object(screener, "KOSDAQ150_CSV", self.kosdaq_cache)
        self.patch_cache.start()
        self.patch_path.start()
        self.patch_kosdaq_path.start()
        self.addCleanup(self.patch_path.stop)
        self.addCleanup(self.patch_kosdaq_path.stop)
        self.addCleanup(self.patch_cache.stop)

    def test_alphanumeric_codes_keep_their_own_names_and_following_rows(self):
        first = [("0126Z0", "삼성에피스홀딩스"), ("090430", "아모레퍼시픽"),
                 ("047040", "대우건설")]
        second = [("0220w0", "한화머시너리앤서비스홀딩스"), ("002030", "아세아")]
        with patch.object(screener, "_http_get", side_effect=[
                source_page(first, 2), source_page(second, 2)]) as fetch:
            result = screener.load_kospi200(refresh=True)
        self.assertEqual(result["ticker"].tolist(),
                         ["0126Z0", "090430", "047040", "0220W0", "002030"])
        self.assertEqual(result["name"].tolist(), [n for _, n in first + second])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(screener.MARKETS["kospi200"]["to_yahoo"]("0126Z0"), "0126Z0.KS")

    def test_old_misaligned_cache_is_never_reused(self):
        pd.DataFrame([{"ticker": "002030", "name": "잘못 연결된 이름", "sector": ""}]).to_csv(
            self.legacy, index=False)
        with patch.object(screener, "_http_get", return_value=source_page([
                ("0220W0", "한화머시너리앤서비스홀딩스"), ("002030", "아세아")], 1)):
            result = screener.load_kospi200()
        self.assertEqual(result["ticker"].tolist(), ["0220W0", "002030"])
        self.assertEqual(result.iloc[1]["name"], "아세아")
        self.assertTrue(self.cache.exists())
        with patch.object(screener, "_http_get") as fetch:
            cached = screener.load_kospi200()
        self.assertEqual(cached["ticker"].tolist(), ["0220W0", "002030"])
        self.assertEqual(cached.iloc[1]["name"], "아세아")
        fetch.assert_not_called()

    def test_missing_advertised_page_cannot_cache_a_partial_universe(self):
        with patch.object(screener, "_http_get", side_effect=[
                source_page([("005930", "삼성전자")], 2), source_page([])]):
            with self.assertRaises(SystemExit):
                screener.load_kospi200(refresh=True)
        self.assertFalse(self.cache.exists())

    def test_refresh_failure_uses_only_new_parser_cache(self):
        pd.DataFrame([{"ticker": "0126Z0", "name": "삼성에피스홀딩스", "sector": ""}]).to_csv(
            self.cache, index=False)
        with patch.object(screener, "_http_get", side_effect=RuntimeError("network unavailable")):
            result = screener.load_kospi200(refresh=True)
        self.assertEqual(result["ticker"].tolist(), ["0126Z0"])

    def test_kosdaq150_uses_all_150_stock_rows_and_excludes_cash(self):
        with patch.object(screener, "_http_get", return_value=kosdaq150_source()) as fetch:
            result = screener.load_kosdaq150(refresh=True)
        self.assertEqual(len(result), 150)
        self.assertEqual(result.iloc[0]["ticker"], "0126Z0")
        self.assertEqual(result.iloc[0]["name"], "영문코드 종목")
        self.assertNotIn("KRD010010001", result["ticker"].tolist())
        self.assertEqual(result["ticker"].nunique(), 150)
        self.assertEqual(screener.MARKETS["kosdaq150"]["to_yahoo"]("0126Z0"), "0126Z0.KQ")
        self.assertTrue(self.kosdaq_cache.exists())
        with patch.object(screener, "_http_get") as cached_fetch:
            cached = screener.load_kosdaq150()
        self.assertEqual(cached["ticker"].tolist(), result["ticker"].tolist())
        cached_fetch.assert_not_called()
        fetch.assert_called_once_with(screener.KODEX_KOSDAQ150)

    def test_kosdaq150_partial_or_duplicate_response_is_not_cached(self):
        for body in (kosdaq150_source(149), kosdaq150_source(150, duplicate=True),
                     kosdaq150_source(product_id="WRONG")):
            with self.subTest(body=body[:80]):
                self.kosdaq_cache.unlink(missing_ok=True)
                with patch.object(screener, "_http_get", return_value=body), \
                     self.assertRaises(SystemExit):
                    screener.load_kosdaq150(refresh=True)
                self.assertFalse(self.kosdaq_cache.exists())

    def test_kosdaq150_refresh_failure_uses_only_a_valid_stale_cache(self):
        valid = pd.DataFrame({
            "ticker": ["0126Z0", *[f"{i:06d}" for i in range(2, 151)]],
            "name": ["영문코드 종목", *[f"코스닥 종목 {i}" for i in range(2, 151)]],
            "sector": [""] * 150,
        })
        valid.to_csv(self.kosdaq_cache, index=False)
        with patch.object(screener, "_http_get", side_effect=RuntimeError("network unavailable")):
            result = screener.load_kosdaq150(refresh=True)
        self.assertEqual(len(result), 150)
        self.assertEqual(result.iloc[0]["ticker"], "0126Z0")


if __name__ == "__main__":
    unittest.main()

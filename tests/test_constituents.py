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


def krx_corp_list(rows):
    """KRX 상장법인목록 다운로드가 돌려주는 HTML 표를 흉내낸다."""
    head = ("<tr><th>회사명</th><th>시장구분</th><th>종목코드</th>"
            "<th>업종</th><th>상장일</th></tr>")
    body = "".join(
        f"<tr><td>{name}</td><td>유가</td><td>{ticker}</td>"
        f"<td>{sector}</td><td>2020-01-01</td></tr>"
        for ticker, name, sector in rows)
    return f"<html><body><table>{head}{body}</table></body></html>"


def krx_rows(n, *, start=1):
    return [(f"{i:06d}", f"코스피 종목 {i}", "제조업") for i in range(start, start + n)]


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
        self.cache = self.root / "kospi_all_constituents.csv"
        self.kosdaq_cache = self.root / "kosdaq150_constituents.csv"
        self.patch_cache = patch.object(screener, "CACHE_DIR", self.root)
        self.patch_path = patch.object(screener, "KOSPI_ALL_CSV", self.cache)
        self.patch_kosdaq_path = patch.object(screener, "KOSDAQ150_CSV", self.kosdaq_cache)
        self.patch_cache.start()
        self.patch_path.start()
        self.patch_kosdaq_path.start()
        self.addCleanup(self.patch_path.stop)
        self.addCleanup(self.patch_kosdaq_path.stop)
        self.addCleanup(self.patch_cache.stop)

    def test_reads_codes_and_names_from_the_krx_listing(self):
        rows = krx_rows(600)
        with patch.object(screener, "_http_get",
                          return_value=krx_corp_list(rows)) as fetch:
            result = screener.load_kospi_all(refresh=True)
        self.assertEqual(len(result), 600)
        self.assertEqual(result.iloc[0]["ticker"], "000001")
        self.assertEqual(result.iloc[0]["name"], "코스피 종목 1")
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(screener.MARKETS["kospi"]["to_yahoo"]("005930"), "005930.KS")

    def test_drops_spacs_and_codes_yahoo_cannot_resolve(self):
        rows = krx_rows(600) + [
            ("0220W0", "한화머시너리앤서비스홀딩스", "기타 금융업"),
            ("123456", "엔에이치스팩34호", "금융 지원 서비스업"),
        ]
        with patch.object(screener, "_http_get", return_value=krx_corp_list(rows)):
            result = screener.load_kospi_all(refresh=True)
        self.assertEqual(len(result), 600)
        self.assertNotIn("0220W0", result["ticker"].tolist())
        self.assertNotIn("123456", result["ticker"].tolist())

    def test_a_suspiciously_short_listing_is_never_cached(self):
        # 목록이 갑자기 짧아지는 것은 형식 변경이나 차단 페이지를 받은 신호다.
        # 그대로 캐시하면 다음 날부터 조용히 축소된 유니버스를 스캔하게 된다.
        with patch.object(screener, "_http_get",
                          return_value=krx_corp_list(krx_rows(20))):
            with self.assertRaises(SystemExit):
                screener.load_kospi_all(refresh=True)
        self.assertFalse(self.cache.exists())

    def test_refresh_failure_falls_back_to_cache(self):
        pd.DataFrame([{"ticker": "005930", "name": "삼성전자", "sector": "제조업"}]).to_csv(
            self.cache, index=False)
        with patch.object(screener, "_http_get",
                          side_effect=RuntimeError("network unavailable")):
            result = screener.load_kospi_all(refresh=True)
        self.assertEqual(result["ticker"].tolist(), ["005930"])

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

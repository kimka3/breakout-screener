"""Offline coverage regressions for the primary constituent source parser."""
from __future__ import annotations

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


class ConstituentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="constituents-regression-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legacy = self.root / "kospi200_constituents.csv"
        self.cache = self.root / "kospi200_constituents_v2.csv"
        self.patch_cache = patch.object(screener, "CACHE_DIR", self.root)
        self.patch_path = patch.object(screener, "KOSPI200_CSV", self.legacy)
        self.patch_cache.start()
        self.patch_path.start()
        self.addCleanup(self.patch_path.stop)
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


if __name__ == "__main__":
    unittest.main()

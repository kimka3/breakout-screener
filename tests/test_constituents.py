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


def kosdaq150_page(*, fund_code="316670", title="KIWOOM 코스닥150",
                   index_name="KOSDAQ150지수", basis_date="2026-10-28"):
    """The official fund page supplies identity and the selected holdings date."""
    return (
        f"<!doctype html><html><head><title>KIWOOM ETF | {title}</title></head>"
        "<body><script>"
        "function fnCommonChart() { var fundCode = '3149'; }"
        "function fnCommonPerformance() { var fundCode = '3149'; }"
        f"function fnPdfDate() {{ var fundCode = '{fund_code}'; "
        "var pdfDt = document.getElementById('pdfDt').value; }"
        "</script>"
        f"<h1>{title}</h1><dl><dt>기초지수</dt><dd>{index_name}</dd></dl>"
        f'<input id="pdfDt" type="date" value="{basis_date}"></body></html>'
    )


def kosdaq150_holdings(stock_count=150):
    """A complete PDF basket includes cash and ISIN-coded stock holdings."""
    tickers = [f"{i:06d}" for i in range(1, stock_count + 1)]
    if tickers:
        tickers[0] = "0126Z0"
    holdings = [{
        "gcode": "KRD010010001", "itemCode": "KRD010010001",
        "itemTitle": "원화예금", "businessDate": "2026.10.28",
    }]
    for i, ticker in enumerate(tickers, start=1):
        isin = f"KR7{ticker}000"
        holdings.append({
            "gcode": isin, "itemCode": isin,
            "itemTitle": "영문코드 종목" if i == 1 else f"코스닥 종목 {i}",
            "businessDate": "2026.10.28",
        })
    return holdings


def kosdaq150_source(stock_count=150, *, holdings=None):
    return json.dumps({
        "pdfList": kosdaq150_holdings(stock_count) if holdings is None else holdings,
    }, ensure_ascii=False)


def kosdaq150_cache(stock_count=150):
    stocks = kosdaq150_holdings(stock_count)[1:]
    return pd.DataFrame({
        "ticker": [row["gcode"][3:9] for row in stocks],
        "name": [row["itemTitle"] for row in stocks],
        "sector": [""] * stock_count,
    })


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

    def assert_kosdaq150_rejected(self, *, page=None, body=None):
        with patch.object(screener, "_http_get", return_value=page or kosdaq150_page()), \
             patch.object(screener, "_http_post",
                          return_value=kosdaq150_source() if body is None else body), \
             self.assertRaises(SystemExit):
            screener.load_kosdaq150(refresh=True)
        self.assertFalse(self.kosdaq_cache.exists())

    def test_kosdaq150_uses_all_150_stock_rows_and_excludes_cash(self):
        with patch.object(screener, "_http_get", return_value=kosdaq150_page()) as fetch, \
             patch.object(screener, "_http_post", return_value=kosdaq150_source()) as post:
            result = screener.load_kosdaq150(refresh=True)
        self.assertEqual(len(result), 150)
        self.assertEqual(result.iloc[0]["ticker"], "0126Z0")
        self.assertEqual(result.iloc[0]["name"], "영문코드 종목")
        self.assertNotIn("KRD010010001", result["ticker"].tolist())
        self.assertEqual(result["ticker"].nunique(), 150)
        self.assertEqual(screener.MARKETS["kosdaq150"]["to_yahoo"]("0126Z0"), "0126Z0.KQ")
        self.assertTrue(self.kosdaq_cache.exists())
        with patch.object(screener, "_http_get") as cached_fetch, \
             patch.object(screener, "_http_post") as cached_post:
            cached = screener.load_kosdaq150()
        self.assertEqual(cached["ticker"].tolist(), result["ticker"].tolist())
        cached_fetch.assert_not_called()
        cached_post.assert_not_called()
        fetch.assert_called_once_with(screener.KIWOOM_KOSDAQ150)
        post.assert_called_once_with(
            screener.KIWOOM_KOSDAQ150_PDF,
            {"schGubun1": "316670", "startDate": "20261028"},
            referer=screener.KIWOOM_KOSDAQ150,
        )

    def test_kosdaq150_item_code_is_optional_when_gcode_is_valid(self):
        holdings = kosdaq150_holdings()
        for row in holdings:
            row.pop("itemCode")
        with patch.object(screener, "_http_get", return_value=kosdaq150_page()), \
             patch.object(screener, "_http_post", return_value=kosdaq150_source(holdings=holdings)):
            result = screener.load_kosdaq150(refresh=True)
        self.assertEqual(len(result), 150)
        self.assertEqual(result["ticker"].nunique(), 150)

    def test_kosdaq150_preview_or_wrong_count_is_never_cached(self):
        # A 30-row cash-plus-29-stock preview is not the index universe.
        for count in (29, 149, 151):
            with self.subTest(stock_count=count):
                self.assert_kosdaq150_rejected(body=kosdaq150_source(count))

    def test_kosdaq150_duplicate_stock_rows_are_never_silently_deduplicated(self):
        replaced = kosdaq150_holdings()
        replaced[-1] = dict(replaced[1])
        appended = kosdaq150_holdings()
        appended.append(dict(appended[1]))
        for holdings in (replaced, appended):
            with self.subTest(row_count=len(holdings)):
                self.assert_kosdaq150_rejected(body=kosdaq150_source(holdings=holdings))

    def test_kosdaq150_rejects_the_wrong_fund_page_before_requesting_holdings(self):
        pages = {
            "fund_code": kosdaq150_page(fund_code="305720"),
            "correct_code_in_unrelated_function": kosdaq150_page(fund_code="305720").replace(
                "var fundCode = '3149';", "var fundCode = '316670';", 1),
            "title": kosdaq150_page(title="KIWOOM 코스닥150선물레버리지"),
            "index": kosdaq150_page(index_name="KOSDAQ150선물지수"),
        }
        for label, page in pages.items():
            with self.subTest(identity=label), \
                 patch.object(screener, "_http_get", return_value=page), \
                 patch.object(screener, "_http_post") as post:
                with self.assertRaises(SystemExit):
                    screener.load_kosdaq150(refresh=True)
                post.assert_not_called()
                self.assertFalse(self.kosdaq_cache.exists())

    def test_kosdaq150_requires_a_real_selected_page_date(self):
        for value in ("", "20261028", "2026-02-30", "2026-13-01", "not-a-date"):
            with self.subTest(basis_date=value), \
                 patch.object(screener, "_http_get",
                              return_value=kosdaq150_page(basis_date=value)), \
                 patch.object(screener, "_http_post") as post:
                with self.assertRaises(SystemExit):
                    screener.load_kosdaq150(refresh=True)
                post.assert_not_called()
                self.assertFalse(self.kosdaq_cache.exists())

    def test_kosdaq150_rejects_missing_or_non_list_holdings(self):
        for payload in (None, [], {}, {"pdfList": None}, {"pdfList": {}},
                        {"pdfList": "not a list"}):
            with self.subTest(payload=payload):
                self.assert_kosdaq150_rejected(body=json.dumps(payload))
        self.assert_kosdaq150_rejected(body="<html>upstream unavailable</html>")

    def test_kosdaq150_rejects_unknown_rows_even_with_150_valid_stocks(self):
        for extra in (None, [], "not a holding", {},
                      {"gcode": "KRD010010002", "itemTitle": "다른 현금성 자산",
                       "businessDate": "2026.10.28"}):
            with self.subTest(extra=extra):
                self.assert_kosdaq150_rejected(
                    body=kosdaq150_source(holdings=[*kosdaq150_holdings(), extra]))

    def test_kosdaq150_rejects_malformed_or_conflicting_stock_codes(self):
        changes = (
            {"gcode": None}, {"gcode": "KR7SHORT000"},
            {"gcode": "US7000001000"}, {"gcode": "KR700000!000"},
            {"gcode": "KR7000001ABC"}, {"itemCode": "KR7000002000"},
        )
        for change in changes:
            with self.subTest(change=change):
                holdings = kosdaq150_holdings()
                holdings[1].update(change)
                self.assert_kosdaq150_rejected(body=kosdaq150_source(holdings=holdings))

    def test_kosdaq150_rejects_empty_or_non_string_stock_names(self):
        for value in (None, "", "  ", 123, [], {}):
            with self.subTest(name=value):
                holdings = kosdaq150_holdings()
                holdings[1]["itemTitle"] = value
                self.assert_kosdaq150_rejected(body=kosdaq150_source(holdings=holdings))

    def test_kosdaq150_requires_all_stock_dates_to_match_the_page(self):
        for value in (None, "", "2026.10.27", "2026.02.30", "20261028", 20261028):
            with self.subTest(business_date=value):
                holdings = kosdaq150_holdings()
                holdings[1]["businessDate"] = value
                self.assert_kosdaq150_rejected(body=kosdaq150_source(holdings=holdings))

    def test_kosdaq150_invalid_fresh_cache_is_refetched(self):
        kosdaq150_cache(29).to_csv(self.kosdaq_cache, index=False)
        with patch.object(screener, "_http_get", return_value=kosdaq150_page()) as fetch, \
             patch.object(screener, "_http_post", return_value=kosdaq150_source()) as post:
            result = screener.load_kosdaq150()
        self.assertEqual(len(result), 150)
        fetch.assert_called_once()
        post.assert_called_once()
        cached = pd.read_csv(self.kosdaq_cache, dtype={"ticker": str})
        self.assertEqual(cached["ticker"].nunique(), 150)

    def test_kosdaq150_network_failure_never_uses_an_invalid_stale_cache(self):
        duplicate = kosdaq150_cache()
        duplicate.loc[149, "ticker"] = duplicate.loc[0, "ticker"]
        missing_name = kosdaq150_cache()
        missing_name.loc[0, "name"] = ""
        invalid_code = kosdaq150_cache()
        invalid_code.loc[0, "ticker"] = "not-a-stock"
        for invalid in (kosdaq150_cache(29), kosdaq150_cache(149), kosdaq150_cache(151),
                        duplicate, missing_name, invalid_code,
                        kosdaq150_cache().drop(columns="name")):
            with self.subTest(rows=len(invalid), columns=list(invalid.columns)):
                invalid.to_csv(self.kosdaq_cache, index=False)
                os.utime(self.kosdaq_cache, (0, 0))
                with patch.object(screener, "_http_get", side_effect=RuntimeError("offline")), \
                     self.assertRaises(SystemExit):
                    screener.load_kosdaq150(refresh=True)

    def test_kosdaq150_refresh_failure_uses_only_a_valid_stale_cache(self):
        kosdaq150_cache().to_csv(self.kosdaq_cache, index=False)
        os.utime(self.kosdaq_cache, (0, 0))
        original = self.kosdaq_cache.read_bytes()
        with patch.object(screener, "_http_get", side_effect=RuntimeError("network unavailable")):
            result = screener.load_kosdaq150(refresh=True)
        self.assertEqual(len(result), 150)
        self.assertEqual(result.iloc[0]["ticker"], "0126Z0")
        self.assertEqual(self.kosdaq_cache.read_bytes(), original)

    def test_kosdaq150_failed_refresh_preserves_a_valid_cache(self):
        kosdaq150_cache().to_csv(self.kosdaq_cache, index=False)
        original = self.kosdaq_cache.read_bytes()
        for body in (kosdaq150_source(29), "invalid JSON"):
            with self.subTest(body=body[:30]), \
                 patch.object(screener, "_http_get", return_value=kosdaq150_page()), \
                 patch.object(screener, "_http_post", return_value=body):
                result = screener.load_kosdaq150(refresh=True)
            self.assertEqual(result["ticker"].nunique(), 150)
            self.assertEqual(self.kosdaq_cache.read_bytes(), original)

    def test_kosdaq150_unreadable_cache_does_not_hide_refresh_failure(self):
        self.kosdaq_cache.write_bytes(b"\xff\xfe\x00not a CSV cache")
        with patch.object(screener, "_http_get", side_effect=RuntimeError("offline")), \
             self.assertRaises(SystemExit):
            screener.load_kosdaq150()


if __name__ == "__main__":
    unittest.main()

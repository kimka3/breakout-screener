"""Offline regressions: real calculations/writers, deterministic market responses.

Run from the repository root: python -m unittest discover -s tests -v
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import certifi
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Keep the real module import while preventing its Windows CA export side effect.
with patch.dict(os.environ, {"SSL_CERT_FILE": certifi.where()}):
    import sp500_breakout as screener


def price_frame(prices, volumes=None, factors=None):
    """Create a complete daily feed, with optional time-varying adjustment factors."""
    index = pd.bdate_range("2026-01-01", periods=len(prices))
    close = pd.Series(prices, index=index, dtype=float)
    factor = pd.Series(factors if factors is not None else 1.0, index=index)
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Adj Close": close * factor,
        "Volume": volumes if volumes is not None else [100.0] * len(index),
    })


def report_payload(path):
    """Decode the injected JSON without executing browser scripts."""
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    html = path.read_text(encoding="utf-8")
    start = html.index("const DATA =") + len("const DATA =")
    return json.JSONDecoder(parse_constant=reject_nonfinite).raw_decode(
        html[start:].lstrip())[0]


class SignalTests(unittest.TestCase):
    def signals(self, frame, basis="adj", hold=True):
        return screener.find_signals(frame, 3, 2, 2.0, basis, require_hold=hold)

    def test_exact_volume_threshold_and_prior_ma_equality_cross(self):
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, 200])
        hits = self.signals(frame)
        self.assertEqual(len(hits), 1)
        hit = hits.iloc[0]
        self.assertEqual(hit["date"], frame.index[-1])
        self.assertEqual(hit["avg_vol"], 100)
        self.assertEqual(hit["vol_ratio"], 2)
        self.assertAlmostEqual(hit["ma"], 32 / 3)
        self.assertAlmostEqual(hit["above_ma_pct"], 12.5)
        self.assertEqual(hit["bars_since"], 0)

    def test_remaining_above_ma_without_cross_is_not_a_signal(self):
        frame = price_frame([10, 11, 12, 13], [100, 100, 100, 1000])
        self.assertTrue(self.signals(frame).empty)

    def test_hold_uses_last_bar_and_allows_intermediate_breach(self):
        frame = price_frame([10, 10, 10, 12, 9, 13],
                            [100, 100, 100, 200, 100, 500])
        hits = self.signals(frame)
        self.assertEqual(hits["date"].tolist(), [frame.index[3], frame.index[5]])
        self.assertEqual(hits["bars_since"].tolist(), [2, 0])
        self.assertTrue(hits["holding"].all())
        fallen = frame.iloc[:-1]
        self.assertTrue(self.signals(fallen).empty)
        self.assertEqual(len(self.signals(fallen, hold=False)), 1)

    def test_zero_or_nonfinite_volume_cannot_create_a_signal(self):
        for volumes in ([0, 0, 0, 200], [100, 100, 100, float("inf")],
                        [100, 100, 100, float("nan")]):
            with self.subTest(volumes=volumes):
                self.assertTrue(self.signals(price_frame([10, 10, 10, 12], volumes)).empty)

    def test_adjusted_mode_requires_adjusted_close(self):
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, 200]).drop(
            columns="Adj Close")
        with self.assertRaisesRegex(ValueError, "Adj Close"):
            self.signals(frame)
        self.assertEqual(len(self.signals(frame, basis="raw")), 1)

    def test_invalid_latest_price_cannot_export_old_signal_even_without_hold(self):
        for value in (0, float("nan"), float("inf")):
            with self.subTest(value=value):
                frame = price_frame([10, 10, 10, 12, value],
                                    [100, 100, 100, 200, 100])
                self.assertTrue(self.signals(frame, hold=False).empty)


class ChartTests(unittest.TestCase):
    def test_adjusted_candles_and_ma_share_a_basis_without_changing_raw_quotes(self):
        # Changing factors catch both raw-candle mixing and a single-factor shortcut.
        frame = price_frame([20, 20, 20, 24], [100, 100, 100, 200],
                            factors=[0.5, 0.5, 0.45, 0.5])
        hit = screener.find_signals(frame, 3, 2, 2, "adj").iloc[0]
        chart = screener.build_series(frame, hit, SimpleNamespace(ma=3, price_basis="adj"),
                                      "TEST", "Test company", "Industrials")
        self.assertEqual(chart["close"], 24)
        self.assertEqual(chart["lastClose"], 24)
        self.assertEqual(hit["signal_close"], 12)
        self.assertEqual(hit["last_signal_close"], 12)
        self.assertEqual(chart["signalClose"], 12)
        self.assertEqual(chart["lastSignalClose"], 12)
        factor = frame["Adj Close"] / frame["Close"]
        for source, target in (("Open", "opens"), ("High", "highs"),
                               ("Low", "lows"), ("Close", "closes")):
            with self.subTest(field=target):
                self.assertEqual(chart[target], (frame[source] * factor).round(4).tolist())
        self.assertAlmostEqual(chart["abovePct"],
                               (chart["signalClose"] / chart["ma"] - 1) * 100,
                               delta=0.05)

    def test_raw_mode_keeps_raw_candles_and_signal_quotes(self):
        frame = price_frame([20, 20, 20, 24], [100, 100, 100, 200],
                            factors=[0.5] * 4)
        hit = screener.find_signals(frame, 3, 2, 2, "raw").iloc[0]
        chart = screener.build_series(frame, hit, SimpleNamespace(ma=3, price_basis="raw"),
                                      "TEST", "", "")
        self.assertEqual(chart["closes"], frame["Close"].tolist())
        self.assertEqual(chart["signalClose"], chart["close"])
        self.assertEqual(chart["lastSignalClose"], chart["lastClose"])

    def test_chart_retains_old_signal_and_latest_bar(self):
        frame = price_frame([10, 10, 10, 12] + list(range(13, 93)),
                            [100, 100, 100, 200] + [100] * 80)
        hit = screener.find_signals(frame, 3, 2, 2, "adj").iloc[0]
        chart = screener.build_series(frame, hit, SimpleNamespace(ma=3, price_basis="adj"),
                                      "TEST", "", "")
        self.assertGreaterEqual(len(chart["dates"]), 70)
        self.assertGreaterEqual(chart["sigIndex"], 0)
        self.assertLess(chart["sigIndex"], len(chart["dates"]))
        self.assertEqual(chart["dates"][chart["sigIndex"]], chart["date"])
        self.assertEqual(chart["dates"][-1], frame.index[-1].date().isoformat())
        for field in ("opens", "highs", "lows", "closes", "mas", "volumes"):
            self.assertEqual(len(chart[field]), len(chart["dates"]))

    def test_chart_normalizes_missing_name_and_sector_to_empty_strings(self):
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, 200])
        hit = screener.find_signals(frame, 3, 2, 2, "adj").iloc[0]
        chart = screener.build_series(frame, hit, SimpleNamespace(ma=3, price_basis="adj"),
                                      "005930", float("nan"), float("nan"), "kospi200")
        self.assertEqual(chart["name"], "")
        self.assertEqual(chart["sector"], "")
        json.dumps(chart, allow_nan=False)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="screener-regression-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / "breakouts.csv"
        self.html = self.root / "report.html"
        self.summary = self.root / "summary.txt"

    def run_main(self, prices, extra=()):
        argv = ["sp500_breakout.py", "--market", "sp500", "--tickers", "TEST",
                "--ma", "3", "--vol-window", "2", "--lookback", "5",
                "--date", "2026-08-31", "--out", str(self.csv),
                "--html", str(self.html), "--summary", str(self.summary), *extra]
        with patch.object(sys, "argv", argv), \
             patch.object(screener, "download_prices", return_value=prices) as download, \
             patch.object(screener, "fetch_fundamentals", return_value={}) as fundamentals, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            download.last_failed = []
            result = screener.main()
        return result, fundamentals

    def test_zero_hits_overwrites_old_csv_with_headers_and_produces_empty_report(self):
        self.csv.write_text("stale,record\nDO_NOT_KEEP,123\n", encoding="utf-8")
        code, fundamentals = self.run_main({"TEST": price_frame([10] * 10)})
        self.assertEqual(code, 0)
        self.assertTrue(self.csv.exists())
        csv = pd.read_csv(self.csv)
        self.assertTrue(csv.empty)
        self.assertTrue({"market", "date", "ticker", "close", "ma3", "vol_ratio",
                         "last_close", "bars_since"}.issubset(csv.columns))
        self.assertNotIn("DO_NOT_KEEP", self.csv.read_text(encoding="utf-8-sig"))
        self.assertEqual(report_payload(self.html)["hits"], [])
        self.assertTrue(self.summary.exists())
        fundamentals.assert_not_called()

    def test_all_downloads_missing_returns_failure_without_success_outputs(self):
        code, fundamentals = self.run_main({})
        self.assertNotEqual(code, 0)
        self.assertFalse(self.html.exists())
        self.assertFalse(self.summary.exists())
        fundamentals.assert_not_called()

    def test_all_histories_too_short_are_not_reported_as_a_successful_empty_scan(self):
        code, _ = self.run_main({"TEST": price_frame([10, 11, 12])})
        self.assertNotEqual(code, 0)
        self.assertFalse(self.html.exists())
        self.assertFalse(self.summary.exists())

    def test_valid_latest_quote_without_evaluable_adjusted_history_is_a_failure(self):
        frame = price_frame([10] * 10)
        frame.loc[frame.index[:-1], "Adj Close"] = float("nan")
        self.assertEqual(frame.iloc[-1]["Adj Close"], 10)
        self.assertEqual(frame.iloc[-1]["Close"], 10)
        code, fundamentals = self.run_main({"TEST": frame})
        self.assertNotEqual(code, 0)
        self.assertFalse(self.html.exists())
        self.assertFalse(self.summary.exists())
        fundamentals.assert_not_called()

    def test_valid_prices_without_evaluable_volume_history_are_a_failure(self):
        frame = price_frame([10] * 10, [float("nan")] * 10)
        self.assertEqual(frame.iloc[-1]["Adj Close"], 10)
        self.assertEqual(frame.iloc[-1]["Close"], 10)
        code, fundamentals = self.run_main({"TEST": frame})
        self.assertNotEqual(code, 0)
        self.assertFalse(self.html.exists())
        self.assertFalse(self.summary.exists())
        fundamentals.assert_not_called()

    def test_kospi_cached_blank_sector_can_use_empty_or_fundamental_fallback(self):
        universe_path = self.root / "kospi200_constituents.csv"
        pd.DataFrame({"ticker": ["005930"], "name": ["삼성전자"], "sector": [""]}).to_csv(
            universe_path, index=False)
        cached_meta = pd.read_csv(universe_path, dtype={"ticker": str})
        self.assertTrue(pd.isna(cached_meta.iloc[0]["sector"]))
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, 200])
        argv = ["sp500_breakout.py", "--market", "kospi200", "--ma", "3",
                "--vol-window", "2", "--lookback", "1", "--date", "2026-08-31",
                "--out", str(self.csv), "--html", str(self.html),
                "--summary", str(self.summary)]
        for fallback in ("", "Technology"):
            with self.subTest(fallback=fallback), \
                 patch.object(sys, "argv", argv), \
                 patch.dict(screener.MARKETS["kospi200"],
                            {"loader": lambda **_kwargs: cached_meta.copy()}), \
                 patch.object(screener, "download_prices", return_value={"005930.KS": frame}), \
                 patch.object(screener, "fetch_fundamentals",
                              return_value={"005930.KS": {"sector": fallback}}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(screener.main(), 0)
                result = pd.read_csv(self.csv, dtype={"ticker": str}, keep_default_na=False)
                self.assertEqual(result.iloc[0]["ticker"], "005930")
                self.assertEqual(result.iloc[0]["sector"], fallback)
                payload = report_payload(self.html)
                self.assertEqual(payload["hits"][0]["name"], "삼성전자")
                self.assertEqual(payload["hits"][0]["sector"], fallback)
                self.assertEqual(payload["hits"][0]["market"], "kospi200")

    def test_main_applies_lookback_and_keeps_raw_output_quotes(self):
        frame = price_frame([20, 20, 20, 24, 18, 26],
                            [100, 100, 100, 200, 100, 500], factors=[0.5] * 6)
        code, fundamentals = self.run_main({"TEST": frame}, extra=("--lookback", "1"))
        self.assertEqual(code, 0)
        csv = pd.read_csv(self.csv)
        self.assertEqual(len(csv), 1)
        self.assertEqual(csv.iloc[0]["close"], 26)
        self.assertEqual(csv.iloc[0]["date"], frame.index[-1].date().isoformat())
        payload = report_payload(self.html)
        self.assertEqual(len(payload["hits"]), 1)
        self.assertEqual(payload["hits"][0]["signalClose"], 13)
        self.assertEqual(payload["historicalAsOf"], "2026-08-31")
        self.assertIsInstance(payload["fundamentalsAsOf"], str)
        self.assertTrue(payload["fundamentalsAsOf"])
        self.assertNotEqual(payload["fundamentalsAsOf"], payload["historicalAsOf"])
        fundamentals.assert_called_once_with(["TEST"])

    def test_report_preserves_source_metadata_and_escapes_script_terminators(self):
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, 200])
        hit = screener.find_signals(frame, 3, 2, 2, "raw").iloc[0]
        args = SimpleNamespace(ma=3, vol_window=2, vol_mult=2, lookback=1,
                               price_basis="raw", no_hold=False, date="2026-08-31",
                               fundamentals_as_of="2026-09-13 12:00 KST")
        attack = '</script><script>alert("external metadata")</script>'
        chart = screener.build_series(frame, hit, args, "TEST", attack, "")
        with redirect_stdout(io.StringIO()):
            screener.write_html([chart], args, [{"id": "sp500"}], self.html)
        payload = report_payload(self.html)
        self.assertEqual(payload["hits"][0]["name"], attack)
        self.assertEqual(payload["historicalAsOf"], args.date)
        self.assertEqual(payload["fundamentalsAsOf"], args.fundamentals_as_of)
        for path in (self.html, self.html.with_suffix(".artifact.html")):
            self.assertNotIn("</script><script>", path.read_text(encoding="utf-8"))

    def test_nonpositive_or_nonfinite_numeric_options_are_rejected_before_download(self):
        for option, value in (("--ma", "0"), ("--ma", "-1"),
                              ("--vol-window", "0"), ("--lookback", "0"),
                              ("--lookback", "-1"), ("--vol-mult", "0"),
                              ("--vol-mult", "nan"), ("--vol-mult", "inf")):
            with self.subTest(option=option, value=value), \
                 patch.object(sys, "argv", ["sp500_breakout.py", "--market", "sp500",
                                            "--tickers", "TEST", "--out", str(self.csv),
                                            option, value]), \
                 patch.object(screener, "download_prices") as download, \
                 redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    screener.main()
                self.assertNotEqual(raised.exception.code, 0)
                download.assert_not_called()

    def test_explicit_tickers_require_an_explicit_market(self):
        with patch.object(sys, "argv", ["sp500_breakout.py", "--tickers", "TEST"]), \
             patch.object(screener, "download_prices") as download, \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                screener.main()
            self.assertNotEqual(raised.exception.code, 0)
            download.assert_not_called()


class CacheTests(unittest.TestCase):
    def test_downloader_preserves_price_rows_with_missing_volume(self):
        frame = price_frame([10, 10, 10, 12], [100, 100, 100, float("nan")])
        with patch.object(screener.yf, "download", return_value=pd.concat({"TEST": frame}, axis=1)), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            prices = screener.download_prices(["TEST"], date(2026, 1, 1), date(2026, 1, 30), use_cache=False)
        pd.testing.assert_frame_equal(prices["TEST"], frame)
        # The original daily volume filter still rejects an unevaluable breakout.
        self.assertTrue(screener.find_signals(prices["TEST"], 3, 2, 2, "adj").empty)

    def test_cache_hit_reconstructs_failed_tickers_without_downloading(self):
        frame = price_frame([10] * 10)
        missing = frame.iloc[:0]

        def downloaded(tickers, **_kwargs):
            names = [tickers] if isinstance(tickers, str) else tickers
            return pd.concat({"TEST": frame}, axis=1) if "TEST" in names else missing

        previous = getattr(screener.download_prices, "last_failed", None)
        self.addCleanup(setattr, screener.download_prices, "last_failed", previous)
        with tempfile.TemporaryDirectory(prefix="screener-cache-") as tmp, \
             patch.object(screener, "CACHE_DIR", Path(tmp)), \
             patch.object(screener.yf, "download", side_effect=downloaded) as download, \
             patch.object(screener.time, "sleep"), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            start, end = date(2026, 1, 1), date(2026, 1, 30)
            first = screener.download_prices(["TEST", "MISSING"], start, end)
            self.assertEqual(set(first), {"TEST"})
            self.assertEqual(screener.download_prices.last_failed, ["MISSING"])
            screener.download_prices.last_failed = []
            download.reset_mock()
            cached = screener.download_prices(["TEST", "MISSING"], start, end)
            self.assertEqual(set(cached), {"TEST"})
            self.assertEqual(screener.download_prices.last_failed, ["MISSING"])
            pd.testing.assert_frame_equal(cached["TEST"], frame)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()

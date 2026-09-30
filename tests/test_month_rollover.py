"""Offline month rollover regressions: exchange close, data receipt, publication."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from test_screener import report_payload, screener
import market_sessions
import monthly_breakout as monthly


def history(target="2026-09"):
    """Sixteen peer-observed months satisfying the unmodified monthly predicate."""
    periods = pd.period_range(end=target, periods=16, freq="M")
    pieces = []
    for i, month in enumerate(periods):
        days = pd.bdate_range(month.start_time, month.end_time.normalize())
        close = 100.0 if i < 9 else (90.0 if i < 15 else 110.0)
        total = 100 if i < 15 else 200
        pieces.append(pd.DataFrame({
            "Open": close - .5, "High": close + 1, "Low": close - 1,
            "Close": close, "Adj Close": close,
            "Volume": [total - len(days) + 1] + [1] * (len(days) - 1),
        }, index=days))
    return pd.concat(pieces)


class ExchangeMonthTests(unittest.TestCase):
    def assert_month(self, market, instant, month, final_session):
        result, endpoint = market_sessions.completed_month(market, pd.Timestamp(instant))
        self.assertEqual(result, pd.Period(month, freq="M"))
        self.assertEqual(endpoint, pd.Timestamp(final_session))
        self.assertIsNone(endpoint.tzinfo)

    def test_september_us_before_regular_close_keeps_august(self):
        self.assert_month("sp500", "2026-09-30T19:59:59Z", "2026-08", "2026-08-31")

    def test_september_us_at_regular_close_rolls_without_waiting_for_midnight(self):
        self.assert_month("sp500", "2026-09-30T20:00:00Z", "2026-09", "2026-09-30")

    def test_october_first_vultr_run_uses_september_for_every_market(self):
        for market in ("sp500", "kospi", "kosdaq150"):
            with self.subTest(market=market):
                self.assert_month(market, "2026-10-01T07:20:00+09:00",
                                  "2026-09", "2026-09-30")

    def test_middle_of_month_never_promotes_an_unfinished_month(self):
        for market in ("sp500", "kospi", "kosdaq150"):
            with self.subTest(market=market):
                self.assert_month(market, "2026-09-15T22:20:00Z",
                                  "2026-08", "2026-08-31")

    def test_korean_close_boundary_is_1530_local(self):
        for market in ("kospi", "kosdaq150"):
            with self.subTest(market=market):
                self.assert_month(market, "2026-09-30T15:29:59+09:00",
                                  "2026-08", "2026-08-31")
                self.assert_month(market, "2026-09-30T15:30:00+09:00",
                                  "2026-09", "2026-09-30")

    def test_us_november_weekend_month_end_uses_friday_early_close(self):
        self.assert_month("sp500", "2025-11-28T17:59:59Z", "2025-10", "2025-10-31")
        self.assert_month("sp500", "2025-11-28T18:00:00Z", "2025-11", "2025-11-28")
        self.assert_month("sp500", "2025-11-30T12:00:00Z", "2025-11", "2025-11-28")
        self.assertEqual(market_sessions.session_close("sp500", "2025-11-28"),
                         pd.Timestamp("2025-11-28T18:00:00Z"))

    def test_us_good_friday_and_weekend_at_month_end_do_not_delay_rollover(self):
        self.assert_month("sp500", "2024-03-28T19:59:59Z", "2024-02", "2024-02-29")
        self.assert_month("sp500", "2024-03-28T20:00:00Z", "2024-03", "2024-03-28")
        self.assertIsNone(market_sessions.session_close("sp500", "2024-03-29"))
        self.assertIsNone(market_sessions.session_close("sp500", "2024-03-31"))

    def test_korean_december_31_exchange_closure_uses_december_30(self):
        for market in ("kospi", "kosdaq150"):
            with self.subTest(market=market):
                self.assert_month(market, "2025-12-30T15:29:59+09:00",
                                  "2025-11", "2025-11-28")
                self.assert_month(market, "2025-12-30T15:30:00+09:00",
                                  "2025-12", "2025-12-30")
                self.assertIsNone(market_sessions.session_close(market, "2025-12-31"))

    def test_live_month_requires_an_unambiguous_timezone_aware_instant(self):
        for instant in ("2026-09-30", pd.Timestamp("2026-09-30T20:00:00"), pd.NaT):
            with self.subTest(instant=instant), self.assertRaises(ValueError):
                market_sessions.completed_month("sp500", instant)

    def test_historical_date_mode_keeps_previous_calendar_month_convention(self):
        for as_of, expected in (("2026-09-30", "2026-08"),
                                ("2025-11-28", "2025-10"),
                                ("2026-10-01", "2026-09")):
            with self.subTest(as_of=as_of):
                self.assertEqual(str(monthly.last_completed_month(as_of)), expected)


class FinalSessionQualityTests(unittest.TestCase):
    def scan(self, prices):
        meta = pd.DataFrame({"ticker": list(prices), "name": "Fixture", "sector": ""})
        return monthly.scan_monthly_market(
            prices, meta, "sp500", "2026-09-30T18:20:00-04:00",
            target_month=pd.Period("2026-09", "M"),
            expected_last_session=pd.Timestamp("2026-09-30"))

    def test_all_feeds_missing_final_session_are_degraded_and_do_not_emit_old_signals(self):
        frame = history().iloc[:-1]
        rows, charts, summary = self.scan({"T0": frame, "T1": frame.copy()})
        self.assertEqual((rows, charts), ([], []))
        self.assertEqual(summary["targetMonth"], "2026-09")
        self.assertEqual(summary["expectedLastTradingDate"], "2026-09-30")
        self.assertEqual(summary["scanned"], 0)
        self.assertEqual(summary["gapped"], 2)
        self.assertEqual(summary["stale"], 2)
        self.assertTrue(summary["degraded"])

    def test_a_complete_received_final_session_accepts_the_same_frozen_predicate(self):
        frame = history()
        rows, charts, summary = self.scan({"T0": frame, "T1": frame.copy()})
        self.assertEqual(summary["targetMonth"], "2026-09")
        self.assertEqual(summary["lastTradingDate"], "2026-09-30")
        self.assertEqual(summary["expectedLastTradingDate"], "2026-09-30")
        self.assertEqual(summary["scanned"], 2)
        self.assertEqual(summary["hits"], 2)
        self.assertFalse(summary["degraded"])
        self.assertEqual({row["date"] for row in rows}, {"2026-09-30"})
        self.assertEqual({chart["date"] for chart in charts}, {"2026-09-30"})

    def test_one_missing_final_session_is_excluded_even_when_peers_are_current(self):
        complete = history()
        prices = {f"T{i}": complete.copy() for i in range(5)}
        prices["T0"] = complete.iloc[:-1]
        rows, _, summary = self.scan(prices)
        self.assertEqual({row["ticker"] for row in rows}, {"T1", "T2", "T3", "T4"})
        self.assertEqual(summary["gapped"], 1)
        self.assertEqual(summary["stale"], 1)
        self.assertEqual(summary["scanned"], 4)
        self.assertFalse(summary["degraded"])


class PartialBarTests(unittest.TestCase):
    def test_sole_intraday_bar_is_removed_and_complete_close_is_retained(self):
        frame = history().iloc[-1:]
        before = screener.drop_partial_bar(frame, "sp500", as_of=pd.Timestamp("2026-09-30T19:59:59Z"))
        after = screener.drop_partial_bar(frame, "sp500", as_of=pd.Timestamp("2026-09-30T20:00:00Z"))
        self.assertTrue(before.empty)
        pd.testing.assert_frame_equal(after, frame)

    def test_early_close_bar_is_retained_after_the_actual_13_hour_close(self):
        frame = history().iloc[-1:].copy()
        frame.index = pd.DatetimeIndex(["2025-11-28"])
        before = screener.drop_partial_bar(frame, "sp500", as_of=pd.Timestamp("2025-11-28T17:59:59Z"))
        after = screener.drop_partial_bar(frame, "sp500", as_of=pd.Timestamp("2025-11-28T18:00:00Z"))
        self.assertTrue(before.empty)
        pd.testing.assert_frame_equal(after, frame)

    def test_prior_completed_bar_survives_during_the_next_session(self):
        frame = history().iloc[-2:-1]
        result = screener.drop_partial_bar(frame, "sp500", as_of=pd.Timestamp("2026-09-30T15:00:00Z"))
        pd.testing.assert_frame_equal(result, frame)


class RequiredEndRecoveryTests(unittest.TestCase):
    def recover(self, fresh):
        complete = history().iloc[-8:]
        prices = {"T0": complete.iloc[:-1], "T1": complete.iloc[:-1].copy()}
        with patch.object(screener, "download_prices", return_value=fresh) as retry, \
             redirect_stdout(io.StringIO()):
            result, audit = screener.recover_price_sessions(
                prices, "sp500", date(2026, 9, 1), date(2026, 10, 1), 3,
                required_end=pd.Timestamp("2026-09-30"),
                as_of=pd.Timestamp("2026-10-01T07:20:00+09:00"))
        retry.assert_called_once()
        self.assertFalse(retry.call_args.kwargs["use_cache"])
        self.assertFalse(retry.call_args.kwargs["threads"])
        self.assertEqual(audit["detected"], 2)
        self.assertEqual(audit["missingDates"], ["2026-09-30"])
        return result, audit

    def test_final_session_absent_from_every_feed_triggers_one_uncached_recovery(self):
        complete = history().iloc[-8:]
        result, audit = self.recover({"T0": complete, "T1": complete.copy()})
        self.assertEqual(audit["repaired"], 2)
        self.assertEqual(audit["remaining"], 0)
        for frame in result.values():
            self.assertIn(pd.Timestamp("2026-09-30"), frame.index)

    def test_failed_recovery_does_not_fabricate_a_final_session(self):
        result, audit = self.recover({})
        self.assertEqual(audit["repaired"], 0)
        self.assertEqual(audit["remaining"], 2)
        for frame in result.values():
            self.assertNotIn(pd.Timestamp("2026-09-30"), frame.index)

    def test_a_dated_null_final_row_is_not_accepted_as_recovered(self):
        complete = history().iloc[-8:].copy()
        complete.loc["2026-09-30", "Adj Close"] = float("nan")
        result, audit = self.recover({"T0": complete, "T1": complete.copy()})
        self.assertEqual(audit["repaired"], 0)
        self.assertEqual(audit["remaining"], 2)
        for frame in result.values():
            self.assertNotIn(pd.Timestamp("2026-09-30"), frame.index)

    def test_post_target_month_ipo_does_not_retry_a_prelisting_final_session(self):
        complete = history().iloc[-8:]
        ipo = complete.iloc[-1:].copy()
        ipo.index = pd.DatetimeIndex(["2026-10-01"])
        prices = {"ESTABLISHED": complete, "IPO": ipo}
        with patch.object(screener, "download_prices") as retry:
            result, audit = screener.recover_price_sessions(
                prices, "sp500", date(2026, 9, 1), date(2026, 10, 2), 3,
                required_end=pd.Timestamp("2026-09-30"),
                as_of=pd.Timestamp("2026-10-02T07:20:00+09:00"))
        retry.assert_not_called()
        self.assertEqual(audit["detected"], 0)
        self.assertEqual(audit["remaining"], 0)
        pd.testing.assert_frame_equal(result["IPO"], ipo)


class IntradayCacheTests(unittest.TestCase):
    def test_cache_stored_before_last_bar_close_is_rejected(self):
        frame = history().iloc[-2:]
        self.assertFalse(screener.price_cache_is_complete(
            {"TEST": frame}, "sp500", pd.Timestamp("2026-09-30T19:59:59Z")))

    def test_cache_stored_at_or_after_last_bar_close_is_accepted(self):
        frame = history().iloc[-2:]
        for instant in ("2026-09-30T20:00:00Z", "2026-09-30T22:20:00Z"):
            with self.subTest(instant=instant):
                self.assertTrue(screener.price_cache_is_complete(
                    {"TEST": frame}, "sp500", pd.Timestamp(instant)))

    def test_early_close_cache_is_accepted_at_13_edt_not_delayed_until_16(self):
        frame = history().iloc[-1:].copy()
        frame.index = pd.DatetimeIndex(["2025-11-28"])
        self.assertFalse(screener.price_cache_is_complete(
            {"TEST": frame}, "sp500", pd.Timestamp("2025-11-28T17:59:59Z")))
        self.assertTrue(screener.price_cache_is_complete(
            {"TEST": frame}, "sp500", pd.Timestamp("2025-11-28T18:00:00Z")))

    def test_naive_cache_timestamp_or_non_session_final_row_is_rejected(self):
        frame = history().iloc[-1:].copy()
        self.assertFalse(screener.price_cache_is_complete(
            {"TEST": frame}, "sp500", pd.Timestamp("2026-09-30T22:20:00")))
        frame.index = pd.DatetimeIndex(["2025-11-27"])
        self.assertFalse(screener.price_cache_is_complete(
            {"TEST": frame}, "sp500", pd.Timestamp("2025-11-28T18:00:00Z")))

    def download_with_cache(self, stored_at):
        cached = history().iloc[-2:].copy()
        fresh = cached.copy()
        fresh.loc["2026-09-30", "Close"] = 115.0
        fresh.loc["2026-09-30", "Adj Close"] = 115.0
        fresh.loc["2026-09-30", "Volume"] = 500
        stamp = pd.Timestamp(stored_at).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(screener, "CACHE_DIR", Path(tmp)), \
                 patch.object(screener.time, "time", return_value=pd.Timestamp("2026-09-30T20:05:00Z").timestamp()), \
                 patch.object(screener.time, "sleep") as sleep, \
                 patch.object(screener.yf, "download", return_value=fresh) as download, \
                 redirect_stdout(io.StringIO()):
                start, end = date(2026, 9, 1), date(2026, 10, 1)
                cache = screener.price_cache_path(["TEST"], start, end)
                cache.write_bytes(pickle.dumps({"TEST": cached}))
                os.utime(cache, (stamp, stamp))
                result = screener.download_prices(["TEST"], start, end, market="sp500")
        sleep.assert_not_called()
        return result, cached, fresh, download

    def test_recent_intraday_pickle_is_bypassed_and_yahoo_final_bar_is_returned(self):
        result, cached, fresh, download = self.download_with_cache("2026-09-30T19:59:00Z")
        download.assert_called_once()
        pd.testing.assert_frame_equal(result["TEST"], fresh)
        self.assertNotEqual(result["TEST"]["Volume"].iloc[-1], cached["Volume"].iloc[-1])

    def test_recent_post_close_pickle_is_used_without_a_network_download(self):
        result, cached, _, download = self.download_with_cache("2026-09-30T20:00:00Z")
        download.assert_not_called()
        pd.testing.assert_frame_equal(result["TEST"], cached)


class LiveCliPublicationTests(unittest.TestCase):
    def test_october_vultr_time_reaches_september_in_all_written_market_reports(self):
        instant = pd.Timestamp("2026-10-01T07:20:00+09:00").to_pydatetime()
        complete = history()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = lambda ticker: pd.DataFrame({"ticker": [ticker], "name": ["Fixture"], "sector": [""]})
            args = ["sp500_breakout.py", "--min-turnover", "0", "--with-monthly", "--no-cache",
                    "--ma", "3", "--out", str(root / "d.csv"),
                    "--monthly-out", str(root / "m.csv"), "--html", str(root / "r.html")]
            with patch.object(sys, "argv", args), \
                 patch.object(screener, "datetime", wraps=datetime) as clock, \
                 patch.dict(screener.MARKETS["sp500"], {"loader": lambda refresh: meta("TEST")}), \
                 patch.dict(screener.MARKETS["kospi"], {"loader": lambda refresh: meta("005930")}), \
                 patch.dict(screener.MARKETS["kosdaq150"], {"loader": lambda refresh: meta("196170")}), \
                 patch.object(screener, "download_prices", side_effect=[
                     {"TEST": complete}, {"005930.KS": complete}, {"196170.KQ": complete}]), \
                 patch.object(screener, "fetch_fundamentals", return_value={}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                clock.now.side_effect = lambda tz=None: instant.astimezone(tz)
                self.assertEqual(screener.main(), 0)
            payload = report_payload(root / "r.html")["screens"][1]
            self.assertEqual({market["id"]: market["targetMonth"] for market in payload["markets"]},
                             {"sp500": "2026-09", "kospi": "2026-09", "kosdaq150": "2026-09"})
            self.assertEqual(len(pd.read_csv(root / "m.csv")), 3)

    def test_unreceived_final_session_blocks_publication_and_preserves_previous_reports(self):
        instant = pd.Timestamp("2026-10-01T07:20:00+09:00").to_pydatetime()
        incomplete = history().iloc[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("d.csv", "m.csv", "r.html", "s.txt")]
            for path in paths:
                path.write_text("previous successful report", encoding="utf-8")
            args = ["sp500_breakout.py", "--market", "sp500", "--tickers", "TEST",
                    "--with-monthly", "--no-cache", "--ma", "3", "--out", str(paths[0]),
                    "--monthly-out", str(paths[1]), "--html", str(paths[2]), "--summary", str(paths[3])]
            with patch.object(sys, "argv", args), \
                 patch.object(screener, "datetime", wraps=datetime) as clock, \
                 patch.object(screener, "download_prices", side_effect=[{"TEST": incomplete}, {}]) as retry, \
                 patch.object(screener, "fetch_fundamentals", return_value={}) as funda, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                clock.now.side_effect = lambda tz=None: instant.astimezone(tz)
                self.assertEqual(screener.main(), 1)
            self.assertEqual(retry.call_count, 2)
            funda.assert_not_called()
            for path in paths:
                self.assertEqual(path.read_text(encoding="utf-8"), "previous successful report")


if __name__ == "__main__":
    unittest.main()

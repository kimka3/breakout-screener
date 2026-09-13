"""Both strategies travel through the real CLI and report writers independently."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from test_screener import report_payload, screener


def long_history():
    index = pd.bdate_range("2025-05-01", "2026-09-11")
    months = index.to_period("M")
    values = {m: 20 - i for i, m in enumerate(pd.period_range("2025-05", "2026-07", freq="M"))}
    values[pd.Period("2026-08", "M")] = 30
    values[pd.Period("2026-09", "M")] = 1  # No current-day hold condition for monthly.
    close = pd.Series([values[m] for m in months], index=index, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - .5,
                         "Close": close, "Adj Close": close,
                         "Volume": [400 if str(m) == "2026-08" else 100 for m in months]})


class CombinedTests(unittest.TestCase):
    def test_monthly_output_allows_unavailable_volume_ratio(self):
        frame = long_history()
        frame.loc["2026-07-31", "Volume"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["sp500_breakout.py", "--with-monthly", "--market", "sp500", "--tickers", "TEST",
                    "--date", "2026-09-13", "--ma", "3", "--lookback", "1", "--no-hold",
                    "--out", str(root / "d.csv"), "--monthly-out", str(root / "m.csv"),
                    "--html", str(root / "r.html"), "--summary", str(root / "s.txt")]
            with patch.object(sys, "argv", args), \
                 patch.object(screener, "download_prices", return_value={"TEST": frame}), \
                 patch.object(screener, "fetch_fundamentals", return_value={}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(screener.main(), 0)
            monthly = report_payload(root / "r.html")["screens"][1]
            self.assertFalse(monthly["volumeFilter"])
            self.assertEqual(len(monthly["hits"]), 1)
            self.assertIsNone(monthly["hits"][0]["volRatio"])
            self.assertIsNone(monthly["hits"][0]["avgVol"])
            self.assertEqual(len(pd.read_csv(root / "m.csv")), 1)
            self.assertNotIn("전월 대비 거래량 2배", (root / "s.txt").read_text(encoding="utf-8"))

    def test_live_month_boundary_uses_each_markets_local_calendar(self):
        from zoneinfo import ZoneInfo

        instant = datetime(2026, 9, 1, 1, tzinfo=ZoneInfo("Asia/Seoul"))
        april = pd.bdate_range("2025-04-01", "2025-04-30")
        prefix = long_history().iloc[:len(april)].copy()
        prefix.index = april
        frame = pd.concat([prefix, long_history()])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = lambda ticker: pd.DataFrame({"ticker": [ticker], "name": ["Fixture"], "sector": [""]})
            args = ["sp500_breakout.py", "--with-monthly", "--ma", "3", "--out", str(root / "d.csv"),
                    "--monthly-out", str(root / "m.csv"), "--html", str(root / "r.html")]
            with patch.object(sys, "argv", args), \
                 patch.object(screener, "datetime", wraps=datetime) as clock, \
                 patch.dict(screener.MARKETS["sp500"], {"loader": lambda refresh: meta("TEST")}), \
                 patch.dict(screener.MARKETS["kospi200"], {"loader": lambda refresh: meta("005930")}), \
                 patch.object(screener, "download_prices", side_effect=[{"TEST": frame}, {"005930.KS": frame}]), \
                 patch.object(screener, "fetch_fundamentals", return_value={}), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                clock.now.side_effect = lambda tz=None: instant.astimezone(tz)
                self.assertEqual(screener.main(), 0)
            monthly = report_payload(root / "r.html")["screens"][1]
            self.assertEqual({m["id"]: m["targetMonth"] for m in monthly["markets"]},
                             {"sp500": "2026-07", "kospi200": "2026-08"})

    def test_monthly_hits_survive_empty_daily_results_and_independent_daily_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = lambda ticker: pd.DataFrame({"ticker": [ticker], "name": ["Fixture"], "sector": [""]})
            args = ["sp500_breakout.py", "--with-monthly", "--market", "all",
                    "--date", "2026-09-13", "--lookback", "1", "--ma", "3",
                    "--vol-mult", "99", "--no-hold", "--out", str(root / "daily.csv"),
                    "--monthly-out", str(root / "monthly.csv"), "--html", str(root / "report.html"),
                    "--summary", str(root / "summary.txt")]
            with patch.object(sys, "argv", args), \
                 patch.dict(screener.MARKETS["sp500"], {"loader": lambda refresh: meta("TEST")}), \
                 patch.dict(screener.MARKETS["kospi200"], {"loader": lambda refresh: meta("005930")}), \
                 patch.object(screener, "download_prices", side_effect=[{"TEST": long_history()},
                                                                       {"005930.KS": long_history()}]) as download, \
                 patch.object(screener, "fetch_fundamentals", return_value={}) as funda, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(screener.main(), 0)
            self.assertEqual(download.call_count, 2)  # Each market is downloaded once for both strategies.
            self.assertEqual(funda.call_args.args[0], ["005930.KS", "TEST"])
            self.assertTrue(pd.read_csv(root / "daily.csv").empty)
            monthly = pd.read_csv(root / "monthly.csv")
            self.assertEqual(len(monthly), 2)
            self.assertEqual(set(monthly["market"]), {"sp500", "kospi200"})
            payload = report_payload(root / "report.html")
            self.assertEqual([s["id"] for s in payload["screens"]], ["daily", "monthly"])
            day, month = payload["screens"]
            self.assertEqual(day["hits"], [])
            self.assertEqual(day["maPeriod"], 3)
            self.assertEqual(day["volMult"], 99)
            self.assertEqual(month["maPeriod"], 10)
            self.assertIsNone(month["volMult"])
            self.assertFalse(month["volumeFilter"])
            self.assertEqual(month["belowMonths"], 6)
            self.assertEqual(len(month["hits"]), 2)
            self.assertTrue(all(m["targetMonth"] == "2026-08" for m in month["markets"]))
            self.assertTrue(all(h["date"] == "2026-08-31" for h in month["hits"]))
            self.assertIn("10개월선 장기 돌파", (root / "summary.txt").read_text(encoding="utf-8"))

    def test_monthly_failure_does_not_overwrite_previous_reports(self):
        # This daily feed is sufficient for MA3 but cannot evaluate sixteen months.
        from test_screener import price_frame
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "daily.csv", root / "monthly.csv", root / "report.html"]
            for path in paths:
                path.write_text("prior successful report", encoding="utf-8")
            args = ["sp500_breakout.py", "--with-monthly", "--market", "sp500",
                    "--tickers", "TEST", "--ma", "3", "--date", "2026-09-13",
                    "--out", str(paths[0]), "--monthly-out", str(paths[1]), "--html", str(paths[2])]
            with patch.object(sys, "argv", args), \
                 patch.object(screener, "download_prices", return_value={"TEST": price_frame([10] * 20)}), \
                 patch.object(screener, "fetch_fundamentals") as funda, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(screener.main(), 1)
            funda.assert_not_called()
            for path in paths:
                self.assertEqual(path.read_text(encoding="utf-8"), "prior successful report")

    def test_same_csv_path_is_rejected_before_network(self):
        args = ["sp500_breakout.py", "--with-monthly", "--out", "same.csv", "--monthly-out", "same.csv"]
        with patch.object(sys, "argv", args), patch.object(screener, "download_prices") as download, \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            screener.main()
        self.assertEqual(exc.exception.code, 2)
        download.assert_not_called()

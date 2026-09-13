"""Offline completed-month signal and daily-to-monthly data quality regressions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monthly_breakout as monthly


def daily_history(monthly_closes=None, monthly_volumes=None):
    """Synthetic weekday sessions with exact integer monthly volume totals."""
    closes = monthly_closes if monthly_closes is not None else [100] * 9 + [90] * 6 + [110]
    totals = monthly_volumes if monthly_volumes is not None else [100] * 15 + [200]
    periods = pd.period_range(end="2026-08", periods=len(closes), freq="M")
    frames = []
    for month, close, total in zip(periods, closes, totals):
        days = pd.bdate_range(month.start_time, month.end_time.normalize())
        volume = [total - len(days) + 1] + [1] * (len(days) - 1)
        frames.append(pd.DataFrame({
            "Open": float(close) - 0.5, "High": float(close) + 1,
            "Low": float(close) - 1, "Close": float(close), "Adj Close": float(close),
            "Volume": volume,
        }, index=days))
    return pd.concat(frames)


class MonthlySignalTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({"ticker": ["TEST"], "name": ["Test"], "sector": [""]})

    def scan(self, frame, peers=None, **kwargs):
        prices = {"TEST": frame, **(peers or {})}
        return monthly.scan_monthly_market(prices, self.meta, "sp500", "2026-09-13", **kwargs)

    def test_last_completed_month_is_always_before_as_of_calendar_month(self):
        for as_of, expected in (("2026-09-13", "2026-08"), ("2026-09-01", "2026-08"),
                                ("2026-08-31", "2026-07"), ("2026-01-01", "2025-12")):
            with self.subTest(as_of=as_of):
                self.assertEqual(str(monthly.last_completed_month(as_of)), expected)

    def test_six_below_months_then_breakout_retains_reference_volume(self):
        rows, charts, summary = self.scan(daily_history())
        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(summary["targetMonth"], "2026-08")
        self.assertEqual(summary["priorBelowStart"], "2026-02")
        self.assertEqual(summary["priorBelowEnd"], "2026-07")
        row, chart = rows[0], charts[0]
        self.assertEqual(row["volume"], 200)
        self.assertEqual(row["previous_month_volume"], 100)
        self.assertEqual(row["vol_ratio"], 2)
        self.assertEqual(row["ma10"], 95)
        self.assertEqual(row["date"], "2026-08-31")
        self.assertEqual(row["last_trading_date"], "2026-08-31")
        self.assertEqual(row["close"], row["last_close"])
        self.assertEqual(row["return_since_%"], 0)
        self.assertEqual(chart["sigIndex"], len(chart["dates"]) - 1)
        self.assertEqual(chart["dates"][-1], row["date"])
        audit = chart["previousMonths"]
        self.assertEqual(len(audit), 6)
        self.assertEqual(audit[0], {"month": "2026-02", "close": 90, "ma": 99, "below": True})
        self.assertEqual(audit[-1], {"month": "2026-07", "close": 90, "ma": 94, "below": True})
        self.assertIn("2026-02-28", chart["dates"])
        json.dumps({"rows": rows, "charts": charts, "summary": summary}, allow_nan=False)

    def test_equality_in_any_prior_month_fails_strict_below_condition(self):
        closes = [100] * 10 + [90] * 5 + [110]
        rows, charts, summary = self.scan(daily_history(closes))
        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(rows, [])
        self.assertEqual(charts, [])

    def test_target_close_equal_to_its_sma_is_not_a_breakout(self):
        # Target's preceding nine closes sum to 900; (900 + 100) / 10 == 100.
        closes = [120] * 9 + [90] * 6 + [100]
        rows, _, summary = self.scan(daily_history(closes))
        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(rows, [])

    def test_adbe_like_volume_decline_does_not_reject_a_price_breakout(self):
        rows, charts, summary = self.scan(daily_history(
            monthly_volumes=[100] * 14 + [127335100, 90134000]))
        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(rows[0]["volume"], 90134000)
        self.assertEqual(rows[0]["previous_month_volume"], 127335100)
        self.assertAlmostEqual(rows[0]["volume"] / rows[0]["previous_month_volume"],
                               0.7078488178043603)
        self.assertEqual(rows[0]["vol_ratio"], 0.71)
        self.assertEqual(charts[0]["volRatio"], 0.71)

    def test_missing_or_unusable_volume_never_blocks_valid_price_signals(self):
        for case, value in (("zero", 0.0), ("nan", float("nan")),
                            ("infinity", float("inf")), ("negative", -1.0),
                            ("missing column", None)):
            with self.subTest(case=case):
                frame = daily_history()
                if case == "missing column":
                    frame = frame.drop(columns="Volume")
                else:
                    frame["Volume"] = value
                rows, charts, summary = self.scan(frame)
                self.assertEqual(summary["scanned"], 1)
                self.assertEqual(summary["hits"], 1)
                self.assertEqual(summary["invalidData"], 0)
                expected = 0 if case == "zero" else None
                self.assertEqual(rows[0]["volume"], expected)
                self.assertEqual(rows[0]["previous_month_volume"], expected)
                self.assertIsNone(rows[0]["vol_ratio"])
                self.assertEqual(charts[0]["volumes"], [expected] * 16)
                json.dumps({"rows": rows, "charts": charts, "summary": summary}, allow_nan=False)

    def test_partial_missing_reference_volume_does_not_become_a_valid_monthly_total(self):
        for month in ("2026-07", "2026-08"):
            with self.subTest(month=month):
                frame = daily_history()
                frame["Volume"] = frame["Volume"].astype(float)
                missing_day = frame.index[frame.index.to_period("M") == pd.Period(month)][3]
                frame.loc[missing_day, "Volume"] = float("nan")
                rows, charts, summary = self.scan(frame)
                self.assertEqual(summary["hits"], 1)
                self.assertEqual(rows[0]["volume"], 200 if month == "2026-07" else None)
                self.assertEqual(rows[0]["previous_month_volume"], 100 if month == "2026-08" else None)
                self.assertIsNone(rows[0]["vol_ratio"])
                self.assertIsNone(charts[0]["volumes"][-2 if month == "2026-07" else -1])
                json.dumps({"rows": rows, "charts": charts}, allow_nan=False)

    def test_current_incomplete_month_is_ignored_and_no_present_day_hold_is_applied(self):
        frame = daily_history()
        current = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5],
                                "Close": [1.0], "Adj Close": [1.0], "Volume": [float("inf")]},
                               index=pd.DatetimeIndex(["2026-09-01"]))
        rows, charts, summary = self.scan(pd.concat([frame, current]))
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(rows[0]["signal_close"], 110)
        self.assertEqual(charts[0]["dates"][-1], "2026-08-31")
        self.assertEqual(charts[0]["lastSignalClose"], 110)

    def test_fifteen_months_are_insufficient_to_evaluate_all_six_prior_smas(self):
        frame = daily_history()
        frame = frame.loc[frame.index.to_period("M") > pd.Period("2025-05")]
        rows, _, summary = self.scan(frame)
        self.assertEqual(rows, [])
        self.assertEqual(summary["scanned"], 0)
        self.assertEqual(summary["noHistory"], 1)

    def test_long_download_keeps_target_fixed_and_chart_smas_fully_warmed(self):
        frame = daily_history([100] * 41 + [90] * 6 + [110], [100] * 47 + [200])
        self.assertEqual(frame.index[0].date().isoformat(), "2022-09-01")
        rows, charts, summary = self.scan(frame)
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(rows[0]["ma10"], 95)
        self.assertEqual(len(charts[0]["dates"]), 36)
        self.assertEqual(charts[0]["dates"][0], "2023-09-30")
        self.assertEqual(charts[0]["dates"][-1], "2026-08-31")
        self.assertNotIn(None, charts[0]["mas"])

    def test_partial_first_required_month_is_excluded_when_peers_show_missing_start(self):
        complete = daily_history()
        rows, _, summary = self.scan(complete.iloc[1:], peers={"PEER": complete})
        self.assertEqual(rows, [])
        self.assertEqual(summary["gapped"], 1)
        self.assertEqual(summary["scanned"], 0)

    def test_missing_calendar_month_is_not_compressed_out_of_sma(self):
        complete = daily_history()
        frame = complete.loc[complete.index.to_period("M") != pd.Period("2025-12")]
        rows, _, summary = self.scan(frame, peers={"PEER": complete})
        self.assertEqual(rows, [])
        self.assertEqual(summary["gapped"], 1)
        self.assertEqual(summary["scanned"], 0)

    def test_missing_target_month_never_falls_back_to_an_older_month(self):
        complete = daily_history()
        frame = complete.loc[complete.index.to_period("M") < pd.Period("2026-08")]
        rows, _, summary = self.scan(frame)
        self.assertEqual(rows, [])
        self.assertEqual(summary["targetMonth"], "2026-08")
        self.assertEqual(summary["gapped"], 1)
        self.assertIsNone(summary["lastTradingDate"])

    def test_missing_month_end_session_is_detected_from_peers(self):
        complete = daily_history()
        rows, _, summary = self.scan(complete.iloc[:-1], peers={"PEER": complete})
        self.assertEqual(rows, [])
        self.assertEqual(summary["gapped"], 1)
        self.assertEqual(summary["stale"], 1)
        self.assertEqual(summary["lastTradingDate"], "2026-08-31")

    def test_missing_price_sessions_in_recent_months_are_detected_from_peers(self):
        complete = daily_history()
        for month in ("2026-07", "2026-08"):
            with self.subTest(month=month):
                day = complete.index[complete.index.to_period("M") == pd.Period(month)][5]
                rows, _, summary = self.scan(complete.drop(index=day), peers={"PEER": complete})
                self.assertEqual(rows, [])
                self.assertEqual(summary["gapped"], 1)
                self.assertEqual(summary["stale"], 0)

    def test_bad_values_and_missing_adjusted_column_are_explicitly_excluded(self):
        for column, value in (("Adj Close", float("inf")), ("Close", 0)):
            with self.subTest(column=column):
                frame = daily_history()
                frame.loc[frame.index[-2], column] = value
                rows, _, summary = self.scan(frame)
                self.assertEqual(rows, [])
                self.assertEqual(summary["invalidData"], 1)
                self.assertIn(column, summary["exclusions"]["TEST"]["detail"])
        frame = daily_history().drop(columns="Adj Close")
        self.assertEqual(self.scan(frame)[2]["invalidData"], 1)
        self.assertEqual(self.scan(frame, price_basis="raw")[2]["hits"], 1)

    def test_missing_download_and_empty_universe_return_quality_counts(self):
        rows, charts, summary = monthly.scan_monthly_market(
            {}, self.meta, "sp500", "2026-09-13")
        self.assertEqual((rows, charts), ([], []))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["scanned"], 0)


class MonthlyAggregationTests(unittest.TestCase):
    def test_ohlc_is_adjusted_daily_before_monthly_aggregation(self):
        frame = pd.DataFrame({"Open": [10.0, 12.0], "High": [40.0, 30.0],
                              "Low": [2.0, 5.0], "Close": [20.0, 15.0],
                              "Adj Close": [10.0, 30.0], "Volume": [40, 60]},
                             index=pd.DatetimeIndex(["2026-08-03", "2026-08-31"]))
        result = monthly.aggregate_monthly_prices(frame, "2026-08", "adj").iloc[0]
        self.assertEqual(result["Open"], 5)
        self.assertEqual(result["High"], 60)
        self.assertEqual(result["Low"], 1)
        self.assertEqual(result["Close"], 30)
        self.assertEqual(result["RawClose"], 15)
        self.assertEqual(result["Volume"], 100)
        self.assertEqual(result["LastTradingDate"], "2026-08-31")
        raw = monthly.aggregate_monthly_prices(frame, "2026-08", "raw").iloc[0]
        self.assertEqual(raw[["Open", "High", "Low", "Close"]].tolist(), [10, 40, 2, 15])

    def test_sma_does_not_bridge_a_missing_calendar_month(self):
        frame = daily_history()
        frame = frame.loc[frame.index.to_period("M") != pd.Period("2026-01")]
        result = monthly.aggregate_monthly_prices(frame, "2026-08")
        self.assertIn(pd.Period("2026-01"), result.index)
        self.assertTrue(pd.isna(result.loc[pd.Period("2026-08"), "MA"]))


if __name__ == "__main__":
    unittest.main()

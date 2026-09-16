"""Deterministic tests for the O'Neil-style relative-strength approximation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relative_strength import calculate_relative_strength  # noqa: E402


DATES = pd.to_datetime([
    "2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15", "2026-01-15",
])


def frame(values, dates=DATES):
    prices = pd.Series(values, index=dates, dtype=float)
    return pd.DataFrame({"Close": prices, "Adj Close": prices})


class RelativeStrengthTests(unittest.TestCase):
    def test_published_quarter_weights_are_applied_to_discrete_quarters(self):
        # Oldest to newest creates quarterly changes of +10%, -10%, +20%, +10%.
        prices = [100, 110, 99, 118.8, 130.68]
        ratings, groups = calculate_relative_strength({("sp500", "TEST"): frame(prices)})
        result = ratings[("sp500", "TEST")]
        self.assertEqual(result["quarterReturnsPct"], [10.0, 20.0, -10.0, 10.0])
        self.assertEqual(result["weightedScorePct"], 8.0)
        self.assertAlmostEqual(result["twelveMonthReturnPct"], 30.68)
        self.assertEqual(result["rating"], 99)
        self.assertEqual(groups["US"]["asOf"], "2026-01-15")

    def test_scores_are_ranked_one_to_ninety_nine_within_country_universe(self):
        inputs = {
            ("sp500", "LOW"): frame([100, 90, 81, 72.9, 65.61]),
            ("sp500", "MID"): frame([100, 100, 100, 100, 100]),
            ("sp500", "HIGH"): frame([100, 120, 144, 172.8, 207.36]),
        }
        ratings, groups = calculate_relative_strength(inputs)
        self.assertEqual(ratings[("sp500", "LOW")]["rating"], 1)
        self.assertEqual(ratings[("sp500", "MID")]["rating"], 50)
        self.assertEqual(ratings[("sp500", "HIGH")]["rating"], 99)
        self.assertEqual(groups["US"]["universeSize"], 3)
        self.assertEqual(groups["US"]["universe"], "S&P 500")

    def test_korean_indices_share_one_country_universe(self):
        inputs = {
            ("kospi200", "005930.KS"): frame([100, 105, 110, 115, 120]),
            ("kosdaq150", "196170.KQ"): frame([100, 110, 120, 130, 140]),
        }
        ratings, groups = calculate_relative_strength(inputs)
        self.assertEqual(groups["KR"]["universe"], "KOSPI 200 + KOSDAQ 150")
        self.assertEqual(groups["KR"]["universeSize"], 2)
        self.assertEqual(ratings[("kospi200", "005930.KS")]["universeSize"], 2)
        self.assertGreater(ratings[("kosdaq150", "196170.KQ")]["rating"],
                           ratings[("kospi200", "005930.KS")]["rating"])

    def test_stale_or_short_history_is_not_silently_ranked(self):
        current = frame([100, 105, 110, 115, 120])
        stale = frame([100, 105, 110, 115], dates=DATES[:-1])
        short = frame([100, 120], dates=DATES[-2:])
        ratings, groups = calculate_relative_strength({
            ("sp500", "CURRENT"): current,
            ("sp500", "STALE"): stale,
            ("sp500", "SHORT"): short,
        })
        self.assertEqual(set(ratings), {("sp500", "CURRENT")})
        self.assertEqual(groups["US"]["universeSize"], 1)


if __name__ == "__main__":
    unittest.main()

"""William O'Neil-style 12-month relative-strength approximation.

The published method gives the latest three-month price change a 40% weight
and each of the preceding three quarters a 20% weight.  The weighted scores
are then ranked against the other securities in the same tracked country
universe and mapped to 1..99.

This repository only tracks the S&P 500 and KOSPI 200 + KOSDAQ 150.  The
result is therefore an approximation within those tracked universes, not the
licensed William O'Neil / IBD Relative Strength Rating for every listed stock.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import pandas as pd


RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)
MAX_ENDPOINT_LAG_DAYS = 7
RS_CSV_COLUMNS = [
    "rs_rating", "rs_weighted_score_%", "rs_12m_return_%",
    "rs_latest_3m_%", "rs_months_4_6_%", "rs_months_7_9_%", "rs_months_10_12_%",
    "rs_as_of", "rs_universe", "rs_universe_size", "rs_price_basis",
]
MARKET_COUNTRY = {
    "sp500": "US",
    "kospi200": "KR",
    "kosdaq150": "KR",
}
MARKET_LABEL = {
    "sp500": "S&P 500",
    "kospi200": "KOSPI 200",
    "kosdaq150": "KOSDAQ 150",
}
MARKET_ORDER = tuple(MARKET_COUNTRY)


def _price_series(frame: pd.DataFrame, price_basis: str) -> pd.Series:
    if price_basis not in ("adj", "raw"):
        raise ValueError("price_basis must be adj or raw")
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("prices require a DataFrame with a DatetimeIndex")
    column = "Adj Close" if price_basis == "adj" else "Close"
    if column not in frame:
        raise ValueError(f"{column} column is missing")
    index = frame.index
    if index.tz is not None:
        index = index.tz_localize(None)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    series = pd.Series(values.to_numpy(), index=index.normalize())
    series = series.where(series.map(math.isfinite) & (series > 0)).dropna()
    return series.groupby(level=0).last().sort_index()


def _quarterly_metrics(series: pd.Series, reference: pd.Timestamp):
    """Return four discrete quarterly changes ending on a common reference date."""
    reference = pd.Timestamp(reference).normalize()
    if reference not in series.index:
        return None

    prices = [float(series.loc[reference])]
    endpoint_dates = [reference]
    for months in (3, 6, 9, 12):
        target = reference - pd.DateOffset(months=months)
        history = series.loc[:target]
        if history.empty:
            return None
        actual = history.index[-1]
        if (target - actual).days > MAX_ENDPOINT_LAG_DAYS:
            return None
        prices.append(float(history.iloc[-1]))
        endpoint_dates.append(actual)

    returns = [(prices[i] / prices[i + 1] - 1) * 100 for i in range(4)]
    if not all(math.isfinite(value) for value in returns):
        return None
    weighted = sum(weight * value for weight, value in zip(RS_WEIGHTS, returns))
    twelve_month = (prices[0] / prices[-1] - 1) * 100
    if not math.isfinite(weighted) or not math.isfinite(twelve_month):
        return None
    return {
        "weightedScorePct": round(weighted, 4),
        "quarterReturnsPct": [round(value, 4) for value in returns],
        "twelveMonthReturnPct": round(twelve_month, 4),
        "endpointDates": [stamp.date().isoformat() for stamp in endpoint_dates],
    }


def _universe_label(markets) -> str:
    ordered = [MARKET_LABEL[market] for market in MARKET_ORDER if market in markets]
    return " + ".join(ordered)


def calculate_relative_strength(frames, price_basis="adj"):
    """Calculate country-grouped O'Neil-style ratings for downloaded frames.

    ``frames`` maps ``(market, provider_ticker)`` to daily OHLCV frames.  A
    country's most common latest priced session is used as a shared endpoint.
    Securities without a valid observation on that session or near each
    three-month boundary are left unrated rather than silently backfilled over
    a long data gap.

    Returns ``(ratings, group_meta)``.  ``ratings`` uses the same tuple keys;
    ``group_meta`` is keyed by country code.
    """
    prepared = {}
    grouped = defaultdict(list)
    markets_by_country = defaultdict(set)
    for key, frame in frames.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("relative-strength frame keys must be (market, ticker)")
        market, _ticker = key
        country = MARKET_COUNTRY.get(market)
        if country is None:
            continue
        markets_by_country[country].add(market)
        try:
            series = _price_series(frame, price_basis)
        except (TypeError, ValueError):
            continue
        if series.empty:
            continue
        prepared[key] = series
        grouped[country].append(key)

    ratings = {}
    group_meta = {}
    for country, keys in grouped.items():
        latest_counts = Counter(prepared[key].index[-1] for key in keys)
        reference = max(latest_counts, key=lambda stamp: (latest_counts[stamp], stamp))
        metrics = {}
        for key in keys:
            value = _quarterly_metrics(prepared[key], reference)
            if value is not None:
                metrics[key] = value

        label = _universe_label(markets_by_country[country])
        size = len(metrics)
        ranked_by_market = Counter(key[0] for key in metrics)
        group_meta[country] = {
            "country": country,
            "asOf": reference.date().isoformat(),
            "universe": label,
            "universeSize": size,
            "rankedByMarket": dict(ranked_by_market),
            "priceBasis": price_basis,
        }
        if not metrics:
            continue

        scores = pd.Series({key: value["weightedScorePct"] for key, value in metrics.items()})
        ordinal = scores.rank(method="average", ascending=True)
        for key, value in metrics.items():
            if size == 1:
                rating = 99
            else:
                rating = 1 + round((float(ordinal.loc[key]) - 1) * 98 / (size - 1))
            ratings[key] = {
                **value,
                "rating": max(1, min(99, int(rating))),
                "asOf": reference.date().isoformat(),
                "universe": label,
                "universeSize": size,
                "priceBasis": price_basis,
                "method": "oneil_12m_quarter_weighted_approximation",
            }

    return ratings, group_meta


def csv_fields(metric):
    """Flatten one rating for the daily/monthly CSV schemas."""
    if metric is None:
        return {column: None for column in RS_CSV_COLUMNS}
    quarters = metric["quarterReturnsPct"]
    return {
        "rs_rating": metric["rating"],
        "rs_weighted_score_%": round(metric["weightedScorePct"], 2),
        "rs_12m_return_%": round(metric["twelveMonthReturnPct"], 2),
        "rs_latest_3m_%": round(quarters[0], 2),
        "rs_months_4_6_%": round(quarters[1], 2),
        "rs_months_7_9_%": round(quarters[2], 2),
        "rs_months_10_12_%": round(quarters[3], 2),
        "rs_as_of": metric["asOf"],
        "rs_universe": metric["universe"],
        "rs_universe_size": metric["universeSize"],
        "rs_price_basis": metric["priceBasis"],
    }


def report_fields(metric):
    """Return the compact, auditable JSON object used by the HTML report."""
    if metric is None:
        return None
    return {
        "rating": metric["rating"],
        "weightedScorePct": round(metric["weightedScorePct"], 2),
        "twelveMonthReturnPct": round(metric["twelveMonthReturnPct"], 2),
        "quarterReturnsPct": [round(value, 2) for value in metric["quarterReturnsPct"]],
        "endpointDates": metric["endpointDates"],
        "asOf": metric["asOf"],
        "universe": metric["universe"],
        "universeSize": metric["universeSize"],
        "priceBasis": metric["priceBasis"],
        "method": metric["method"],
    }

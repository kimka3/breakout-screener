"""Completed-month SMA10 price breakout screening over downloaded daily bars.

This module performs no downloads, fundamental lookups, or file writes. The peer
calendar can expose missing sessions seen in other supplied securities; it cannot
prove that a session omitted by every supplied security was a trading session.
"""
from __future__ import annotations

from collections import Counter
import math

import pandas as pd


MA_MONTHS = 10
BELOW_MONTHS = 6
VOLUME_MONTHS = 3
REQUIRED_MONTHS = MA_MONTHS + BELOW_MONTHS
CHART_MONTHS = 36
MARKET_DISPLAY = {
    "sp500": {"label": "S&P 500", "currency": "USD", "decimals": 2},
    "kospi200": {"label": "KOSPI 200", "currency": "KRW", "decimals": 0},
    "kosdaq150": {"label": "KOSDAQ 150", "currency": "KRW", "decimals": 0},
}


def last_completed_month(as_of):
    """The calendar month before as_of, even when as_of is itself month-end."""
    stamp = pd.Timestamp(as_of)
    if pd.isna(stamp):
        raise ValueError("as_of must be a valid date")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.to_period("M") - 1


def monthly_result_columns():
    return ["market", "date", "target_month", "last_trading_date", "ticker", "name",
            "sector", "close", "signal_close", "ma10", "above_ma_%", "volume",
            "previous_month_volume", "avg_volume_3m", "vol_ratio", "volume_window_start",
            "volume_window_end", "prior_below_start", "prior_below_end",
            "prior_below_months", "last_close", "last_signal_close", "last_above_ma_%",
            "return_since_%", "bars_since", "latest_date", "latest_close",
            "latest_signal_close", "price_basis", "per", "forward_per", "eps",
            "fundamentals_as_of"]


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _month_end(month):
    return month.to_timestamp(how="end").normalize().date().isoformat()


def _daily_frame(frame, target):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("daily prices require a DatetimeIndex")
    daily = frame.copy()
    index = daily.index
    if index.tz is not None:
        index = index.tz_localize(None)
    daily.index = index.normalize()
    daily = daily.loc[daily.index.to_period("M") <= target].sort_index()
    if daily.index.hasnans or daily.index.has_duplicates:
        raise ValueError("daily dates contain missing or duplicate sessions")
    return daily


def _numeric(series, positive=True):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    valid = values.map(math.isfinite) & ((values > 0) if positive else (values >= 0))
    return values.where(valid)


def _optional_volume(value):
    """Preserve valid zeroes and serialize unavailable historical totals as null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number >= 0 else None


def _latest_quote(frame, as_of, price_basis):
    """Read the newest supplied daily row up to as_of; never backfill a bad quote.

    The caller removes unfinished daily bars. This function separately prevents
    future rows from leaking into a historical run and does not affect screening.
    """
    index = frame.index
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    usable = frame.loc[index <= cutoff.normalize()].copy()
    usable.index = index[index <= cutoff.normalize()]
    if usable.empty:
        return None, None, None
    latest_date = usable.index.max()
    latest = usable.loc[usable.index == latest_date]
    result_date = latest_date.date().isoformat()
    if len(latest) != 1:
        return result_date, None, None
    basis_column = "Adj Close" if price_basis == "adj" else "Close"
    if "Close" not in latest or basis_column not in latest:
        return result_date, None, None
    raw = _numeric(latest["Close"]).iloc[0]
    selected = _numeric(latest[basis_column]).iloc[0]
    if pd.isna(raw) or pd.isna(selected):
        return result_date, None, None
    return result_date, float(raw), float(selected)


def aggregate_monthly_prices(frame, target_month, price_basis="adj"):
    """Adjust each DAILY OHLC first, then aggregate completed calendar months.

    Close is the selected signal basis; RawClose retains the provider Close.
    A missing observation never silently becomes a valid monthly extreme/total.
    The scanner validates volume for the target and preceding three months.
    """
    if price_basis not in ("adj", "raw"):
        raise ValueError("price_basis must be adj or raw")
    target = pd.Period(target_month, freq="M")
    daily = _daily_frame(frame, target)
    required = ["Open", "High", "Low", "Close"]
    if price_basis == "adj":
        required.append("Adj Close")
    for column in required:
        if column not in daily:
            raise ValueError(f"{column} column is missing")
    if daily.empty:
        return pd.DataFrame()
    raw_close = _numeric(daily["Close"])
    price = _numeric(daily["Adj Close"]) if price_basis == "adj" else raw_close
    factor = price / raw_close
    volume = (_numeric(daily["Volume"], positive=False) if "Volume" in daily
              else pd.Series(float("nan"), index=daily.index))
    values = pd.DataFrame({
        "Open": _numeric(daily["Open"]) * factor,
        "High": _numeric(daily["High"]) * factor,
        "Low": _numeric(daily["Low"]) * factor,
        "Close": price,
        "RawClose": raw_close,
        "Volume": volume,
    }, index=daily.index)
    groups = values.groupby(values.index.to_period("M"))
    monthly = groups.agg({
        "Open": lambda s: s.iloc[0],
        "High": lambda s: s.max(skipna=False),
        "Low": lambda s: s.min(skipna=False),
        "Close": lambda s: s.iloc[-1],
        "RawClose": lambda s: s.iloc[-1],
        "Volume": lambda s: s.sum(min_count=len(s)),
    })
    monthly["LastTradingDate"] = groups.apply(lambda g: g.index[-1].date().isoformat())
    # Reindex before rolling: a missing calendar month must not compress the SMA.
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), target, freq="M"))
    monthly["MA"] = monthly["Close"].rolling(MA_MONTHS, min_periods=MA_MONTHS).mean()
    return monthly


def _chart(monthly, row, audit, volume_audit, yahoo):
    window = monthly.iloc[-CHART_MONTHS:]

    def clean(column):
        return [None if pd.isna(v) or not math.isfinite(float(v)) else round(float(v), 4)
                for v in window[column]]

    return {
        "market": row["market"], "ticker": row["ticker"], "name": row["name"],
        "sector": row["sector"], "date": row["date"], "targetMonth": row["target_month"],
        "lastTradingDate": row["last_trading_date"], "close": row["close"],
        "signalClose": row["signal_close"], "priceBasis": row["price_basis"],
        "ma": row["ma10"], "abovePct": row["above_ma_%"], "volume": row["volume"],
        "avgVol": row["avg_volume_3m"], "volRatio": row["vol_ratio"],
        "latestDate": row["latest_date"], "latestClose": row["latest_close"],
        "latestSignalClose": row["latest_signal_close"],
        "lastClose": row["last_close"], "lastSignalClose": row["last_signal_close"],
        "lastMa": row["ma10"], "lastPct": row["last_above_ma_%"],
        "retPct": row["return_since_%"],
        "barsSince": 0, "sigIndex": len(window) - 1,
        "dates": [_month_end(month) for month in window.index],
        "opens": clean("Open"), "highs": clean("High"), "lows": clean("Low"),
        "closes": clean("Close"), "mas": clean("MA"),
        "volumes": [_optional_volume(v) for v in window["Volume"]],
        "previousMonths": audit, "priorVolumeMonths": volume_audit, "_y": yahoo,
    }


def scan_monthly_market(prices, meta, market, as_of, price_basis="adj"):
    """Return (CSV rows, chart payloads, market quality summary), without I/O.

    Screening requires 16 consecutive completed calendar months. Every observed
    peer session within those 16 months must exist in the security's data; this
    covers monthly endpoints and the OHLC bars shown in the chart. The target
    close must exceed SMA10, and each of the preceding six closes must be strictly
    below its own SMA10. Target volume must be strictly above the arithmetic mean
    of the preceding three complete monthly totals. Latest daily quotes are
    separate display fields and never change the completed-month predicate.
    Bad securities are excluded, counted, and never replaced by an older signal.
    """
    if market not in MARKET_DISPLAY:
        raise ValueError(f"unsupported market: {market}")
    if price_basis not in ("adj", "raw"):
        raise ValueError("price_basis must be adj or raw")
    target = last_completed_month(as_of)
    required_months = pd.period_range(target - (REQUIRED_MONTHS - 1), target, freq="M")
    previous_months = pd.period_range(target - BELOW_MONTHS, target - 1, freq="M")
    volume_months = pd.period_range(target - VOLUME_MONTHS, target - 1, freq="M")
    info = meta.copy()
    if "ticker" not in info:
        raise ValueError("metadata requires ticker")
    info["ticker"] = info["ticker"].map(_text)
    for field in ("name", "sector"):
        info[field] = info[field].map(_text) if field in info else ""
    if "yahoo" not in info:
        if market in ("kospi200", "kosdaq150"):
            suffix = ".KS" if market == "kospi200" else ".KQ"
            info["yahoo"] = info["ticker"].map(lambda t: f"{t.zfill(6)}{suffix}")
        else:
            info["yahoo"] = info["ticker"].map(lambda t: t.replace(".", "-").upper())
    info = info.drop_duplicates("yahoo")
    prepared, preparation_errors = {}, {}
    for yahoo, frame in prices.items():
        try:
            prepared[yahoo] = _daily_frame(frame, target)
        except (TypeError, ValueError) as exc:
            preparation_errors[yahoo] = str(exc)

    counts = Counter()
    for frame in prepared.values():
        counts.update(frame.index)
    # Small explicitly supplied universes cannot provide multiple corroborations.
    quorum = max(1, int(len(prepared) * 0.03))
    calendar = pd.DatetimeIndex(sorted(day for day, count in counts.items() if count >= quorum))
    period_days = {month: set(calendar[calendar.to_period("M") == month])
                   for month in required_months}
    expected_days = set().union(*period_days.values())
    target_days = period_days[target]
    summary = {
        "id": market, **MARKET_DISPLAY[market], "latest": _month_end(target),
        "targetMonth": str(target), "priorBelowStart": str(previous_months[0]),
        "priorBelowEnd": str(previous_months[-1]),
        "volumeWindowStart": str(volume_months[0]), "volumeWindowEnd": str(volume_months[-1]),
        "lastTradingDate": max(target_days).date().isoformat() if target_days else None,
        "requested": len(info), "scanned": 0, "failed": 0, "noHistory": 0,
        "shortHistory": 0, "invalidData": 0, "gapped": 0, "stale": 0,
        "degraded": False, "hits": 0, "calendarSource": "peer-observed daily sessions",
        "calendarQuorum": quorum, "calendarPeers": len(prepared), "exclusions": {},
    }
    rows, charts = [], []

    def exclude(yahoo, kind, detail):
        summary[kind] += 1
        summary["exclusions"][yahoo] = {"kind": kind, "detail": detail}

    for record in info.to_dict("records"):
        yahoo = record["yahoo"]
        if yahoo not in prices:
            exclude(yahoo, "failed", "daily prices not downloaded")
            continue
        if yahoo in preparation_errors:
            exclude(yahoo, "invalidData", preparation_errors[yahoo])
            continue
        daily = prepared[yahoo]
        if daily.empty or daily.index[0].to_period("M") > required_months[0]:
            exclude(yahoo, "noHistory", "fewer than 16 consecutive completed months")
            continue
        absent_calendar = [str(month) for month, days in period_days.items() if not days]
        if absent_calendar:
            exclude(yahoo, "gapped", "peer calendar has no sessions for " + ", ".join(absent_calendar))
            continue
        missing = sorted(expected_days - set(daily.index))
        if missing:
            if max(target_days) not in daily.index:
                summary["stale"] += 1
            exclude(yahoo, "gapped", "missing peer sessions: " + ", ".join(
                day.date().isoformat() for day in missing[:12]))
            continue
        try:
            # Check every input contributing to the predicate and its chart bars.
            relevant = daily.loc[daily.index.to_period("M") >= required_months[0]]
            columns = ["Open", "High", "Low", "Close"]
            if price_basis == "adj":
                columns.append("Adj Close")
            for column in columns:
                if column not in relevant:
                    raise ValueError(f"{column} column is missing")
                if _numeric(relevant[column]).isna().any():
                    raise ValueError(f"invalid {column} values in required months")
            monthly = aggregate_monthly_prices(daily, target, price_basis)
            required = monthly.reindex(required_months)
            if required[["Open", "High", "Low", "Close", "RawClose"]].isna().any().any():
                raise ValueError("incomplete monthly bars")
            prior = monthly.loc[previous_months]
            current = monthly.loc[target]
            if prior["MA"].isna().any() or pd.isna(current["MA"]):
                raise ValueError("SMA10 cannot be evaluated for all six prior months")
            totals = monthly.loc[pd.period_range(volume_months[0], target, freq="M"), "Volume"]
            if _numeric(totals, positive=False).isna().any():
                raise ValueError("invalid Volume totals in target or preceding three months")
            volume = float(current["Volume"])
            previous_volumes = [float(monthly.loc[month, "Volume"]) for month in volume_months]
            average_volume = sum(previous_volumes) / VOLUME_MONTHS
            if not math.isfinite(average_volume):
                raise ValueError("preceding three-month Volume average is not finite")
        except (KeyError, TypeError, ValueError) as exc:
            exclude(yahoo, "invalidData", str(exc))
            continue
        summary["scanned"] += 1
        if not ((prior["Close"] < prior["MA"]).all()
                and current["Close"] > current["MA"] and volume > average_volume):
            continue
        prior_volume = _optional_volume(monthly.loc[target - 1, "Volume"])
        ratio = volume / average_volume if average_volume > 0 else None
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        decimals = MARKET_DISPLAY[market]["decimals"]
        above = (float(current["Close"]) / float(current["MA"]) - 1) * 100
        latest_date, latest_close, latest_signal_close = _latest_quote(prices[yahoo], as_of, price_basis)
        return_since = ((latest_close / float(current["RawClose"]) - 1) * 100
                        if latest_close is not None else None)
        row = {
            "market": market, "date": _month_end(target), "target_month": str(target),
            "last_trading_date": current["LastTradingDate"], "ticker": record["ticker"],
            "name": record["name"], "sector": record["sector"],
            "close": round(float(current["RawClose"]), decimals),
            "signal_close": round(float(current["Close"]), 4),
            "ma10": round(float(current["MA"]), 4), "above_ma_%": round(above, 2),
            "volume": int(volume), "previous_month_volume": prior_volume,
            "avg_volume_3m": round(average_volume, 4),
            "vol_ratio": round(ratio, 2) if ratio is not None else None,
            "volume_window_start": str(volume_months[0]), "volume_window_end": str(volume_months[-1]),
            "prior_below_start": str(previous_months[0]),
            "prior_below_end": str(previous_months[-1]), "prior_below_months": BELOW_MONTHS,
            "last_close": round(float(current["RawClose"]), decimals),
            "last_signal_close": round(float(current["Close"]), 4),
            "last_above_ma_%": round(above, 2),
            "return_since_%": round(return_since, 2) if return_since is not None else None,
            "bars_since": 0, "latest_date": latest_date,
            "latest_close": round(latest_close, decimals) if latest_close is not None else None,
            "latest_signal_close": round(latest_signal_close, 4) if latest_signal_close is not None else None,
            "price_basis": price_basis, "_y": yahoo,
        }
        audit = [{"month": str(month), "close": round(float(prior.loc[month, "Close"]), 4),
                  "ma": round(float(prior.loc[month, "MA"]), 4), "below": True}
                 for month in previous_months]
        rows.append(row)
        volume_audit = [{"month": str(month), "volume": int(value)}
                        for month, value in zip(volume_months, previous_volumes)]
        charts.append(_chart(monthly, row, audit, volume_audit, yahoo))
    summary["hits"] = len(rows)
    downloaded = len(info) - summary["failed"]
    summary["degraded"] = bool(downloaded and summary["gapped"] > downloaded * 0.2)
    return rows, charts, summary

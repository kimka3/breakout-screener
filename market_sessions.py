"""Offline exchange-session cutoffs for live daily and monthly reports.

The calendar library supplies holidays and special closes. A completed session
does not prove that the price provider has delivered its final bar; callers must
also validate the expected last session against their downloaded data.
"""
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


MARKET_CALENDARS = {
    "sp500": ("XNYS", "America/New_York"),
    "kospi": ("XKRX", "Asia/Seoul"),
    "kosdaq150": ("XKRX", "Asia/Seoul"),
}


@lru_cache(maxsize=12)
def _calendar(name, year):
    # Explicit bounds avoid the library's moving default date range.
    return xcals.get_calendar(name, start=f"{year}-01-01", end=f"{year}-12-31")


@lru_cache(maxsize=48)
def _month_schedule(market, month):
    name, _ = MARKET_CALENDARS[market]
    calendar = _calendar(name, month.year)
    schedule = calendar.schedule.loc[month.start_time:month.end_time, ["close"]]
    if schedule.empty:
        raise ValueError(f"exchange calendar has no sessions for {market} {month}")
    return schedule


def session_close(market, session_date):
    """UTC regular-session close, or None for a holiday/weekend date."""
    day = pd.Timestamp(session_date)
    if day.tzinfo is not None:
        day = day.tz_convert(MARKET_CALENDARS[market][1]).tz_localize(None)
    day = day.normalize()
    schedule = _month_schedule(market, day.to_period("M"))
    return schedule.at[day, "close"] if day in schedule.index else None


def completed_month(market, as_of):
    """Latest fully closed exchange month and its expected final session.

    Live instants must be timezone-aware. Historical date-only queries retain
    their separate 'previous calendar month' convention in monthly_breakout.
    """
    stamp = pd.Timestamp(as_of)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("live as_of must be a valid timezone-aware instant")
    stamp = stamp.tz_convert(MARKET_CALENDARS[market][1])
    month = stamp.tz_localize(None).to_period("M")
    schedule = _month_schedule(market, month)
    if stamp < schedule["close"].iloc[-1]:
        month -= 1
        schedule = _month_schedule(market, month)
    return month, schedule.index[-1].normalize()

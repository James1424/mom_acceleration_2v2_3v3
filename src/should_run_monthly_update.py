"""Decide whether the scheduled GitHub Actions run should update the project.

The workflow is scheduled on the first several weekdays of each month because
GitHub cron cannot express "first trading day" directly. This script makes the
final decision using the NYSE trading calendar.

Manual workflow_dispatch runs should skip this check and run unconditionally.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd


def _first_trading_day_with_market_calendar(today: date) -> date:
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar("XNYS")
    start = today.replace(day=1).isoformat()
    end = today.isoformat()
    schedule = cal.schedule(start_date=start, end_date=end)
    if schedule.empty:
        raise RuntimeError(f"No XNYS sessions found from {start} to {end}")
    return schedule.index[0].date()


def _first_trading_day_fallback(today: date) -> date:
    """Weekend-only fallback used only if pandas_market_calendars is unavailable."""
    first = pd.Timestamp(today.replace(day=1))
    return pd.bdate_range(first, periods=1)[0].date()


def first_trading_day(today: date) -> date:
    try:
        return _first_trading_day_with_market_calendar(today)
    except Exception as exc:
        print(f"Warning: NYSE calendar unavailable, using weekday-only fallback: {exc}")
        return _first_trading_day_fallback(today)


def write_github_output(should_run: bool, today: date, first_day: date) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"should_run={'true' if should_run else 'false'}\n")
        f.write(f"today={today.isoformat()}\n")
        f.write(f"first_trading_day={first_day.isoformat()}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Override current date, YYYY-MM-DD. Mostly for tests.")
    args = parser.parse_args()

    if args.date:
        today = pd.Timestamp(args.date).date()
    else:
        today = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()

    first_day = first_trading_day(today)
    should_run = today == first_day
    print(f"Today in New York: {today}")
    print(f"First NYSE trading day this month: {first_day}")
    print(f"should_run={should_run}")
    write_github_output(should_run, today, first_day)


if __name__ == "__main__":
    main()

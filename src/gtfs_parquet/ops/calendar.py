"""Calendar and date operations.

Functions for determining active services, date ranges, and
trip/stop activity across dates.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def get_dates(feed: Feed) -> list[dt.date]:
    """Return all dates the feed covers, sorted ascending.

    Combines the date range from ``calendar.txt`` with individual dates
    from ``calendar_dates.txt``.

    Args:
        feed: The GTFS feed.
    """
    dates: set[dt.date] = set()

    if feed.calendar is not None and feed.calendar.shape[0] > 0:
        start = feed.calendar["start_date"].min()
        end = feed.calendar["end_date"].max()
        if start is not None and end is not None:
            d = start
            while d <= end:
                dates.add(d)
                d += dt.timedelta(days=1)

    if feed.calendar_dates is not None and feed.calendar_dates.shape[0] > 0:
        for d in feed.calendar_dates["date"].drop_nulls().to_list():
            dates.add(d)

    return sorted(dates)


def get_first_week(feed: Feed) -> list[dt.date]:
    """Return the first Monday–Sunday week (or partial) of the feed's date range.

    Args:
        feed: The GTFS feed.

    Returns:
        Feed dates that fall within the first calendar week, or ``[]`` if
        the feed has no dates.
    """
    dates = get_dates(feed)
    if not dates:
        return []
    start = dates[0]
    monday = start - dt.timedelta(days=start.weekday())
    sunday = monday + dt.timedelta(days=6)
    return [d for d in dates if monday <= d <= sunday]


def get_week(feed: Feed, k: int) -> list[dt.date]:
    """Return the *k*-th Monday–Sunday week (0-indexed) of the feed's date range.

    Args:
        feed: The GTFS feed.
        k: Zero-based week index.
    """
    dates = get_dates(feed)
    if not dates:
        return []
    start = dates[0]
    monday = start - dt.timedelta(days=start.weekday()) + dt.timedelta(weeks=k)
    sunday = monday + dt.timedelta(days=6)
    return [d for d in dates if monday <= d <= sunday]


def subset_dates(feed: Feed, dates: list[dt.date]) -> list[dt.date]:
    """Filter dates to only those within the feed's date range."""
    all_dates = set(get_dates(feed))
    return sorted(d for d in dates if d in all_dates)


def _get_active_services_df(feed: Feed, date: dt.date) -> pl.DataFrame:
    """Return a single-column DataFrame of service_ids active on date."""
    frames: list[pl.DataFrame] = []

    if feed.calendar is not None and feed.calendar.shape[0] > 0:
        day_col = DAY_NAMES[date.weekday()]
        active = feed.calendar.filter(
            (pl.col("start_date") <= date)
            & (pl.col("end_date") >= date)
            & (pl.col(day_col) == 1)
        ).select("service_id")
        frames.append(active)

    if feed.calendar_dates is not None and feed.calendar_dates.shape[0] > 0:
        on_date = feed.calendar_dates.filter(pl.col("date") == date)
        added = on_date.filter(pl.col("exception_type") == 1).select("service_id")
        removed = on_date.filter(pl.col("exception_type") == 2).select("service_id")
        if added.shape[0] > 0:
            frames.append(added)
        if frames and removed.shape[0] > 0:
            combined = pl.concat(frames).unique()
            return combined.join(removed, on="service_id", how="anti")
        elif removed.shape[0] > 0 and not frames:
            return pl.DataFrame({"service_id": []}, schema={"service_id": pl.Utf8})

    if not frames:
        return pl.DataFrame({"service_id": []}, schema={"service_id": pl.Utf8})

    return pl.concat(frames).unique()


def get_active_services(feed: Feed, date: dt.date) -> list[str]:
    """Return service IDs active on the given date.

    Args:
        feed: The GTFS feed.
        date: The date to query.
    """
    return sorted(_get_active_services_df(feed, date)["service_id"].to_list())


def compute_trip_activity(feed: Feed, dates: list[dt.date]) -> pl.DataFrame:
    """Mark each trip as active (1) or inactive (0) on each date.

    Args:
        feed: The GTFS feed.
        dates: Dates to check.

    Returns:
        A DataFrame with columns ``trip_id`` plus one ``Int8`` column per
        date (ISO-formatted name).
    """
    if feed.trips is None:
        return pl.DataFrame()

    result = feed.trips.select("trip_id", "service_id")

    for date in dates:
        active_services = _get_active_services_df(feed, date)
        col_name = date.isoformat()
        # Use a left join + is_not_null to mark active, fully vectorized
        tagged = (
            result.select("service_id")
            .with_row_index("_i")
            .join(
                active_services.with_columns(pl.lit(1, dtype=pl.Int8).alias(col_name)),
                on="service_id",
                how="left",
            )
            .select("_i", col_name)
            .with_columns(pl.col(col_name).fill_null(0))
            .sort("_i")
        )
        result = result.with_columns(tagged[col_name])

    return result.drop("service_id")


def compute_busiest_date(feed: Feed, dates: list[dt.date]) -> dt.date:
    """Return the date with the most active trips.

    Args:
        feed: The GTFS feed.
        dates: Candidate dates to evaluate.
    """
    if feed.trips is None:
        return dates[0]

    best_date = dates[0]
    best_count = 0
    for date in dates:
        active_services = _get_active_services_df(feed, date)
        count = feed.trips.join(active_services, on="service_id", how="semi").shape[0]
        if count > best_count:
            best_count = count
            best_date = date
    return best_date

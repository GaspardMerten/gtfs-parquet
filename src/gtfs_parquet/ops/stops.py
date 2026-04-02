"""Stop operations — timetables, activity, and per-stop statistics."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from gtfs_parquet.ops.calendar import _get_active_services_df, get_active_services
from gtfs_parquet.ops.routes import _mean_headway_min

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def _active_trip_ids(feed: Feed, date: dt.date) -> pl.DataFrame:
    """Single-column DataFrame of trip_ids active on date. For joins."""
    if feed.trips is None:
        return pl.DataFrame({"trip_id": []}, schema={"trip_id": pl.Utf8})
    active_services = _get_active_services_df(feed, date)
    return feed.trips.join(active_services, on="service_id", how="semi").select("trip_id")


def get_stops(
    feed: Feed,
    date: dt.date | None = None,
    route_ids: list[str] | None = None,
    trip_ids: list[str] | None = None,
) -> pl.DataFrame | None:
    """Get stops, optionally filtered by date, route, or trip.

    Args:
        feed: The GTFS feed.
        date: Keep only stops served by trips active on this date.
        route_ids: Keep only stops served by these routes.
        trip_ids: Keep only stops served by these trips.

    Returns:
        A DataFrame of stops, or ``None`` if the feed has no stops table.
    """
    if feed.stops is None:
        return None
    if date is None and route_ids is None and trip_ids is None:
        return feed.stops

    if feed.trips is None or feed.stop_times is None:
        return feed.stops.head(0)

    relevant_trips = feed.trips
    if date is not None:
        active_services = _get_active_services_df(feed, date)
        relevant_trips = relevant_trips.join(active_services, on="service_id", how="semi")
    if route_ids is not None:
        relevant_trips = relevant_trips.filter(pl.col("route_id").is_in(route_ids))
    if trip_ids is not None:
        relevant_trips = relevant_trips.filter(pl.col("trip_id").is_in(trip_ids))

    active_stop_ids = (
        feed.stop_times
        .join(relevant_trips.select("trip_id"), on="trip_id", how="semi")
        .select("stop_id")
        .unique()
    )
    return feed.stops.join(active_stop_ids, on="stop_id", how="semi")


def get_stop_times(feed: Feed, date: dt.date | None = None) -> pl.DataFrame | None:
    """Get stop_times, optionally filtered to trips active on *date*.

    Args:
        feed: The GTFS feed.
        date: If given, keep only stop-times for active trips.

    Returns:
        A DataFrame, or ``None`` if the feed has no stop_times table.
    """
    if feed.stop_times is None:
        return None
    if date is None:
        return feed.stop_times
    active_trips = _active_trip_ids(feed, date)
    return feed.stop_times.join(active_trips, on="trip_id", how="semi")


def get_start_and_end_times(
    feed: Feed, date: dt.date | None = None
) -> tuple[int | None, int | None]:
    """Return (first_departure_seconds, last_arrival_seconds)."""
    st = get_stop_times(feed, date)
    if st is None or st.shape[0] == 0:
        return None, None
    first = st["departure_time"].drop_nulls().dt.total_milliseconds().min()
    last = st["arrival_time"].drop_nulls().dt.total_milliseconds().max()
    return (
        int(first // 1000) if first is not None else None,
        int(last // 1000) if last is not None else None,
    )


def build_stop_timetable(
    feed: Feed,
    stop_id: str,
    dates: list[dt.date],
) -> pl.DataFrame:
    """Build a full timetable for a stop on the given dates."""
    if feed.trips is None or feed.stop_times is None:
        return pl.DataFrame()

    stop_st = feed.stop_times.filter(pl.col("stop_id") == stop_id)
    if stop_st.shape[0] == 0:
        return pl.DataFrame()

    frames = []
    trip_cols = [c for c in ["trip_id", "route_id", "trip_headsign", "direction_id"] if c in feed.trips.columns]
    for date in dates:
        active_trips = _active_trip_ids(feed, date)
        st = stop_st.join(active_trips, on="trip_id", how="semi")
        if st.shape[0] == 0:
            continue
        st = st.with_columns(pl.lit(date).alias("date"))
        st = st.join(feed.trips.select(trip_cols), on="trip_id", how="left")
        frames.append(st)

    if not frames:
        return pl.DataFrame()

    result = pl.concat(frames, how="diagonal_relaxed")
    sort_cols = ["date"]
    if "departure_time" in result.columns:
        sort_cols.append("departure_time")
    return result.sort(sort_cols)


def compute_stop_activity(feed: Feed, dates: list[dt.date]) -> pl.DataFrame:
    """Mark each stop as active (1) or not (0) on each date.

    Only includes stops that appear in stop_times (matching gtfs-kit).
    """
    if feed.stops is None or feed.stop_times is None:
        return pl.DataFrame()

    referenced_stops = feed.stop_times.select("stop_id").unique().sort("stop_id")
    result = referenced_stops

    for date in dates:
        active_trips = _active_trip_ids(feed, date)
        active_stop_ids = (
            feed.stop_times
            .join(active_trips, on="trip_id", how="semi")
            .select("stop_id")
            .unique()
        )
        col_name = date.isoformat()
        result = result.join(
            active_stop_ids.with_columns(pl.lit(1, dtype=pl.Int8).alias(col_name)),
            on="stop_id",
            how="left",
        ).with_columns(pl.col(col_name).fill_null(0))

    return result


def compute_stop_stats(
    feed: Feed,
    dates: list[dt.date],
    stop_ids: list[str] | None = None,
    *,
    split_directions: bool = False,
) -> pl.DataFrame:
    """Compute per-stop per-date statistics.

    Args:
        feed: The GTFS feed.
        dates: Dates to compute stats for.
        stop_ids: If given, restrict to these stops.
        split_directions: If ``True``, produce separate rows per direction.

    Returns:
        A DataFrame sorted by ``(date, stop_id)``.
    """
    if feed.trips is None or feed.stop_times is None:
        return pl.DataFrame()

    frames = []
    for date in dates:
        active_services = _get_active_services_df(feed, date)
        active_trips = feed.trips.join(active_services, on="service_id", how="semi")
        if active_trips.shape[0] == 0:
            continue

        st = feed.stop_times.join(active_trips, on="trip_id", how="inner")
        if stop_ids is not None:
            st = st.filter(pl.col("stop_id").is_in(stop_ids))
        if st.shape[0] == 0:
            continue

        group_cols = ["stop_id"]
        if split_directions and "direction_id" in st.columns:
            group_cols.append("direction_id")

        agg_exprs = [
            pl.col("trip_id").n_unique().alias("num_trips"),
            pl.col("departure_time").min().alias("start_time"),
            pl.col("departure_time").max().alias("end_time"),
        ]
        if "route_id" in st.columns:
            agg_exprs.append(pl.col("route_id").n_unique().alias("num_routes"))

        stats = st.group_by(group_cols).agg(agg_exprs)
        stats = stats.with_columns(_mean_headway_min("num_trips", "start_time", "end_time"))
        stats = stats.with_columns(pl.lit(date).alias("date"))
        frames.append(stats)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "stop_id"])

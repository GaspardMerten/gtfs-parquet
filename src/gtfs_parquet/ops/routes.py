"""Route operations — timetables and per-route statistics."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from gtfs_parquet.constants import MS_PER_MINUTE
from gtfs_parquet.ops.calendar import get_active_services
from gtfs_parquet.ops.trips import compute_trip_stats

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def get_routes(feed: Feed, date: dt.date | None = None) -> pl.DataFrame | None:
    """Get routes, optionally filtered to those with active trips on *date*.

    Args:
        feed: The GTFS feed.
        date: If given, keep only routes that have at least one active trip.

    Returns:
        A DataFrame of routes, or ``None`` if the feed has no routes table.
    """
    if feed.routes is None:
        return None
    if date is None:
        return feed.routes
    services = get_active_services(feed, date)
    if feed.trips is None:
        return feed.routes.head(0)
    active_route_ids = (
        feed.trips
        .filter(pl.col("service_id").is_in(services))
        .select("route_id")
        .unique()
    )
    return feed.routes.join(active_route_ids, on="route_id", how="semi")


def build_route_timetable(
    feed: Feed,
    route_id: str,
    dates: list[dt.date],
) -> pl.DataFrame:
    """Build a full timetable for a route on the given dates.

    Args:
        feed: The GTFS feed.
        route_id: The route to build the timetable for.
        dates: Dates to include.

    Returns:
        A DataFrame of stop-time rows sorted by date and departure time.
    """
    if feed.trips is None or feed.stop_times is None:
        return pl.DataFrame()

    route_trips = feed.trips.filter(pl.col("route_id") == route_id)

    frames = []
    for date in dates:
        services = set(get_active_services(feed, date))
        active_trips = route_trips.filter(pl.col("service_id").is_in(services))
        if active_trips.shape[0] == 0:
            continue
        st = feed.stop_times.join(active_trips, on="trip_id", how="semi")
        st = st.with_columns(pl.lit(date).alias("date"))
        trip_cols = [c for c in ["trip_id", "trip_headsign", "direction_id", "service_id"] if c in active_trips.columns]
        st = st.join(active_trips.select(trip_cols), on="trip_id", how="left")
        frames.append(st)

    if not frames:
        return pl.DataFrame()

    result = pl.concat(frames)
    sort_cols = ["date"]
    if "departure_time" in result.columns:
        sort_cols.append("departure_time")
    return result.sort(sort_cols)


def _mean_headway_min(num_trips: str, start: str, end: str) -> pl.Expr:
    """Compute mean headway in minutes: (end - start) / (n - 1)."""
    return (
        pl.when(pl.col(num_trips) > 1)
        .then(
            (pl.col(end).dt.total_milliseconds() - pl.col(start).dt.total_milliseconds())
            / (pl.col(num_trips) - 1)
            / MS_PER_MINUTE
        )
        .otherwise(None)
        .alias("mean_headway_min")
    )


def compute_route_stats(
    feed: Feed,
    dates: list[dt.date],
    trip_stats: pl.DataFrame | None = None,
    *,
    split_directions: bool = False,
) -> pl.DataFrame:
    """Compute per-route per-date statistics.

    Args:
        feed: The GTFS feed.
        dates: Dates to compute stats for.
        trip_stats: Pre-computed trip stats (from :func:`compute_trip_stats`).
            Computed automatically if ``None``.
        split_directions: If ``True``, produce separate rows per direction.

    Returns:
        A DataFrame sorted by ``(date, route_id)``.
    """
    if feed.trips is None or feed.stop_times is None:
        return pl.DataFrame()

    if trip_stats is None:
        trip_stats = compute_trip_stats(feed)

    frames = []
    for date in dates:
        services = set(get_active_services(feed, date))
        active = trip_stats.filter(
            pl.col("service_id").is_in(services) if "service_id" in trip_stats.columns
            else pl.lit(True)
        )
        if active.shape[0] == 0:
            continue

        group_cols = ["route_id"]
        if split_directions and "direction_id" in active.columns:
            group_cols.append("direction_id")

        agg_exprs = [
            pl.col("trip_id").count().alias("num_trips"),
            pl.col("num_stops").mean().alias("mean_num_stops"),
            pl.col("start_time").min().alias("start_time"),
            pl.col("end_time").max().alias("end_time"),
            pl.col("is_loop").max().alias("is_loop"),
            pl.col("duration_h").sum().alias("service_duration_h"),
        ]
        if "distance" in active.columns:
            agg_exprs.append(pl.col("distance").sum().alias("service_distance"))
            agg_exprs.append(pl.col("distance").mean().alias("mean_trip_distance"))
        if not split_directions and "direction_id" in active.columns:
            agg_exprs.append(
                (pl.col("direction_id").n_unique() > 1).cast(pl.Int8).alias("is_bidirectional")
            )

        stats = active.group_by(group_cols).agg(agg_exprs)
        stats = stats.with_columns(_mean_headway_min("num_trips", "start_time", "end_time"))

        if "service_distance" in stats.columns:
            stats = stats.with_columns(
                pl.when(pl.col("service_duration_h") > 0)
                .then(pl.col("service_distance") / pl.col("service_duration_h"))
                .otherwise(None)
                .alias("service_speed")
            )

        stats = stats.with_columns(pl.lit(date).alias("date"))

        if feed.routes is not None:
            meta_cols = ["route_id"]
            for c in ("route_short_name", "route_type"):
                if c in feed.routes.columns:
                    meta_cols.append(c)
            stats = stats.join(feed.routes.select(meta_cols), on="route_id", how="left")

        frames.append(stats)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed").sort(["date", "route_id"])

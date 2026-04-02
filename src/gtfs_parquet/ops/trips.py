"""Trip operations — filtering and per-trip statistics."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from gtfs_parquet.constants import LOOP_DISTANCE_THRESHOLD_M, MS_PER_HOUR
from gtfs_parquet.geo import haversine_m
from gtfs_parquet.ops.calendar import get_active_services

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def get_trips(feed: Feed, date: dt.date | None = None) -> pl.DataFrame | None:
    """Get trips, optionally filtered to those active on *date*.

    Args:
        feed: The GTFS feed.
        date: If given, only trips whose service is active on this date.

    Returns:
        A DataFrame of trips, or ``None`` if the feed has no trips table.
    """
    if feed.trips is None:
        return None
    if date is None:
        return feed.trips
    services = get_active_services(feed, date)
    return feed.trips.filter(pl.col("service_id").is_in(services))


def compute_trip_stats(
    feed: Feed,
    *,
    route_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Compute per-trip statistics.

    Columns produced: *trip_id*, *route_id*, *num_stops*, *start_time*,
    *end_time*, *start_stop_id*, *end_stop_id*, *is_loop*, *duration_h*,
    *distance*, *speed*.  Optional columns (e.g. *route_type*) are
    included when available.

    Args:
        feed: The GTFS feed.
        route_ids: If given, restrict to trips on these routes.

    Returns:
        A DataFrame with one row per trip.
    """
    if feed.trips is None or feed.stop_times is None:
        return pl.DataFrame()

    trips = feed.trips
    if route_ids is not None:
        trips = trips.filter(pl.col("route_id").is_in(route_ids))

    st = feed.stop_times

    trip_agg = (
        st.group_by("trip_id")
        .agg(
            pl.col("stop_sequence").count().alias("num_stops"),
            pl.col("departure_time").sort_by("stop_sequence").first().alias("start_time"),
            pl.col("departure_time").sort_by("stop_sequence").last().alias("end_time"),
            pl.col("stop_id").sort_by("stop_sequence").first().alias("start_stop_id"),
            pl.col("stop_id").sort_by("stop_sequence").last().alias("end_stop_id"),
            *([pl.col("shape_dist_traveled").max().alias("distance")]
              if "shape_dist_traveled" in st.columns else []),
        )
    )

    trip_agg = trip_agg.with_columns(
        (
            (pl.col("end_time").dt.total_milliseconds() - pl.col("start_time").dt.total_milliseconds())
            / MS_PER_HOUR
        ).alias("duration_h")
    )

    trip_agg = _compute_is_loop(trip_agg, feed)

    if "distance" in trip_agg.columns:
        trip_agg = trip_agg.with_columns(
            pl.when(pl.col("duration_h") > 0)
            .then(pl.col("distance") / pl.col("duration_h"))
            .otherwise(None)
            .alias("speed")
        )
    else:
        trip_agg = trip_agg.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("distance"),
            pl.lit(None, dtype=pl.Float64).alias("speed"),
        )

    result = trips.join(trip_agg, on="trip_id", how="inner")

    if feed.routes is not None:
        route_cols = ["route_id"]
        for c in ("route_short_name", "route_type"):
            if c in feed.routes.columns:
                route_cols.append(c)
        result = result.join(feed.routes.select(route_cols), on="route_id", how="left", suffix="_route")

    out_cols = ["trip_id", "route_id"]
    for c in ("route_short_name", "route_type", "direction_id", "shape_id", "service_id"):
        if c in result.columns:
            out_cols.append(c)
    out_cols += [
        "num_stops", "start_time", "end_time", "start_stop_id", "end_stop_id",
        "is_loop", "duration_h", "distance", "speed",
    ]
    return result.select([c for c in out_cols if c in result.columns])


def _compute_is_loop(trip_agg: pl.DataFrame, feed: Feed) -> pl.DataFrame:
    """Determine if each trip is a loop based on geographic proximity of endpoints."""
    has_coords = (
        feed.stops is not None
        and "stop_lat" in feed.stops.columns
        and "stop_lon" in feed.stops.columns
    )
    if not has_coords:
        return trip_agg.with_columns(
            (pl.col("start_stop_id") == pl.col("end_stop_id")).cast(pl.Int8).alias("is_loop")
        )

    stop_coords = feed.stops.select("stop_id", "stop_lat", "stop_lon")
    trip_agg = (
        trip_agg
        .join(
            stop_coords.rename({"stop_id": "start_stop_id", "stop_lat": "start_lat", "stop_lon": "start_lon"}),
            on="start_stop_id", how="left",
        )
        .join(
            stop_coords.rename({"stop_id": "end_stop_id", "stop_lat": "end_lat", "stop_lon": "end_lon"}),
            on="end_stop_id", how="left",
        )
    )

    dist_m = haversine_m(
        pl.col("start_lat"), pl.col("start_lon"),
        pl.col("end_lat"), pl.col("end_lon"),
    )

    return (
        trip_agg
        .with_columns((dist_m < LOOP_DISTANCE_THRESHOLD_M).cast(pl.Int8).fill_null(0).alias("is_loop"))
        .drop("start_lat", "start_lon", "end_lat", "end_lon")
    )

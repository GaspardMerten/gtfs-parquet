"""Feed cleaning operations.

Strip whitespace from ID columns and remove orphan records that are
not referenced by any trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def clean(feed: Feed) -> "Feed":
    """Apply all cleaning steps and return a new Feed.

    Runs :func:`clean_ids` followed by :func:`drop_zombies`.

    Args:
        feed: The original feed.
    """
    result = clean_ids(feed)
    result = drop_zombies(result)
    return result


def clean_ids(feed: Feed) -> "Feed":
    """Strip leading/trailing whitespace from all string ID columns.

    Args:
        feed: The original feed.

    Returns:
        A new Feed with cleaned IDs.
    """
    from gtfs_parquet.feed import Feed as FeedClass
    from dataclasses import fields

    new = FeedClass()
    for f in fields(feed):
        df = getattr(feed, f.name)
        if df is None:
            setattr(new, f.name, None)
            continue
        str_id_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 and c.endswith("_id")]
        if str_id_cols:
            df = df.with_columns([pl.col(c).str.strip_chars() for c in str_id_cols])
        setattr(new, f.name, df)
    return new


def drop_zombies(feed: Feed) -> "Feed":
    """Remove orphan records not referenced by any trip.

    Keeps only routes, stops, calendar entries, shapes, frequencies,
    and transfers that are transitively referenced by at least one trip.
    Parent stations of active stops are also preserved.

    Args:
        feed: The original feed.

    Returns:
        A new Feed with orphans removed.
    """
    from gtfs_parquet.feed import Feed as FeedClass
    from dataclasses import fields

    new = FeedClass()
    for f in fields(feed):
        setattr(new, f.name, getattr(feed, f.name))

    if new.trips is None or new.trips.shape[0] == 0:
        return new

    # Referenced IDs from trips (as single-column DataFrames for joins)
    used_routes = new.trips.select("route_id").unique()
    used_services = new.trips.select("service_id").unique()
    used_shapes = (
        new.trips.select("shape_id").drop_nulls().unique()
        if "shape_id" in new.trips.columns
        else None
    )
    used_trip_ids = new.trips.select("trip_id").unique()

    # Filter stop_times to valid trips
    if new.stop_times is not None:
        new.stop_times = new.stop_times.join(used_trip_ids, on="trip_id", how="semi")

    # Used stops from stop_times
    used_stop_ids = (
        new.stop_times.select("stop_id").drop_nulls().unique()
        if new.stop_times is not None and "stop_id" in new.stop_times.columns
        else None
    )

    # Routes
    if new.routes is not None:
        new.routes = new.routes.join(used_routes, on="route_id", how="semi")

    # Agency — keep agencies referenced by active routes
    if new.agency is not None and new.routes is not None and "agency_id" in new.routes.columns and "agency_id" in new.agency.columns:
        used_agencies = new.routes.select("agency_id").drop_nulls().unique()
        new.agency = new.agency.join(used_agencies, on="agency_id", how="semi")

    # Stops — keep used stops + their parent stations
    if new.stops is not None and used_stop_ids is not None:
        active_stops = new.stops.join(used_stop_ids, on="stop_id", how="semi")
        if "parent_station" in new.stops.columns:
            parent_ids = active_stops.select(
                pl.col("parent_station").alias("stop_id")
            ).drop_nulls().unique()
            parents = new.stops.join(parent_ids, on="stop_id", how="semi")
            new.stops = pl.concat([active_stops, parents]).unique(subset=["stop_id"])
        else:
            new.stops = active_stops

    # Calendar
    if new.calendar is not None:
        new.calendar = new.calendar.join(used_services, on="service_id", how="semi")

    # Calendar dates
    if new.calendar_dates is not None:
        new.calendar_dates = new.calendar_dates.join(used_services, on="service_id", how="semi")

    # Shapes
    if new.shapes is not None and used_shapes is not None:
        new.shapes = new.shapes.join(used_shapes, on="shape_id", how="semi")

    # Frequencies
    if new.frequencies is not None:
        new.frequencies = new.frequencies.join(used_trip_ids, on="trip_id", how="semi")

    # Transfers
    if new.transfers is not None and used_stop_ids is not None:
        cols = new.transfers.columns
        if "from_stop_id" in cols and "to_stop_id" in cols:
            new.transfers = new.transfers.filter(
                pl.col("from_stop_id").is_in(used_stop_ids["stop_id"])
                & pl.col("to_stop_id").is_in(used_stop_ids["stop_id"])
            )

    # Feed info always kept
    new.feed_info = feed.feed_info

    return new

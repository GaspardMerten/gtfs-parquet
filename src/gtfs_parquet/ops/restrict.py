"""Feed restriction / subsetting operations.

All functions return a new :class:`~gtfs_parquet.feed.Feed` instance;
the original feed is never mutated.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from gtfs_parquet.ops.calendar import _get_active_services_df

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def restrict_to_trips(feed: Feed, trip_ids: list[str]) -> "Feed":
    """Return a new Feed restricted to the given trips and their dependencies.

    Cascading: keeps only routes, stops, services, shapes, and transfers
    referenced by the selected trips.

    Args:
        feed: The original feed.
        trip_ids: Trip IDs to keep.
    """
    from gtfs_parquet.feed import Feed as FeedClass

    new = FeedClass()
    trip_ids_df = pl.DataFrame({"trip_id": trip_ids})

    # Trips
    if feed.trips is None:
        return new
    new.trips = feed.trips.join(trip_ids_df, on="trip_id", how="semi")

    used_routes = new.trips.select("route_id").unique()
    used_services = new.trips.select("service_id").unique()
    used_shapes = (
        new.trips.select("shape_id").drop_nulls().unique()
        if "shape_id" in new.trips.columns
        else None
    )

    # Stop times
    if feed.stop_times is not None:
        new.stop_times = feed.stop_times.join(trip_ids_df, on="trip_id", how="semi")
        used_stops = new.stop_times.select("stop_id").drop_nulls().unique() if "stop_id" in new.stop_times.columns else None
    else:
        used_stops = None

    # Routes
    if feed.routes is not None:
        new.routes = feed.routes.join(used_routes, on="route_id", how="semi")

    # Agency
    if feed.agency is not None:
        if new.routes is not None and "agency_id" in new.routes.columns and "agency_id" in feed.agency.columns:
            used_agencies = new.routes.select("agency_id").drop_nulls().unique()
            new.agency = feed.agency.join(used_agencies, on="agency_id", how="semi")
        else:
            new.agency = feed.agency

    # Stops + parent stations
    if feed.stops is not None and used_stops is not None:
        new.stops = feed.stops.join(used_stops, on="stop_id", how="semi")
        if "parent_station" in new.stops.columns:
            parent_ids = new.stops.select(pl.col("parent_station").alias("stop_id")).drop_nulls().unique()
            parents = feed.stops.join(parent_ids, on="stop_id", how="semi")
            new.stops = pl.concat([new.stops, parents]).unique(subset=["stop_id"])

    # Calendar
    if feed.calendar is not None:
        new.calendar = feed.calendar.join(used_services, on="service_id", how="semi")
    if feed.calendar_dates is not None:
        new.calendar_dates = feed.calendar_dates.join(used_services, on="service_id", how="semi")

    # Shapes
    if feed.shapes is not None and used_shapes is not None:
        new.shapes = feed.shapes.join(used_shapes, on="shape_id", how="semi")

    # Frequencies
    if feed.frequencies is not None:
        new.frequencies = feed.frequencies.join(trip_ids_df, on="trip_id", how="semi")

    # Transfers
    if feed.transfers is not None and used_stops is not None:
        cols = feed.transfers.columns
        if "from_stop_id" in cols and "to_stop_id" in cols:
            new.transfers = feed.transfers.filter(
                pl.col("from_stop_id").is_in(used_stops["stop_id"])
                & pl.col("to_stop_id").is_in(used_stops["stop_id"])
            )

    new.feed_info = feed.feed_info
    return new


def restrict_to_routes(feed: Feed, route_ids: list[str]) -> "Feed":
    """Return a new Feed restricted to trips on the given routes.

    Args:
        feed: The original feed.
        route_ids: Route IDs to keep.
    """
    if feed.trips is None:
        return feed
    trip_ids = feed.trips.filter(
        pl.col("route_id").is_in(route_ids)
    )["trip_id"].to_list()
    return restrict_to_trips(feed, trip_ids)


def restrict_to_dates(feed: Feed, dates: list[dt.date]) -> "Feed":
    """Return a new Feed restricted to trips active on at least one of *dates*.

    Args:
        feed: The original feed.
        dates: Dates to keep.
    """
    if feed.trips is None:
        return feed

    # Collect all active services across all dates
    service_frames = [_get_active_services_df(feed, d) for d in dates]
    all_services = pl.concat(service_frames).unique() if service_frames else pl.DataFrame({"service_id": []}, schema={"service_id": pl.Utf8})

    trip_ids = feed.trips.join(all_services, on="service_id", how="semi")["trip_id"].to_list()
    return restrict_to_trips(feed, trip_ids)

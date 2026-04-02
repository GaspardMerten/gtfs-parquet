"""Network-level statistics and feed description."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from gtfs_parquet.ops.calendar import get_active_services, get_dates, get_first_week
from gtfs_parquet.ops.trips import compute_trip_stats

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed


def describe(feed: Feed, sample_date: dt.date | None = None) -> pl.DataFrame:
    """Produce a summary description of the feed.

    Args:
        feed: The GTFS feed.
        sample_date: Date used for active-trip/route counts.  Falls back
            to the first weekday in the first week.

    Returns:
        A two-column DataFrame: ``indicator``, ``value``.
    """
    rows: list[dict[str, str]] = []

    def add(indicator: str, value: object) -> None:
        rows.append({"indicator": indicator, "value": str(value) if value is not None else ""})

    # Agencies
    if feed.agency is not None and "agency_name" in feed.agency.columns:
        names = feed.agency["agency_name"].to_list()
        add("agencies", ", ".join(str(n) for n in names))
    if feed.agency is not None and "agency_timezone" in feed.agency.columns:
        tz = feed.agency["agency_timezone"][0]
        add("timezone", tz)

    # Date range
    dates = get_dates(feed)
    if dates:
        add("start_date", dates[0].isoformat())
        add("end_date", dates[-1].isoformat())
        add("num_dates", len(dates))

    # Counts
    for name, table in [
        ("num_routes", feed.routes),
        ("num_trips", feed.trips),
        ("num_stops", feed.stops),
    ]:
        add(name, table.shape[0] if table is not None else 0)

    if feed.stop_times is not None:
        add("num_stop_times", feed.stop_times.shape[0])
    if feed.shapes is not None:
        add("num_shapes", feed.shapes.select("shape_id").n_unique() if "shape_id" in feed.shapes.columns else 0)

    # Sample date stats
    if sample_date is None and dates:
        week = get_first_week(feed)
        if week:
            # Pick a weekday from the first week
            weekdays = [d for d in week if d.weekday() < 5]
            sample_date = weekdays[0] if weekdays else week[0]

    if sample_date is not None:
        services = get_active_services(feed, sample_date)
        add("sample_date", sample_date.isoformat())
        if feed.trips is not None:
            active_trips = feed.trips.filter(pl.col("service_id").is_in(services))
            add("active_trips", active_trips.shape[0])
            if feed.routes is not None:
                active_routes = feed.routes.join(
                    active_trips.select("route_id").unique(), on="route_id", how="semi"
                )
                add("active_routes", active_routes.shape[0])

    return pl.DataFrame(rows)


def compute_network_stats(
    feed: Feed,
    dates: list[dt.date],
    trip_stats: pl.DataFrame | None = None,
    *,
    split_route_types: bool = False,
) -> pl.DataFrame:
    """Compute network-wide statistics per date.

    Args:
        feed: The GTFS feed.
        dates: Dates to compute stats for.
        trip_stats: Pre-computed trip stats.  Computed automatically if ``None``.
        split_route_types: If ``True``, produce separate rows per route type.

    Returns:
        A DataFrame with columns *date*, *num_routes*, *num_trips*,
        *num_stops*, *service_duration_h*, and optionally
        *service_distance* and *service_speed*.
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

        group_cols: list[str] = []
        if split_route_types and "route_type" in active.columns:
            group_cols.append("route_type")

        agg_exprs = [
            pl.col("trip_id").n_unique().alias("num_trips"),
            pl.col("route_id").n_unique().alias("num_routes"),
            pl.col("duration_h").sum().alias("service_duration_h"),
        ]
        if "distance" in active.columns:
            agg_exprs.append(pl.col("distance").sum().alias("service_distance"))

        if group_cols:
            stats = active.group_by(group_cols).agg(agg_exprs)
        else:
            stats = pl.DataFrame({
                "num_trips": [active["trip_id"].n_unique()],
                "num_routes": [active["route_id"].n_unique()] if "route_id" in active.columns else [0],
                "service_duration_h": [active["duration_h"].sum()] if "duration_h" in active.columns else [0.0],
                **({"service_distance": [active["distance"].sum()]} if "distance" in active.columns else {}),
            })

        # Speed (guard div-by-zero)
        if "service_distance" in stats.columns:
            stats = stats.with_columns(
                pl.when(pl.col("service_duration_h") > 0)
                .then(pl.col("service_distance") / pl.col("service_duration_h"))
                .otherwise(None)
                .alias("service_speed")
            )

        # Count active stops
        active_trips_df = feed.trips.filter(pl.col("service_id").is_in(services))
        active_st = feed.stop_times.join(active_trips_df.select("trip_id"), on="trip_id", how="semi")
        num_stops = active_st["stop_id"].n_unique()
        stats = stats.with_columns(pl.lit(num_stops).alias("num_stops").cast(pl.UInt32))

        stats = stats.with_columns(pl.lit(date).alias("date"))
        frames.append(stats)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed").sort("date")

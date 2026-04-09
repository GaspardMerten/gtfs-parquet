"""Graph and connection operations for network analysis.

Provides methods for building timetable graphs, computing segment
frequencies, and generating CSA-compatible connection lists.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from gtfs_parquet.feed import Feed

_MS_PER_MIN = 60_000
_MS_PER_HOUR = 3_600_000

_EMPTY_PAIRS_SCHEMA = {
    "stop_id": pl.Utf8,
    "next_stop_id": pl.Utf8,
    "dep_min": pl.Float64,
    "arr_min": pl.Float64,
    "trip_id": pl.Utf8,
    "service_id": pl.Utf8,
}


def _consecutive_pairs(
    feed: Feed,
    service_ids: list[str],
    hour_filter: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Consecutive stop pairs from filtered stop_times.

    Returns columns: stop_id, next_stop_id, dep_min, arr_min, trip_id,
    service_id.  Times are in minutes from midnight.
    """
    if feed.stop_times is None or feed.trips is None:
        return pl.DataFrame(schema=_EMPTY_PAIRS_SCHEMA)

    active_trips = (
        feed.trips.lazy()
        .filter(pl.col("service_id").is_in(service_ids))
        .select("trip_id", "service_id")
    )

    st_cols = ["trip_id", "stop_id", "stop_sequence", "departure_time", "arrival_time"]
    has_pickup = "pickup_type" in feed.stop_times.columns
    has_dropoff = "drop_off_type" in feed.stop_times.columns
    if has_pickup:
        st_cols.append("pickup_type")
    if has_dropoff:
        st_cols.append("drop_off_type")

    st = (
        feed.stop_times.lazy()
        .select([c for c in st_cols if c in feed.stop_times.columns])
        .join(active_trips, on="trip_id", how="inner")
    )

    # Exclude pass-through stops (no pickup AND no drop-off).
    if has_pickup and has_dropoff:
        st = st.filter(
            ~((pl.col("pickup_type") == 1) & (pl.col("drop_off_type") == 1))
        )

    if hour_filter is not None:
        start_ms = hour_filter[0] * _MS_PER_HOUR
        end_ms = hour_filter[1] * _MS_PER_HOUR
        st = st.filter(
            (pl.col("departure_time").dt.total_milliseconds() >= start_ms)
            & (pl.col("departure_time").dt.total_milliseconds() < end_ms)
        )

    return (
        st.sort("trip_id", "stop_sequence")
        .with_columns(
            pl.col("stop_id").shift(-1).over("trip_id").alias("next_stop_id"),
            pl.col("arrival_time").shift(-1).over("trip_id").alias("next_arr"),
        )
        .filter(pl.col("next_stop_id").is_not_null())
        .select(
            "stop_id",
            "next_stop_id",
            (pl.col("departure_time").dt.total_milliseconds() / _MS_PER_MIN).alias("dep_min"),
            (pl.col("next_arr").dt.total_milliseconds() / _MS_PER_MIN).alias("arr_min"),
            "trip_id",
            "service_id",
        )
        .collect()
    )


# ------------------------------------------------------------------
# 1. build_timetable_graph
# ------------------------------------------------------------------

def build_timetable_graph(
    feed: Feed,
    service_ids: list[str],
    hour_filter: tuple[int, int] | None = None,
) -> dict[str, list[tuple[str, float, float, str]]]:
    """Build an adjacency-list timetable graph of consecutive stop pairs.

    For each trip, consecutive stops (by stop_sequence) produce directed
    edges carrying departure and arrival times in minutes from midnight.

    Args:
        feed: The GTFS feed.
        service_ids: Service IDs whose trips to include.
        hour_filter: Optional ``(start_hour, end_hour)`` — keep only
            departures in ``[start, end)`` hours.

    Returns:
        ``{stop_id: [(next_stop_id, dep_min, arr_min, trip_id), …]}``.
    """
    pairs = _consecutive_pairs(feed, service_ids, hour_filter)
    if pairs.shape[0] == 0:
        return {}

    graph: dict[str, list[tuple[str, float, float, str]]] = {}

    col_stop = pairs["stop_id"].to_list()
    col_next = pairs["next_stop_id"].to_list()
    col_dep = pairs["dep_min"].to_list()
    col_arr = pairs["arr_min"].to_list()
    col_trip = pairs["trip_id"].to_list()

    for i in range(len(col_stop)):
        sid = col_stop[i]
        edges = graph.get(sid)
        if edges is None:
            edges = []
            graph[sid] = edges
        edges.append((col_next[i], col_dep[i], col_arr[i], col_trip[i]))

    return graph


# ------------------------------------------------------------------
# 2. get_service_day_counts
# ------------------------------------------------------------------

def get_service_day_counts(
    feed: Feed,
    dates: list[dt.date],
) -> dict[str, int]:
    """Count how many of *dates* each service_id is active on.

    Combines ``calendar`` and ``calendar_dates`` in a single vectorised
    pass — O(C + D) instead of the per-date loop.

    Args:
        feed: The GTFS feed.
        dates: Dates to evaluate.

    Returns:
        ``{service_id: active_day_count}``.
    """
    if not dates:
        return {}

    dates_df = pl.DataFrame({"date": dates}).with_columns(
        pl.col("date").dt.weekday().alias("_wd")  # 1=Mon … 7=Sun
    )

    pairs: list[pl.DataFrame] = []

    if feed.calendar is not None and feed.calendar.shape[0] > 0:
        day_match = (
            pl.when(pl.col("_wd") == 1).then(pl.col("monday"))
            .when(pl.col("_wd") == 2).then(pl.col("tuesday"))
            .when(pl.col("_wd") == 3).then(pl.col("wednesday"))
            .when(pl.col("_wd") == 4).then(pl.col("thursday"))
            .when(pl.col("_wd") == 5).then(pl.col("friday"))
            .when(pl.col("_wd") == 6).then(pl.col("saturday"))
            .when(pl.col("_wd") == 7).then(pl.col("sunday"))
            .otherwise(0)
        )
        active = (
            feed.calendar.lazy()
            .join(dates_df.lazy(), how="cross")
            .filter(
                (pl.col("date") >= pl.col("start_date"))
                & (pl.col("date") <= pl.col("end_date"))
                & (day_match == 1)
            )
            .select("service_id", "date")
            .collect()
        )
        pairs.append(active)

    if feed.calendar_dates is not None and feed.calendar_dates.shape[0] > 0:
        cd = feed.calendar_dates.filter(pl.col("date").is_in(dates))
        added = cd.filter(pl.col("exception_type") == 1).select("service_id", "date")
        removed = cd.filter(pl.col("exception_type") == 2).select("service_id", "date")

        if added.shape[0] > 0:
            pairs.append(added)

        if pairs and removed.shape[0] > 0:
            combined = pl.concat(pairs).unique(["service_id", "date"])
            combined = combined.join(removed, on=["service_id", "date"], how="anti")
            pairs = [combined]
        elif removed.shape[0] > 0:
            pairs = []

    if not pairs:
        return {}

    result = (
        pl.concat(pairs)
        .unique(["service_id", "date"])
        .group_by("service_id")
        .len()
    )
    return dict(zip(result["service_id"].to_list(), result["len"].to_list()))


# ------------------------------------------------------------------
# 3. build_stop_lookup
# ------------------------------------------------------------------

def build_stop_lookup(
    feed: Feed,
    parent_stations: bool = True,
) -> dict[str, dict]:
    """Build a ``{stop_id: info}`` lookup, optionally resolving parent stations.

    When *parent_stations* is ``True`` and a stop has a ``parent_station``,
    coordinates and name are taken from the parent row (falling back to the
    child's own values when the parent lacks them).

    Args:
        feed: The GTFS feed.
        parent_stations: Resolve to parent-station coordinates.

    Returns:
        ``{stop_id: {"stop_id": …, "stop_name": …, "stop_lat": …, "stop_lon": …}}``.
    """
    if feed.stops is None:
        return {}

    info_cols = ["stop_id"]
    for c in ("stop_name", "stop_lat", "stop_lon"):
        if c in feed.stops.columns:
            info_cols.append(c)

    if not parent_stations or "parent_station" not in feed.stops.columns:
        return {row["stop_id"]: row for row in feed.stops.select(info_cols).to_dicts()}

    df = feed.stops.select([*info_cols, "parent_station"])

    # Self-join to get parent's info
    parent_rename = {c: f"_p_{c}" for c in info_cols if c != "stop_id"}
    parent_info = (
        df.select(
            pl.col("stop_id").alias("parent_station"),
            *[pl.col(c).alias(f"_p_{c}") for c in info_cols if c != "stop_id"],
        )
    )

    resolved = df.join(parent_info, on="parent_station", how="left")

    # Prefer parent values via coalesce
    select_exprs: list[pl.Expr] = [pl.col("stop_id")]
    for c in info_cols:
        if c == "stop_id":
            continue
        select_exprs.append(pl.coalesce(f"_p_{c}", c).alias(c))

    final = resolved.select(select_exprs)
    return {row["stop_id"]: row for row in final.to_dicts()}


# ------------------------------------------------------------------
# 4. compute_segment_frequencies
# ------------------------------------------------------------------

def compute_segment_frequencies(
    feed: Feed,
    service_ids: list[str],
    hour_filter: tuple[int, int] | None = None,
    service_day_counts: dict[str, int] | None = None,
) -> dict[tuple[str, str], float]:
    """Average daily trips per directed stop-pair segment.

    Each trip traversal of a segment ``(A, B)`` contributes
    ``service_day_counts[service_id]`` trip-days. The total is divided
    by the sum of unique service-days to yield an average daily
    frequency. When *service_day_counts* is ``None``, every traversal
    counts as 1 (raw trip count, not daily average).

    Args:
        feed: The GTFS feed.
        service_ids: Service IDs whose trips to include.
        hour_filter: Optional ``(start_hour, end_hour)``.
        service_day_counts: Output of :func:`get_service_day_counts`.

    Returns:
        ``{(from_stop_id, to_stop_id): frequency}``.
    """
    pairs = _consecutive_pairs(feed, service_ids, hour_filter)
    if pairs.shape[0] == 0:
        return {}

    if service_day_counts is not None:
        # Map service_id -> weight, aggregate weighted counts per segment
        weight_df = pl.DataFrame({
            "service_id": list(service_day_counts.keys()),
            "_w": list(service_day_counts.values()),
        }).cast({"_w": pl.Float64})
        total_days = sum(service_day_counts.values())

        freq = (
            pairs.lazy()
            .join(weight_df.lazy(), on="service_id", how="left")
            .with_columns(pl.col("_w").fill_null(0.0))
            .group_by("stop_id", "next_stop_id")
            .agg(pl.col("_w").sum().alias("_wsum"))
            .with_columns(
                (pl.col("_wsum") / max(total_days, 1)).alias("freq")
            )
            .collect()
        )
    else:
        freq = (
            pairs.lazy()
            .group_by("stop_id", "next_stop_id")
            .len()
            .rename({"len": "freq"})
            .with_columns(pl.col("freq").cast(pl.Float64))
            .collect()
        )

    col_from = freq["stop_id"].to_list()
    col_to = freq["next_stop_id"].to_list()
    col_freq = freq["freq"].to_list()
    return {(col_from[i], col_to[i]): col_freq[i] for i in range(len(col_from))}


# ------------------------------------------------------------------
# 5. compute_connections
# ------------------------------------------------------------------

def compute_connections(
    feed: Feed,
    service_ids: list[str],
    hour_filter: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Time-sorted connections for the Connection Scan Algorithm.

    Returns a DataFrame sorted by ``dep_min`` with columns:

    * **dep_min** — departure time in minutes from midnight
    * **dep_stop_id** — departure stop
    * **arr_min** — arrival time in minutes from midnight
    * **arr_stop_id** — arrival stop
    * **trip_id**

    Args:
        feed: The GTFS feed.
        service_ids: Service IDs whose trips to include.
        hour_filter: Optional ``(start_hour, end_hour)``.

    Returns:
        A :class:`polars.DataFrame` of connections.
    """
    pairs = _consecutive_pairs(feed, service_ids, hour_filter)
    if pairs.shape[0] == 0:
        return pl.DataFrame(
            schema={
                "dep_min": pl.Float64,
                "dep_stop_id": pl.Utf8,
                "arr_min": pl.Float64,
                "arr_stop_id": pl.Utf8,
                "trip_id": pl.Utf8,
            }
        )

    return (
        pairs.select(
            pl.col("dep_min"),
            pl.col("stop_id").alias("dep_stop_id"),
            pl.col("arr_min"),
            pl.col("next_stop_id").alias("arr_stop_id"),
            pl.col("trip_id"),
        )
        .sort("dep_min", "trip_id")
    )


# ------------------------------------------------------------------
# 6. served_stations
# ------------------------------------------------------------------

def served_stations(
    feed: Feed,
    service_ids: list[str],
    hour_filter: tuple[int, int] | None = None,
) -> set[str]:
    """Stop/station IDs served by the given services.

    Filters stop_times to active trips, excludes pass-through stops,
    and resolves to ``parent_station`` where available.

    Args:
        feed: The GTFS feed.
        service_ids: Service IDs whose trips to include.
        hour_filter: Optional ``(start_hour, end_hour)``.

    Returns:
        A set of station IDs.
    """
    if feed.stop_times is None or feed.trips is None:
        return set()

    active_trips = (
        feed.trips.lazy()
        .filter(pl.col("service_id").is_in(service_ids))
        .select("trip_id")
    )

    st_cols = ["trip_id", "stop_id", "departure_time"]
    has_pickup = "pickup_type" in feed.stop_times.columns
    has_dropoff = "drop_off_type" in feed.stop_times.columns
    if has_pickup:
        st_cols.append("pickup_type")
    if has_dropoff:
        st_cols.append("drop_off_type")

    st = (
        feed.stop_times.lazy()
        .select([c for c in st_cols if c in feed.stop_times.columns])
        .join(active_trips, on="trip_id", how="semi")
    )

    if has_pickup and has_dropoff:
        st = st.filter(
            ~((pl.col("pickup_type") == 1) & (pl.col("drop_off_type") == 1))
        )

    if hour_filter is not None:
        start_ms = hour_filter[0] * _MS_PER_HOUR
        end_ms = hour_filter[1] * _MS_PER_HOUR
        st = st.filter(
            (pl.col("departure_time").dt.total_milliseconds() >= start_ms)
            & (pl.col("departure_time").dt.total_milliseconds() < end_ms)
        )

    stop_ids = st.select("stop_id").unique().collect()

    # Resolve to parent stations
    if (
        feed.stops is not None
        and "parent_station" in feed.stops.columns
    ):
        resolved = stop_ids.join(
            feed.stops.select("stop_id", "parent_station"),
            on="stop_id",
            how="left",
        ).with_columns(
            pl.when(
                pl.col("parent_station").is_not_null()
                & (pl.col("parent_station") != "")
            )
            .then(pl.col("parent_station"))
            .otherwise(pl.col("stop_id"))
            .alias("station_id")
        )
        return set(resolved["station_id"].to_list())

    return set(stop_ids["stop_id"].to_list())
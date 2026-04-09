"""Equivalence tests: gtfs-parquet results must match gtfs-kit results.

Requires: run `python tests/precompute_gk.py` first to cache gtfs-kit results.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import polars as pl
import pytest

from gtfs_parquet.ops.calendar import (
    get_active_services,
    get_dates,
    get_first_week,
    compute_busiest_date,
    compute_trip_activity,
)
from gtfs_parquet.ops.trips import get_trips, compute_trip_stats
from gtfs_parquet.ops.routes import get_routes, build_route_timetable, compute_route_stats
from gtfs_parquet.ops.stops import (
    get_stops,
    get_stop_times,
    get_start_and_end_times,
    compute_stop_activity,
    compute_stop_stats,
)
from gtfs_parquet.ops.network import describe
from gtfs_parquet.ops.restrict import restrict_to_routes, restrict_to_dates

ZIP_PATH = "/tmp/stib_gtfs.zip"
CACHE_PATH = Path("/tmp/gk_cache.pkl")


@pytest.fixture(scope="session")
def gk():
    if not CACHE_PATH.exists():
        pytest.skip("Run `python tests/precompute_gk.py` first")
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def gp_feed():
    if not Path(ZIP_PATH).exists():
        pytest.skip(f"{ZIP_PATH} not found")
    from gtfs_parquet import parse_gtfs_zip
    return parse_gtfs_zip(ZIP_PATH)


@pytest.fixture(scope="session")
def gp_week(gp_feed):
    return get_first_week(gp_feed)


@pytest.fixture(scope="session")
def gp_trip_stats(gp_feed):
    return compute_trip_stats(gp_feed)


def gk_date_to_dt(d: str) -> dt.date:
    return dt.date(int(d[:4]), int(d[4:6]), int(d[6:8]))

def dt_to_gk_date(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def dur_to_timestr(d) -> str | None:
    if d is None:
        return None
    total_secs = int(d.total_seconds())
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# Calendar
def test_get_dates(gp_feed, gk):
    assert get_dates(gp_feed) == [gk_date_to_dt(d) for d in gk["dates"]]

def test_get_first_week(gp_feed, gk):
    assert get_first_week(gp_feed) == [gk_date_to_dt(d) for d in gk["first_week"]]

def test_get_active_services(gp_feed, gp_week, gk):
    for gp_d in gp_week:
        assert get_active_services(gp_feed, gp_d) == gk["active_services"][dt_to_gk_date(gp_d)]

def test_compute_busiest_date(gp_feed, gp_week, gk):
    assert compute_busiest_date(gp_feed, gp_week) == gk_date_to_dt(gk["busiest_date"])


# Trips
def test_get_trips_date_filter(gp_feed, gp_week, gk):
    assert sorted(get_trips(gp_feed, gp_week[0])["trip_id"].to_list()) == gk["trip_ids_date0"]

def test_compute_trip_activity(gp_feed, gp_week, gk):
    gp_ta = compute_trip_activity(gp_feed, gp_week).sort("trip_id")
    gk_ta = gk["trip_activity"]
    assert gp_ta["trip_id"].to_list() == gk_ta["trip_id"].tolist()
    for gp_d in gp_week:
        assert gp_ta[gp_d.isoformat()].to_list() == gk_ta[dt_to_gk_date(gp_d)].tolist()

def test_compute_trip_stats_ids(gp_trip_stats, gk):
    assert sorted(gp_trip_stats["trip_id"].to_list()) == gk["trip_stats"]["trip_id"].tolist()

def test_compute_trip_stats_num_stops(gp_trip_stats, gk):
    gp = gp_trip_stats.sort("trip_id")
    assert gp["num_stops"].to_list() == gk["trip_stats"]["num_stops"].tolist()

def test_compute_trip_stats_is_loop(gp_trip_stats, gk):
    gp = gp_trip_stats.sort("trip_id")
    assert gp["is_loop"].to_list() == [int(x) for x in gk["trip_stats"]["is_loop"].tolist()]

def test_compute_trip_stats_start_time(gp_trip_stats, gk):
    gp = gp_trip_stats.sort("trip_id")
    assert [dur_to_timestr(d) for d in gp["start_time"].to_list()] == gk["trip_stats"]["start_time"].tolist()

def test_compute_trip_stats_end_time(gp_trip_stats, gk):
    gp = gp_trip_stats.sort("trip_id")
    assert [dur_to_timestr(d) for d in gp["end_time"].to_list()] == gk["trip_stats"]["end_time"].tolist()

def test_compute_trip_stats_duration(gp_trip_stats, gk):
    gp = gp_trip_stats.sort("trip_id")
    for a, b in zip(gp["duration_h"].to_list(), gk["trip_stats"]["duration"].tolist()):
        if a is not None and b is not None:
            assert abs(a - b) < 0.001


# Routes
def test_get_routes_date_filter(gp_feed, gp_week, gk):
    assert sorted(get_routes(gp_feed, gp_week[0])["route_id"].to_list()) == sorted(gk["route_ids_date0"])

def test_build_route_timetable(gp_feed, gp_week, gk):
    gp_tt = build_route_timetable(gp_feed, gk["route_id_0"], [gp_week[0]])
    assert gp_tt.shape[0] == gk["route_timetable_rows"]

def test_compute_route_stats_ids(gp_feed, gp_week, gp_trip_stats, gk):
    gp_rs = compute_route_stats(gp_feed, [gp_week[0]], gp_trip_stats)
    assert sorted(gp_rs["route_id"].to_list()) == sorted(gk["route_stats"]["route_id"].tolist())

def test_compute_route_stats_num_trips(gp_feed, gp_week, gp_trip_stats, gk):
    gp_rs = compute_route_stats(gp_feed, [gp_week[0]], gp_trip_stats).sort("route_id")
    gk_rs = gk["route_stats"]
    gp_trips = dict(zip(gp_rs["route_id"].to_list(), gp_rs["num_trips"].to_list()))
    gk_trips = dict(zip(gk_rs["route_id"].tolist(), gk_rs["num_trips"].tolist()))
    for rid in gp_trips:
        assert gp_trips[rid] == gk_trips[rid], f"Route {rid}: {gp_trips[rid]} vs {gk_trips[rid]}"


# Stops
def test_get_stops_date_filter(gp_feed, gp_week, gk):
    assert sorted(get_stops(gp_feed, date=gp_week[0])["stop_id"].to_list()) == sorted(gk["stop_ids_date0"])

def test_get_stop_times_date_filter(gp_feed, gp_week, gk):
    assert get_stop_times(gp_feed, gp_week[0]).shape[0] == gk["stop_times_rows_date0"]

def test_get_start_and_end_times(gp_feed, gp_week, gk):
    gp_start, gp_end = get_start_and_end_times(gp_feed, gp_week[0])
    assert gp_start == gk["start_time_s"]
    assert gp_end == gk["end_time_s"]

def test_compute_stop_activity(gp_feed, gp_week, gk):
    gp_sa = compute_stop_activity(gp_feed, gp_week).sort("stop_id")
    gk_sa = gk["stop_activity"]
    assert gp_sa["stop_id"].to_list() == gk_sa["stop_id"].tolist()
    for gp_d in gp_week:
        assert gp_sa[gp_d.isoformat()].to_list() == gk_sa[dt_to_gk_date(gp_d)].tolist()

def test_compute_stop_stats_num_trips(gp_feed, gp_week, gk):
    gp_ss = compute_stop_stats(gp_feed, [gp_week[0]]).sort("stop_id")
    gk_ss = gk["stop_stats"]
    gp_trips = dict(zip(gp_ss["stop_id"].to_list(), gp_ss["num_trips"].to_list()))
    gk_trips = dict(zip(gk_ss["stop_id"].tolist(), gk_ss["num_trips"].tolist()))
    mismatches = [f"  {sid}: gp={gp_trips[sid]} gk={gk_trips[sid]}"
                  for sid in sorted(gp_trips) if sid in gk_trips and gp_trips[sid] != gk_trips[sid]]
    assert not mismatches, "Stop stats num_trips mismatches:\n" + "\n".join(mismatches[:20])


# Describe
def test_describe_counts(gp_feed, gk):
    gp_desc = describe(gp_feed)
    gp_dict = dict(zip(gp_desc["indicator"].to_list(), gp_desc["value"].to_list()))
    gk_dict = gk["describe"]
    assert int(gp_dict["num_routes"]) == gk_dict["num_routes"]
    assert int(gp_dict["num_trips"]) == gk_dict["num_trips"]
    assert int(gp_dict["num_stops"]) == gk_dict["num_stops"]


# Restriction
def test_restrict_to_routes_trips(gp_feed, gk):
    gp_sub = restrict_to_routes(gp_feed, [gk["route_id_0"]])
    assert sorted(gp_sub.trips["trip_id"].to_list()) == gk["restrict_routes_trip_ids"]

def test_restrict_to_routes_stops(gp_feed, gk):
    gp_sub = restrict_to_routes(gp_feed, [gk["route_id_0"]])
    assert sorted(gp_sub.stops["stop_id"].to_list()) == gk["restrict_routes_stop_ids"]

def test_restrict_to_dates(gp_feed, gp_week, gk):
    gp_sub = restrict_to_dates(gp_feed, [gp_week[0]])
    assert sorted(gp_sub.trips["trip_id"].to_list()) == gk["restrict_dates_trip_ids"]

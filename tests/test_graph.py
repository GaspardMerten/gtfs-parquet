"""Tests for graph and connection operations (ops/graph.py)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from gtfs_parquet.feed import Feed
from gtfs_parquet.ops.graph import (
    build_stop_lookup,
    build_timetable_graph,
    compute_connections,
    compute_segment_frequencies,
    get_service_day_counts,
    served_stations,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _ms(h: int, m: int = 0) -> int:
    """Hours + minutes to milliseconds (Duration representation)."""
    return (h * 3600 + m * 60) * 1000


@pytest.fixture()
def feed() -> Feed:
    """Minimal feed with 3 trips across 2 services and 5 stops.

    Topology:
        T1 (WD): S1 -> S2 -> S3   departs 08:00, 08:10, 08:20
        T2 (WD): S1 -> S2 -> S3   departs 09:00, 09:10, 09:20
        T3 (WE): S3 -> S4         departs 10:00, 10:15

    Stops S1 and S2 have parent_station P1.
    """
    stops = pl.DataFrame({
        "stop_id": ["P1", "S1", "S2", "S3", "S4"],
        "stop_name": ["Parent One", "Stop One", "Stop Two", "Stop Three", "Stop Four"],
        "stop_lat": [42.36, 42.3601, 42.3602, 42.37, 42.38],
        "stop_lon": [-71.06, -71.0601, -71.0602, -71.07, -71.08],
        "location_type": [1, 0, 0, 0, 0],
        "parent_station": ["", "P1", "P1", "", ""],
    }).cast({
        "stop_lat": pl.Float32,
        "stop_lon": pl.Float32,
        "location_type": pl.Int8,
    })

    routes = pl.DataFrame({
        "route_id": ["R1", "R2"],
        "route_type": [3, 3],
    }).cast({"route_type": pl.Int8})

    trips = pl.DataFrame({
        "trip_id": ["T1", "T2", "T3"],
        "route_id": ["R1", "R1", "R2"],
        "service_id": ["WD", "WD", "WE"],
    })

    stop_times = pl.DataFrame({
        "trip_id": ["T1", "T1", "T1", "T2", "T2", "T2", "T3", "T3"],
        "stop_id": ["S1", "S2", "S3", "S1", "S2", "S3", "S3", "S4"],
        "stop_sequence": [1, 2, 3, 1, 2, 3, 1, 2],
        "arrival_time": [
            _ms(8, 0), _ms(8, 10), _ms(8, 20),
            _ms(9, 0), _ms(9, 10), _ms(9, 20),
            _ms(10, 0), _ms(10, 15),
        ],
        "departure_time": [
            _ms(8, 0), _ms(8, 10), _ms(8, 20),
            _ms(9, 0), _ms(9, 10), _ms(9, 20),
            _ms(10, 0), _ms(10, 15),
        ],
        "pickup_type": [0, 0, 0, 0, 0, 0, 0, 0],
        "drop_off_type": [0, 0, 0, 0, 0, 0, 0, 0],
    }).cast({
        "stop_sequence": pl.Int16,
        "arrival_time": pl.Duration("ms"),
        "departure_time": pl.Duration("ms"),
        "pickup_type": pl.Int8,
        "drop_off_type": pl.Int8,
    })

    calendar = pl.DataFrame({
        "service_id": ["WD", "WE"],
        "monday": [1, 0],
        "tuesday": [1, 0],
        "wednesday": [1, 0],
        "thursday": [1, 0],
        "friday": [1, 0],
        "saturday": [0, 1],
        "sunday": [0, 1],
        "start_date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
        "end_date": [dt.date(2024, 12, 31), dt.date(2024, 12, 31)],
    }).cast({c: pl.Int8 for c in [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]})

    return Feed(
        stops=stops,
        routes=routes,
        trips=trips,
        stop_times=stop_times,
        calendar=calendar,
    )


@pytest.fixture()
def week_dates() -> list[dt.date]:
    """Mon 2024-01-01 through Sun 2024-01-07."""
    return [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(7)]


@pytest.fixture()
def weekday_dates(week_dates) -> list[dt.date]:
    """Mon–Fri of the first week."""
    return [d for d in week_dates if d.weekday() < 5]


# ------------------------------------------------------------------
# 1. build_timetable_graph
# ------------------------------------------------------------------

class TestBuildTimetableGraph:
    def test_basic_structure(self, feed):
        graph = build_timetable_graph(feed, ["WD"])
        assert set(graph.keys()) == {"S1", "S2"}
        assert len(graph["S1"]) == 2  # T1 and T2
        assert len(graph["S2"]) == 2

    def test_edge_values(self, feed):
        graph = build_timetable_graph(feed, ["WD"])
        edges_s1 = sorted(graph["S1"], key=lambda e: e[1])
        # T1: S1->S2, dep=480min, arr=490min
        assert edges_s1[0] == ("S2", 480.0, 490.0, "T1")
        # T2: S1->S2, dep=540min, arr=550min
        assert edges_s1[1] == ("S2", 540.0, 550.0, "T2")

    def test_weekend_service(self, feed):
        graph = build_timetable_graph(feed, ["WE"])
        assert set(graph.keys()) == {"S3"}
        assert len(graph["S3"]) == 1
        assert graph["S3"][0] == ("S4", 600.0, 615.0, "T3")

    def test_both_services(self, feed):
        graph = build_timetable_graph(feed, ["WD", "WE"])
        assert "S1" in graph
        assert "S3" in graph
        # S3 should have WE edge (S3->S4) only; WD trips end at S3
        s3_next_stops = {e[0] for e in graph["S3"]}
        assert "S4" in s3_next_stops

    def test_hour_filter(self, feed):
        graph = build_timetable_graph(feed, ["WD"], hour_filter=(9, 10))
        # Only T2 departs between 9:00-10:00
        assert set(graph.keys()) == {"S1", "S2"}
        assert len(graph["S1"]) == 1
        assert graph["S1"][0][3] == "T2"

    def test_hour_filter_excludes_all(self, feed):
        graph = build_timetable_graph(feed, ["WD"], hour_filter=(12, 13))
        assert graph == {}

    def test_empty_service_ids(self, feed):
        graph = build_timetable_graph(feed, [])
        assert graph == {}

    def test_nonexistent_service(self, feed):
        graph = build_timetable_graph(feed, ["NONEXISTENT"])
        assert graph == {}

    def test_empty_feed(self):
        graph = build_timetable_graph(Feed(), ["WD"])
        assert graph == {}


# ------------------------------------------------------------------
# 2. get_service_day_counts
# ------------------------------------------------------------------

class TestGetServiceDayCounts:
    def test_full_week(self, feed, week_dates):
        counts = get_service_day_counts(feed, week_dates)
        # Mon-Fri = 5 weekdays, Sat-Sun = 2 weekend days
        assert counts["WD"] == 5
        assert counts["WE"] == 2

    def test_weekdays_only(self, feed, weekday_dates):
        counts = get_service_day_counts(feed, weekday_dates)
        assert counts["WD"] == 5
        assert "WE" not in counts

    def test_single_weekend_day(self, feed):
        counts = get_service_day_counts(feed, [dt.date(2024, 1, 6)])  # Saturday
        assert "WD" not in counts
        assert counts["WE"] == 1

    def test_outside_calendar_range(self, feed):
        counts = get_service_day_counts(feed, [dt.date(2025, 6, 1)])
        assert counts == {}

    def test_empty_dates(self, feed):
        counts = get_service_day_counts(feed, [])
        assert counts == {}

    def test_no_calendar(self):
        feed = Feed(
            trips=pl.DataFrame({"trip_id": ["T1"], "route_id": ["R1"], "service_id": ["S1"]}),
        )
        counts = get_service_day_counts(feed, [dt.date(2024, 1, 1)])
        assert counts == {}

    def test_calendar_dates_only(self):
        """Feed with only calendar_dates (no calendar table)."""
        feed = Feed(
            calendar_dates=pl.DataFrame({
                "service_id": ["S1", "S1", "S2"],
                "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
                "exception_type": [1, 1, 1],
            }).cast({"exception_type": pl.Int8}),
        )
        counts = get_service_day_counts(feed, [dt.date(2024, 1, 1), dt.date(2024, 1, 2)])
        assert counts["S1"] == 2
        assert counts["S2"] == 1

    def test_calendar_dates_removal(self, feed):
        """calendar_dates exception_type=2 removes a service from a date."""
        feed.calendar_dates = pl.DataFrame({
            "service_id": ["WD"],
            "date": [dt.date(2024, 1, 1)],  # Monday — normally active
            "exception_type": [2],
        }).cast({"exception_type": pl.Int8})
        counts = get_service_day_counts(feed, [dt.date(2024, 1, 1), dt.date(2024, 1, 2)])
        # WD should be active on Jan 2 (Tue) but not Jan 1 (removed)
        assert counts["WD"] == 1


# ------------------------------------------------------------------
# 3. build_stop_lookup
# ------------------------------------------------------------------

class TestBuildStopLookup:
    def test_with_parent_stations(self, feed):
        lookup = build_stop_lookup(feed, parent_stations=True)
        # S1 should have P1's coordinates
        assert lookup["S1"]["stop_lat"] == pytest.approx(42.36, abs=0.001)
        assert lookup["S1"]["stop_lon"] == pytest.approx(-71.06, abs=0.001)
        # S2 should also have P1's coordinates
        assert lookup["S2"]["stop_lat"] == pytest.approx(42.36, abs=0.001)
        # S3 has no parent — keeps own coords
        assert lookup["S3"]["stop_lat"] == pytest.approx(42.37, abs=0.001)

    def test_without_parent_stations(self, feed):
        lookup = build_stop_lookup(feed, parent_stations=False)
        # S1 keeps its own coordinates
        assert lookup["S1"]["stop_lat"] == pytest.approx(42.3601, abs=0.0001)

    def test_all_stops_present(self, feed):
        lookup = build_stop_lookup(feed)
        assert set(lookup.keys()) == {"P1", "S1", "S2", "S3", "S4"}

    def test_includes_name(self, feed):
        lookup = build_stop_lookup(feed)
        assert lookup["S1"]["stop_name"] == "Parent One"  # resolved to parent

    def test_empty_feed(self):
        lookup = build_stop_lookup(Feed())
        assert lookup == {}

    def test_no_parent_station_column(self):
        stops = pl.DataFrame({
            "stop_id": ["S1", "S2"],
            "stop_lat": [1.0, 2.0],
            "stop_lon": [3.0, 4.0],
        }).cast({"stop_lat": pl.Float32, "stop_lon": pl.Float32})
        feed = Feed(stops=stops)
        lookup = build_stop_lookup(feed, parent_stations=True)
        assert lookup["S1"]["stop_lat"] == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------
# 4. compute_segment_frequencies
# ------------------------------------------------------------------

class TestComputeSegmentFrequencies:
    def test_unweighted(self, feed):
        freq = compute_segment_frequencies(feed, ["WD"])
        # S1->S2: 2 trips (T1, T2), S2->S3: 2 trips
        assert freq[("S1", "S2")] == 2.0
        assert freq[("S2", "S3")] == 2.0
        assert ("S3", "S4") not in freq  # WE only

    def test_weighted(self, feed, week_dates):
        counts = get_service_day_counts(feed, week_dates)
        freq = compute_segment_frequencies(feed, ["WD", "WE"], service_day_counts=counts)
        total_days = sum(counts.values())  # 5 + 2 = 7
        # S1->S2: 2 trips × 5 WD days / 7 total
        assert freq[("S1", "S2")] == pytest.approx(2 * 5 / total_days)
        # S3->S4: 1 trip × 2 WE days / 7 total
        assert freq[("S3", "S4")] == pytest.approx(1 * 2 / total_days)

    def test_hour_filter(self, feed):
        freq = compute_segment_frequencies(feed, ["WD"], hour_filter=(9, 10))
        assert freq[("S1", "S2")] == 1.0  # only T2
        assert freq[("S2", "S3")] == 1.0

    def test_empty(self, feed):
        freq = compute_segment_frequencies(feed, [])
        assert freq == {}


# ------------------------------------------------------------------
# 5. compute_connections
# ------------------------------------------------------------------

class TestComputeConnections:
    def test_columns(self, feed):
        conns = compute_connections(feed, ["WD"])
        assert set(conns.columns) == {"dep_min", "dep_stop_id", "arr_min", "arr_stop_id", "trip_id"}

    def test_sorted_by_departure(self, feed):
        conns = compute_connections(feed, ["WD"])
        dep_times = conns["dep_min"].to_list()
        assert dep_times == sorted(dep_times)

    def test_correct_connections(self, feed):
        conns = compute_connections(feed, ["WD"])
        assert conns.shape[0] == 4  # 2 edges × 2 trips
        rows = conns.to_dicts()
        # First connection: T1 S1->S2 at 480min
        assert rows[0]["dep_min"] == 480.0
        assert rows[0]["dep_stop_id"] == "S1"
        assert rows[0]["arr_min"] == 490.0
        assert rows[0]["arr_stop_id"] == "S2"
        assert rows[0]["trip_id"] == "T1"

    def test_hour_filter(self, feed):
        conns = compute_connections(feed, ["WD"], hour_filter=(9, 10))
        assert conns.shape[0] == 2  # T2 only: 2 edges

    def test_empty(self, feed):
        conns = compute_connections(feed, [])
        assert conns.shape[0] == 0
        assert set(conns.columns) == {"dep_min", "dep_stop_id", "arr_min", "arr_stop_id", "trip_id"}


# ------------------------------------------------------------------
# 6. served_stations
# ------------------------------------------------------------------

class TestServedStations:
    def test_resolves_parent_stations(self, feed):
        stations = served_stations(feed, ["WD"])
        # S1 -> P1, S2 -> P1, S3 has no parent -> S3
        assert stations == {"P1", "S3"}

    def test_weekend_service(self, feed):
        stations = served_stations(feed, ["WE"])
        # S3, S4 (no parents)
        assert stations == {"S3", "S4"}

    def test_both_services(self, feed):
        stations = served_stations(feed, ["WD", "WE"])
        assert stations == {"P1", "S3", "S4"}

    def test_hour_filter(self, feed):
        stations = served_stations(feed, ["WD"], hour_filter=(9, 10))
        assert stations == {"P1", "S3"}

    def test_empty(self, feed):
        stations = served_stations(feed, [])
        assert stations == set()

    def test_empty_feed(self):
        stations = served_stations(Feed(), ["WD"])
        assert stations == set()

    def test_no_parent_station_column(self):
        """Without parent_station column, returns raw stop_ids."""
        stops = pl.DataFrame({"stop_id": ["S1", "S2"]})
        trips = pl.DataFrame({"trip_id": ["T1"], "route_id": ["R1"], "service_id": ["WD"]})
        stop_times = pl.DataFrame({
            "trip_id": ["T1", "T1"],
            "stop_id": ["S1", "S2"],
            "stop_sequence": [1, 2],
            "departure_time": [_ms(8), _ms(9)],
            "arrival_time": [_ms(8), _ms(9)],
        }).cast({"stop_sequence": pl.Int16, "departure_time": pl.Duration("ms"), "arrival_time": pl.Duration("ms")})
        feed = Feed(stops=stops, trips=trips, stop_times=stop_times)
        assert served_stations(feed, ["WD"]) == {"S1", "S2"}


# ------------------------------------------------------------------
# Pass-through stop exclusion
# ------------------------------------------------------------------

class TestPassThroughExclusion:
    def test_passthrough_stops_excluded(self):
        """Stops with pickup_type=1 AND drop_off_type=1 should be skipped."""
        trips = pl.DataFrame({"trip_id": ["T1"], "route_id": ["R1"], "service_id": ["WD"]})
        stop_times = pl.DataFrame({
            "trip_id": ["T1", "T1", "T1"],
            "stop_id": ["A", "B", "C"],
            "stop_sequence": [1, 2, 3],
            "departure_time": [_ms(8), _ms(8, 10), _ms(8, 20)],
            "arrival_time": [_ms(8), _ms(8, 10), _ms(8, 20)],
            "pickup_type": [0, 1, 0],
            "drop_off_type": [0, 1, 0],
        }).cast({
            "stop_sequence": pl.Int16,
            "departure_time": pl.Duration("ms"),
            "arrival_time": pl.Duration("ms"),
            "pickup_type": pl.Int8,
            "drop_off_type": pl.Int8,
        })
        feed = Feed(trips=trips, stop_times=stop_times)

        graph = build_timetable_graph(feed, ["WD"])
        # B is passthrough, so we should get A->C directly
        assert "A" in graph
        assert graph["A"][0][0] == "C"
        assert "B" not in graph


# ------------------------------------------------------------------
# Feed method wrappers
# ------------------------------------------------------------------

class TestFeedWrappers:
    """Verify Feed method wrappers delegate correctly."""

    def test_build_timetable_graph(self, feed):
        assert feed.build_timetable_graph(["WD"]) == build_timetable_graph(feed, ["WD"])

    def test_get_service_day_counts(self, feed, week_dates):
        assert feed.get_service_day_counts(week_dates) == get_service_day_counts(feed, week_dates)

    def test_build_stop_lookup(self, feed):
        assert feed.build_stop_lookup() == build_stop_lookup(feed)

    def test_compute_segment_frequencies(self, feed):
        assert feed.compute_segment_frequencies(["WD"]) == compute_segment_frequencies(feed, ["WD"])

    def test_compute_connections(self, feed):
        a = feed.compute_connections(["WD"])
        b = compute_connections(feed, ["WD"])
        assert a.equals(b)

    def test_served_stations(self, feed):
        assert feed.served_stations(["WD"]) == served_stations(feed, ["WD"])

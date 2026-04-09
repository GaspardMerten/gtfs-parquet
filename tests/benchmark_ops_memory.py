"""Profile memory usage of operations on De Lijn feed."""

from __future__ import annotations

import gc
import os
import resource
from pathlib import Path

from gtfs_parquet.ops.calendar import (
    get_dates, get_first_week, get_active_services, compute_busiest_date,
    compute_trip_activity,
)
from gtfs_parquet.ops.trips import get_trips, compute_trip_stats
from gtfs_parquet.ops.routes import get_routes, build_route_timetable, compute_route_stats
from gtfs_parquet.ops.stops import (
    get_stops, get_stop_times, get_start_and_end_times,
    build_stop_timetable, compute_stop_stats, compute_stop_activity,
)
from gtfs_parquet.ops.network import describe, compute_network_stats
from gtfs_parquet.ops.restrict import restrict_to_dates
from gtfs_parquet.ops.clean import clean

DELIJN_URL = "https://gtfs.irail.be/de-lijn/de_lijn-gtfs.zip"
DELIJN_PATH = Path("/tmp/delijn_gtfs.zip")


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(label: str):
    class _M:
        def __enter__(self):
            gc.collect()
            self.before = rss_mb()
            return self
        def __exit__(self, *args):
            gc.collect()
            self.after = rss_mb()
            self.delta = self.after - self.before
            print(f"  {label:<40} {self.before:>8.0f} → {self.after:>8.0f}  Δ{self.delta:>+8.0f} MB")
    return _M()


def main() -> None:
    if not DELIJN_PATH.exists():
        import httpx
        print("Downloading De Lijn feed...")
        r = httpx.get(DELIJN_URL, follow_redirects=True, timeout=120)
        DELIJN_PATH.write_bytes(r.content)

    from gtfs_parquet import parse_gtfs_zip

    print("Parsing feed...")
    feed = parse_gtfs_zip(DELIJN_PATH)
    gc.collect()
    baseline = rss_mb()
    print(f"Baseline RSS after parse: {baseline:.0f} MB\n")

    print(f"  {'Operation':<40} {'Before':>8} → {'After':>8}  {'Delta':>9}")
    print(f"  {'-' * 72}")

    with measure("get_dates"):
        dates = get_dates(feed)

    with measure("get_first_week"):
        week = get_first_week(feed)

    with measure("get_active_services"):
        services = get_active_services(feed, week[0])

    with measure("compute_busiest_date (7 days)"):
        busiest = compute_busiest_date(feed, week)

    with measure("get_trips (date filter)"):
        trips = get_trips(feed, week[0])

    with measure("compute_trip_stats"):
        trip_stats = compute_trip_stats(feed)

    with measure("get_routes (date filter)"):
        routes = get_routes(feed, week[0])

    with measure("compute_route_stats (1 date)"):
        route_stats = compute_route_stats(feed, [week[0]], trip_stats)

    with measure("get_stops (date filter)"):
        stops = get_stops(feed, date=week[0])

    with measure("get_stop_times (date filter)"):
        st = get_stop_times(feed, week[0])

    with measure("get_start_and_end_times"):
        se = get_start_and_end_times(feed, week[0])

    with measure("build_stop_timetable"):
        stop_id = feed.stops["stop_id"][0]
        stop_tt = build_stop_timetable(feed, stop_id, [week[0]])

    with measure("compute_stop_stats (1 date)"):
        stop_stats = compute_stop_stats(feed, [week[0]])

    with measure("describe"):
        desc = describe(feed)

    with measure("compute_network_stats (1 date)"):
        net = compute_network_stats(feed, [week[0]], trip_stats)

    with measure("restrict_to_dates (1 date)"):
        sub2 = restrict_to_dates(feed, [week[0]])

    with measure("clean"):
        cleaned = clean(feed)

    with measure("compute_trip_activity (7 days)"):
        ta = compute_trip_activity(feed, week)

    with measure("compute_stop_activity (7 days)"):
        sa = compute_stop_activity(feed, week)

    with measure("build_route_timetable"):
        route_id = feed.routes["route_id"][0]
        rt = build_route_timetable(feed, route_id, [week[0]])

    print(f"\n  Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()

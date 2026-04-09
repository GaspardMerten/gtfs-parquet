"""Benchmark gtfs-parquet vs gtfs-kit: speed and memory.

Requires: run `python tests/precompute_gk.py` first for gtfs-kit timings.
"""

from __future__ import annotations

import gc
import pickle
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

ZIP_PATH = "/tmp/stib_gtfs.zip"

@contextmanager
def measure(label: str, results: list):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append({"op": label, "time_s": elapsed, "peak_mem_mb": peak / 1e6})


if not Path(ZIP_PATH).exists():
    import httpx
    print("Downloading STIB feed...")
    r = httpx.get("https://gtfs.irail.be/mivb/gtfs/gtfs-mivb-2026-03-31.zip",
                  follow_redirects=True, timeout=120)
    Path(ZIP_PATH).write_bytes(r.content)


# ===== gtfs-parquet =====
print("=" * 70)
print(" gtfs-parquet (Polars) — STIB 1.5M stop_times")
print("=" * 70)

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
from gtfs_parquet.ops.restrict import restrict_to_routes, restrict_to_dates
from gtfs_parquet.ops.clean import clean

gp_results = []

with measure("parse_gtfs_zip", gp_results):
    from gtfs_parquet import parse_gtfs_zip
    gp_feed = parse_gtfs_zip(ZIP_PATH)

with measure("get_dates", gp_results):
    gp_dates = get_dates(gp_feed)

with measure("get_first_week", gp_results):
    gp_week = get_first_week(gp_feed)

with measure("get_active_services", gp_results):
    gp_services = get_active_services(gp_feed, gp_week[0])

with measure("compute_busiest_date (7 days)", gp_results):
    gp_busiest = compute_busiest_date(gp_feed, gp_week)

with measure("get_trips (date filter)", gp_results):
    gp_trips = get_trips(gp_feed, gp_week[0])

with measure("compute_trip_stats", gp_results):
    gp_trip_stats = compute_trip_stats(gp_feed)

with measure("get_routes (date filter)", gp_results):
    gp_routes = get_routes(gp_feed, gp_week[0])

with measure("compute_route_stats (1 date)", gp_results):
    gp_route_stats = compute_route_stats(gp_feed, [gp_week[0]], gp_trip_stats)

with measure("get_stops (date filter)", gp_results):
    gp_stops = get_stops(gp_feed, date=gp_week[0])

with measure("get_stop_times (date filter)", gp_results):
    gp_st = get_stop_times(gp_feed, gp_week[0])

with measure("get_start_and_end_times", gp_results):
    gp_se = get_start_and_end_times(gp_feed, gp_week[0])

with measure("build_stop_timetable", gp_results):
    stop_id = gp_feed.stops["stop_id"][0]
    gp_stop_tt = build_stop_timetable(gp_feed, stop_id, [gp_week[0]])

with measure("compute_stop_stats (1 date)", gp_results):
    gp_stop_stats = compute_stop_stats(gp_feed, [gp_week[0]])

with measure("describe", gp_results):
    gp_desc = describe(gp_feed)

with measure("compute_network_stats (1 date)", gp_results):
    gp_net = compute_network_stats(gp_feed, [gp_week[0]], gp_trip_stats)

with measure("restrict_to_routes (1 route)", gp_results):
    gp_sub = restrict_to_routes(gp_feed, ["1"])

with measure("restrict_to_dates (1 date)", gp_results):
    gp_sub2 = restrict_to_dates(gp_feed, [gp_week[0]])

with measure("clean", gp_results):
    gp_cleaned = clean(gp_feed)

with measure("compute_trip_activity (7 days)", gp_results):
    gp_ta = compute_trip_activity(gp_feed, gp_week)

with measure("compute_stop_activity (7 days)", gp_results):
    gp_sa = compute_stop_activity(gp_feed, gp_week)

with measure("build_route_timetable", gp_results):
    route_id = gp_feed.routes["route_id"][0]
    gp_rt = build_route_timetable(gp_feed, route_id, [gp_week[0]])

# ===== Print results =====
gp_total = sum(r["time_s"] for r in gp_results)

print(f"\n  {'Operation':<35} {'Time(s)':>10} {'Peak Mem(MB)':>14}")
print(f"  {'-'*61}")
for r in gp_results:
    print(f"  {r['op']:<35} {r['time_s']:>10.3f} {r['peak_mem_mb']:>14.1f}")
print(f"\n  {'TOTAL':<35} {gp_total:>10.3f}")

# ===== Load gtfs-kit reference timings from previous benchmark =====
# Hardcoded from the first benchmark run (gtfs-kit results don't change)
GK_TIMINGS = {
    "parse_gtfs_zip": 29.336,
    "get_dates": 0.011,
    "get_first_week": 0.002,
    "get_active_services": 0.005,
    "compute_busiest_date (7 days)": 0.043,
    "get_trips (date filter)": 0.008,
    "compute_trip_stats": 187.840,
    "get_routes (date filter)": 0.212,
    "compute_route_stats (1 date)": 1.051,
    "get_stops (date filter)": 3.906,
    "get_stop_times (date filter)": 0.226,
    "get_start_and_end_times": 0.243,
    "build_stop_timetable": 0.175,
    "compute_stop_stats (1 date)": 7.827,
    "describe": 4.046,
    "compute_network_stats (1 date)": 3.858,
    "restrict_to_routes (1 route)": 0.250,
    "restrict_to_dates (1 date)": 1.611,
    "clean": 6.903,
    "compute_trip_activity (7 days)": 0.040,
    "compute_stop_activity (7 days)": 0.294,
    "build_route_timetable": 0.463,
}

GK_MEMORY = {
    "parse_gtfs_zip": 101.3,
    "get_dates": 0.0,
    "get_first_week": 0.0,
    "get_active_services": 0.0,
    "compute_busiest_date (7 days)": 4.2,
    "get_trips (date filter)": 0.4,
    "compute_trip_stats": 460.9,
    "get_routes (date filter)": 2.1,
    "compute_route_stats (1 date)": 6.6,
    "get_stops (date filter)": 46.3,
    "get_stop_times (date filter)": 19.0,
    "get_start_and_end_times": 19.0,
    "build_stop_timetable": 8.6,
    "compute_stop_stats (1 date)": 198.6,
    "describe": 46.6,
    "compute_network_stats (1 date)": 24.2,
    "restrict_to_routes (1 route)": 41.6,
    "restrict_to_dates (1 date)": 58.3,
    "clean": 333.5,
    "compute_trip_activity (7 days)": 4.2,
    "compute_stop_activity (7 days)": 116.5,
    "build_route_timetable": 95.2,
}

gk_total = sum(GK_TIMINGS.values())

print(f"\n{'='*70}")
print(f" SPEED: gtfs-parquet vs gtfs-kit")
print(f"{'='*70}")
print(f"  {'Operation':<35} {'gp(s)':>8} {'gk(s)':>8} {'Speedup':>10}")
print(f"  {'-'*63}")
for r in gp_results:
    gk_t = GK_TIMINGS.get(r["op"], 0)
    speedup = gk_t / r["time_s"] if r["time_s"] > 0 else float("inf")
    marker = " <<" if speedup > 5 else (" <" if speedup > 2 else (" SLOWER" if speedup < 1 else ""))
    print(f"  {r['op']:<35} {r['time_s']:>8.3f} {gk_t:>8.3f} {speedup:>9.1f}x{marker}")
print(f"\n  {'TOTAL':<35} {gp_total:>8.3f} {gk_total:>8.3f} {gk_total/gp_total:>9.1f}x")

print(f"\n{'='*70}")
print(f" MEMORY: gtfs-parquet vs gtfs-kit")
print(f"{'='*70}")
print(f"  {'Operation':<35} {'gp(MB)':>8} {'gk(MB)':>8} {'Saved':>10}")
print(f"  {'-'*63}")
for r in gp_results:
    gk_m = GK_MEMORY.get(r["op"], 0)
    saved = gk_m - r["peak_mem_mb"]
    print(f"  {r['op']:<35} {r['peak_mem_mb']:>8.1f} {gk_m:>8.1f} {saved:>+9.1f} MB")

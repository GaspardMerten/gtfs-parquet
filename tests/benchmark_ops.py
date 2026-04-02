"""Benchmark: gtfs-parquet vs gtfs-kit on the same GTFS feed."""

import gc
import time
import tracemalloc

PATH = "/tmp/stib_gtfs.zip"

results = {}


def bench(label, fn):
    gc.collect()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    print(f"  {label}: {elapsed:.2f}s, peak mem: {peak:.0f} MB")
    results[label] = (elapsed, peak)
    return result


# === gtfs-kit ===
print("=== gtfs-kit (pandas) ===")
import gtfs_kit as gk

tracemalloc.start()
gk_feed = bench("load", lambda: gk.read_feed(PATH, dist_units="km"))
gk_dates = bench("get_dates", lambda: gk_feed.get_dates())
bench("get_first_week", lambda: gk_feed.get_first_week())
bench("compute_busiest_date", lambda: gk_feed.compute_busiest_date(gk_dates))
gk_ts = bench("compute_trip_stats", lambda: gk_feed.compute_trip_stats())
gk_week = gk_feed.get_first_week()
bench("compute_route_stats", lambda: gk_feed.compute_route_stats(gk_week, trip_stats=gk_ts))
bench("compute_stop_stats", lambda: gk_feed.compute_stop_stats(gk_week))
tracemalloc.stop()

gk_results = dict(results)
results.clear()

# === gtfs-parquet ===
print("\n=== gtfs-parquet (polars) ===")
from gtfs_parquet import parse_gtfs

tracemalloc.start()
gp_feed = bench("load", lambda: parse_gtfs(PATH))
gp_dates = bench("get_dates", lambda: gp_feed.get_dates())
bench("get_first_week", lambda: gp_feed.get_first_week())
bench("compute_busiest_date", lambda: gp_feed.compute_busiest_date(gp_dates))
gp_ts = bench("compute_trip_stats", lambda: gp_feed.compute_trip_stats())
gp_week = gp_feed.get_first_week()
bench("compute_route_stats", lambda: gp_feed.compute_route_stats(gp_week))
bench("compute_stop_stats", lambda: gp_feed.compute_stop_stats(gp_week))
tracemalloc.stop()

gp_results = dict(results)

# === Summary ===
print("\n" + "=" * 70)
print(f"{'Operation':<25} {'gtfs-kit':>12} {'gtfs-parquet':>14} {'Speedup':>10} {'Mem ratio':>10}")
print("-" * 70)
for label in gk_results:
    gk_t, gk_m = gk_results[label]
    gp_t, gp_m = gp_results[label]
    speedup = gk_t / gp_t if gp_t > 0 else float("inf")
    mem_ratio = gk_m / gp_m if gp_m > 0 else float("inf")
    print(f"{label:<25} {gk_t:>10.2f}s {gp_t:>12.2f}s {speedup:>9.1f}x {mem_ratio:>9.1f}x")

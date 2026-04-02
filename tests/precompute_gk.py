"""Precompute all gtfs-kit results and save to pickle. Run once."""

import pickle
import time
from pathlib import Path

import gtfs_kit as gk
from gtfs_kit.helpers import timestr_to_seconds

ZIP_PATH = "/tmp/stib_gtfs.zip"
CACHE_PATH = Path("/tmp/gk_cache.pkl")

print("Loading gtfs-kit feed...")
t0 = time.perf_counter()
feed = gk.read_feed(ZIP_PATH, dist_units="km")
print(f"  read_feed: {time.perf_counter()-t0:.1f}s")

results = {}

# Week/dates
results["dates"] = feed.get_dates()
results["first_week"] = feed.get_first_week()
d0 = results["first_week"][0]
week = results["first_week"]

# Services
results["active_services"] = {}
for d in week:
    results["active_services"][d] = sorted(feed.get_active_services(d))

# Busiest
results["busiest_date"] = feed.compute_busiest_date(week)

# Trips
results["trip_ids_date0"] = sorted(feed.get_trips(d0)["trip_id"].tolist())

# Trip activity
t0 = time.perf_counter()
ta = feed.compute_trip_activity(week)
results["trip_activity"] = ta.sort_values("trip_id").reset_index(drop=True)
print(f"  compute_trip_activity: {time.perf_counter()-t0:.1f}s")

# Trip stats (THE SLOW ONE)
t0 = time.perf_counter()
ts = feed.compute_trip_stats()
results["trip_stats"] = ts.sort_values("trip_id").reset_index(drop=True)
print(f"  compute_trip_stats: {time.perf_counter()-t0:.1f}s")

# Routes
results["route_ids_date0"] = sorted(feed.get_routes(d0)["route_id"].tolist())

route_id = feed.routes["route_id"].iloc[0]
results["route_id_0"] = route_id

# Route timetable
results["route_timetable_rows"] = feed.build_route_timetable(route_id, [d0]).shape[0]

# Route stats
t0 = time.perf_counter()
rs = feed.compute_route_stats([d0], ts)
results["route_stats"] = rs.sort_values("route_id").reset_index(drop=True)
print(f"  compute_route_stats: {time.perf_counter()-t0:.1f}s")

# Stops
results["stop_ids_date0"] = sorted(feed.get_stops(d0)["stop_id"].tolist())

# Stop times
results["stop_times_rows_date0"] = feed.get_stop_times(d0).shape[0]

# Start/end times
se = feed.get_start_and_end_times(d0)
results["start_time_s"] = int(timestr_to_seconds(se[0]))
results["end_time_s"] = int(timestr_to_seconds(se[1]))

# Stop activity
t0 = time.perf_counter()
sa = feed.compute_stop_activity(week)
results["stop_activity"] = sa.sort_values("stop_id").reset_index(drop=True)
print(f"  compute_stop_activity: {time.perf_counter()-t0:.1f}s")

# Stop stats
t0 = time.perf_counter()
ss = feed.compute_stop_stats([d0])
results["stop_stats"] = ss.sort_values("stop_id").reset_index(drop=True)
print(f"  compute_stop_stats: {time.perf_counter()-t0:.1f}s")

# Describe
desc = feed.describe()
results["describe"] = dict(zip(desc["indicator"].tolist(), desc["value"].tolist()))

# Restrict
sub_route = feed.restrict_to_routes([route_id])
results["restrict_routes_trip_ids"] = sorted(sub_route.trips["trip_id"].tolist())
results["restrict_routes_stop_ids"] = sorted(sub_route.stops["stop_id"].tolist())

sub_date = feed.restrict_to_dates([d0])
results["restrict_dates_trip_ids"] = sorted(sub_date.trips["trip_id"].tolist())

with open(CACHE_PATH, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved {len(results)} results to {CACHE_PATH}")

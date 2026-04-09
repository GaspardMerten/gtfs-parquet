# gtfs-parquet

Parse [GTFS](https://gtfs.org/) feeds to/from [Parquet](https://parquet.apache.org/) via [Polars](https://pola.rs/) — fast, compact, typed.

## Features

- Parse GTFS from **zip**, **directory**, or **URL**
- Write to **Parquet** (zstd-compressed, sorted for optimal compression) or back to **GTFS**
- Strongly typed schemas with optimised dtypes (Float32 coords, Int16 sequences)
- Built on Polars — zero-copy reads, lazy evaluation ready
- Operations: calendar expansion, network analysis, route/stop/trip stats, timetable graphs, CSA connections

## Installation

```bash
pip install gtfs-parquet
```

## Quick start

```python
from gtfs_parquet import parse_gtfs, write_parquet, read_parquet, write_gtfs

# Parse a GTFS zip (local path or URL)
feed = parse_gtfs("gtfs.zip")

# Write to Parquet — directory, .zip, or .tar (auto-detected by extension)
write_parquet(feed, "output/")           # directory of .parquet files
write_parquet(feed, "output.zip")        # zip archive (no extra compression)
write_parquet(feed, "output.tar")        # tar archive (single file, no extra compression)

# Read back (same formats)
feed = read_parquet("output/")
feed = read_parquet("output.zip")
feed = read_parquet("output.tar")

# Convert back to GTFS zip
write_gtfs(feed, "roundtrip.zip")
```

Additional top-level helpers: `parse_gtfs_dir`, `parse_gtfs_zip`, `write_gtfs_dir`.

## Compression

Parquet output is **significantly smaller** than the original GTFS zip thanks to
zstd compression, sorted row groups, and optimised column types:

| Feed     | GTFS zip | Parquet |  Saving |
|----------|----------|---------|---------|
| STIB     |  5.5 MB  |  3.2 MB |  42.8 % |
| TEC      | 95.2 MB  | 23.9 MB |  75.0 % |
| De Lijn  | 195 MB   | 55.0 MB |  71.8 % |

## Feed object

`Feed` is a plain dataclass with one optional `polars.DataFrame` attribute per GTFS
file (e.g. `feed.stops`, `feed.routes`, `feed.stop_times`). Only files present
in the source feed are populated.

```python
feed.tables()    # dict of all non-None tables
feed.validate()  # check required files and columns
```

## Operations

All operations are standalone functions that take a `Feed` as their first argument.
Import from the `ops` submodules:

```python
from gtfs_parquet.ops.calendar import (
    get_dates, get_first_week, get_week, get_active_services,
    subset_dates, compute_trip_activity, compute_busiest_date,
)
from gtfs_parquet.ops.trips import get_trips, compute_trip_stats
from gtfs_parquet.ops.routes import get_routes, build_route_timetable, compute_route_stats
from gtfs_parquet.ops.stops import (
    get_stops, get_stop_times, get_start_and_end_times,
    build_stop_timetable, compute_stop_activity, compute_stop_stats,
)
from gtfs_parquet.ops.network import describe, compute_network_stats
from gtfs_parquet.ops.restrict import restrict_to_routes, restrict_to_dates, restrict_to_trips
from gtfs_parquet.ops.clean import clean, clean_ids, drop_zombies
from gtfs_parquet.ops.graph import (
    build_timetable_graph, get_service_day_counts, build_stop_lookup,
    compute_segment_frequencies, compute_connections, served_stations,
)

dates = get_dates(feed)
week = get_first_week(feed)
services = get_active_services(feed, dates[0])

trip_stats = compute_trip_stats(feed)
route_stats = compute_route_stats(feed, [dates[0]], trip_stats)
stop_stats = compute_stop_stats(feed, [dates[0]])

# Timetable graph for routing
graph = build_timetable_graph(feed, services, hour_filter=(6, 22))

# CSA-compatible connections
connections = compute_connections(feed, services)

# Segment frequencies weighted by service days
day_counts = get_service_day_counts(feed, dates)
freqs = compute_segment_frequencies(feed, services, service_day_counts=day_counts)
```

The API is inspired by [gtfs-kit](https://github.com/mrcagney/gtfs_kit),
re-implemented on Polars for significantly better performance.

### Performance vs gtfs-kit

Benchmarked on the STIB (Brussels) feed (~5.5 MB, ~9 000 trips):

| Operation             | gtfs-kit (pandas) | gtfs-parquet (Polars) | Speedup |
|-----------------------|------------------:|----------------------:|--------:|
| Load feed             |           2.97 s  |              0.40 s   |     7x  |
| `compute_trip_stats`  |          57.46 s  |              0.05 s   |  1149x  |
| `compute_stop_stats`  |           9.67 s  |              0.19 s   |    51x  |
| `compute_route_stats` |           2.12 s  |              0.08 s   |    27x  |
| `compute_busiest_date`|           0.07 s  |              0.06 s   |     1x  |

Peak process memory: **1020 MB** (gtfs-kit) vs **744 MB** (gtfs-parquet).

## License

[MIT](LICENSE)

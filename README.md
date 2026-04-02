# gtfs-parquet

Parse [GTFS](https://gtfs.org/) feeds to/from [Parquet](https://parquet.apache.org/) via [Polars](https://pola.rs/) — fast, compact, typed.

## Features

- Parse GTFS from **zip**, **directory**, or **URL**
- Write to **Parquet** (zstd-compressed, sorted for optimal compression) or back to **GTFS**
- Strongly typed schemas with optimised dtypes (Float32 coords, Int16 sequences)
- Built on Polars — zero-copy reads, lazy evaluation ready
- Operations: calendar expansion, network analysis, route stats, stop clustering, trip metrics

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

## Compression

Parquet output is **significantly smaller** than the original GTFS zip thanks to
zstd compression, sorted row groups, and optimised column types:

| Feed     | GTFS zip | Parquet |  Saving |
|----------|----------|---------|---------|
| STIB     |  5.5 MB  |  3.2 MB |  42.8 % |
| TEC      | 95.2 MB  | 23.9 MB |  75.0 % |
| De Lijn  | 195 MB   | 55.0 MB |  71.8 % |

## Feed object

`Feed` is a dataclass with one optional `polars.DataFrame` attribute per GTFS
file (e.g. `feed.stops`, `feed.routes`, `feed.stop_times`). Only files present
in the source feed are populated.

## Operations

The operations API is inspired by [gtfs-kit](https://github.com/mrcagney/gtfs_kit),
re-implemented on Polars for significantly better performance.

```python
from gtfs_parquet.ops import calendar, network, routes, stops, trips

# Expand calendar + calendar_dates into per-date service table
services = calendar.dates(feed)

# Route-level statistics
route_stats = routes.stats(feed)

# Stop-level statistics
stop_stats = stops.stats(feed)

# Trip-level metrics
trip_stats = trips.stats(feed)

# Network graph analysis
net_stats = network.stats(feed)
```

### Performance vs gtfs-kit

Benchmarked on the STIB (Brussels) feed (~5.5 MB, ~9 000 trips):

| Operation             | gtfs-kit (pandas) | gtfs-parquet (Polars) | Speedup |
|-----------------------|------------------:|----------------------:|--------:|
| Load feed             |          27.84 s  |              0.50 s   |    56×  |
| `compute_trip_stats`  |         216.40 s  |              0.07 s   |  2919×  |
| `compute_stop_stats`  |          52.41 s  |              0.20 s   |   264×  |
| `compute_route_stats` |           6.76 s  |              0.11 s   |    61×  |
| `compute_busiest_date`|           0.22 s  |              0.10 s   |     2×  |

Peak memory for `compute_trip_stats`: **500 MB** (gtfs-kit) vs **< 1 MB** (gtfs-parquet).

## License

[MIT](LICENSE)

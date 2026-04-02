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

# Write to Parquet directory
write_parquet(feed, "output/")

# Read back
feed = read_parquet("output/")

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

## License

[MIT](LICENSE)

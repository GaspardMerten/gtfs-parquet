"""gtfs-parquet: Parse GTFS feeds to/from Parquet via Polars.

This package provides a high-performance GTFS parser built on Polars,
with Parquet output for compact storage and fast reads.
"""

from gtfs_parquet._version import __version__
from gtfs_parquet.feed import Feed
from gtfs_parquet.parse import parse_gtfs, parse_gtfs_dir, parse_gtfs_zip
from gtfs_parquet.write import read_parquet, write_gtfs, write_gtfs_dir, write_parquet

__all__ = [
    "__version__",
    "Feed",
    "parse_gtfs",
    "parse_gtfs_dir",
    "parse_gtfs_zip",
    "read_parquet",
    "write_gtfs",
    "write_gtfs_dir",
    "write_parquet",
]

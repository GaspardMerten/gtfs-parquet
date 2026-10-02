"""gtfs-parquet: Parse GTFS feeds to/from Parquet via Polars.

This package provides a high-performance GTFS parser built on Polars,
with Parquet output for compact storage and fast reads.
"""

from gtfs_parquet._version import __version__
from gtfs_parquet.convert import convert_gtfs_zip
from gtfs_parquet.feed import Feed
from gtfs_parquet.netex import convert_netex, parse_netex
from gtfs_parquet.parse import parse_gtfs, parse_gtfs_dir, parse_gtfs_zip
from gtfs_parquet.write import (
    read_parquet,
    to_parquet_bytes,
    write_gtfs,
    write_gtfs_dir,
    write_parquet,
)

__all__ = [
    "__version__",
    "convert_gtfs_zip",
    "convert_netex",
    "Feed",
    "parse_gtfs",
    "parse_gtfs_dir",
    "parse_gtfs_zip",
    "parse_netex",
    "read_parquet",
    "to_parquet_bytes",
    "write_gtfs",
    "write_gtfs_dir",
    "write_parquet",
]

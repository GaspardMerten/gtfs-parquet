"""Write a :class:`~gtfs_parquet.feed.Feed` to Parquet or back to GTFS format."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from gtfs_parquet.feed import Feed
from gtfs_parquet.schema import ALL_SCHEMAS, GtfsFileSchema, date_columns, time_columns


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def write_parquet(
    feed: Feed,
    path: str | Path,
    *,
    compression: str = "zstd",
    compression_level: int = 9,
) -> None:
    """Write all Feed tables as individual Parquet files in a directory.

    Tables are sorted by their schema's :attr:`~gtfs_parquet.schema.GtfsFileSchema.sort_keys`
    before writing to improve compression via Parquet's internal encodings.

    Args:
        feed: The feed to write.
        path: Output directory (created if it does not exist).
        compression: Parquet compression codec.
        compression_level: Compression level for the chosen codec.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, df in feed.tables().items():
        schema = ALL_SCHEMAS.get(name)
        if schema and schema.sort_keys:
            # Only sort by keys that actually exist in the DataFrame.
            keys = [k for k in schema.sort_keys if k in df.columns]
            if keys:
                df = df.sort(keys)
        df.write_parquet(
            path / f"{name}.parquet",
            compression=compression,
            compression_level=compression_level,
        )


def read_parquet(path: str | Path) -> Feed:
    """Read a directory of Parquet files back into a Feed.

    Each ``.parquet`` file whose stem matches a Feed attribute is loaded.

    Args:
        path: Directory containing the Parquet files.

    Returns:
        A populated :class:`~gtfs_parquet.feed.Feed`.
    """
    path = Path(path)
    feed = Feed()
    for pq_file in path.glob("*.parquet"):
        table_name = pq_file.stem
        if hasattr(feed, table_name):
            setattr(feed, table_name, pl.read_parquet(pq_file))
    return feed


# ---------------------------------------------------------------------------
# GTFS (CSV) output
# ---------------------------------------------------------------------------


def write_gtfs(feed: Feed, path: str | Path) -> None:
    """Write a Feed as a GTFS zip file.

    Args:
        feed: The feed to write.
        path: Output ``.zip`` file path.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in feed.tables().items():
            schema = ALL_SCHEMAS.get(name)
            csv_df = _to_gtfs_csv_df(df, schema) if schema else df
            csv_bytes = csv_df.write_csv().encode("utf-8")
            zf.writestr(f"{name}.txt", csv_bytes)


def write_gtfs_dir(feed: Feed, path: str | Path) -> None:
    """Write a Feed as a directory of GTFS CSV files.

    Args:
        feed: The feed to write.
        path: Output directory (created if it does not exist).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, df in feed.tables().items():
        schema = ALL_SCHEMAS.get(name)
        csv_df = _to_gtfs_csv_df(df, schema) if schema else df
        csv_df.write_csv(path / f"{name}.txt")


# ---------------------------------------------------------------------------
# Helpers: convert typed columns back to GTFS string format
# ---------------------------------------------------------------------------


def _to_gtfs_csv_df(df: pl.DataFrame, schema: GtfsFileSchema) -> pl.DataFrame:
    """Convert typed DataFrame back to all-string for GTFS CSV output."""
    date_cols = set(date_columns(schema))
    time_cols = set(time_columns(schema))

    exprs: list[pl.Expr] = []
    for col_name in df.columns:
        if col_name in date_cols:
            exprs.append(_format_date_col(col_name))
        elif col_name in time_cols:
            exprs.append(_format_time_col(col_name))
        else:
            exprs.append(pl.col(col_name).cast(pl.Utf8).fill_null(""))
    return df.select(exprs)


def _format_date_col(col_name: str) -> pl.Expr:
    """Format a pl.Date column back to YYYYMMDD string."""
    return pl.col(col_name).dt.strftime("%Y%m%d").fill_null("")


def _format_time_col(col_name: str) -> pl.Expr:
    """Format a pl.Duration column back to HH:MM:SS string — fully vectorized."""
    total_secs = pl.col(col_name).dt.total_milliseconds() // 1000
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    # Zero-pad each component and concatenate
    return (
        h.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + m.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + s.cast(pl.Utf8).str.zfill(2)
    ).fill_null("").alias(col_name)

"""Write a :class:`~gtfs_parquet.feed.Feed` to Parquet or back to GTFS format."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import polars as pl

from gtfs_parquet.feed import Feed
from gtfs_parquet.schema import ALL_SCHEMAS, GtfsFileSchema, date_columns, time_columns


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _prepare_table(name: str, df: pl.DataFrame, compression: str, compression_level: int) -> bytes:
    """Sort a table by its schema sort keys and serialise to Parquet bytes."""
    schema = ALL_SCHEMAS.get(name)
    if schema and schema.sort_keys:
        keys = [k for k in schema.sort_keys if k in df.columns]
        if keys:
            df = df.sort(keys)
    buf = io.BytesIO()
    df.write_parquet(buf, compression=compression, compression_level=compression_level)
    return buf.getvalue()


def write_parquet(
    feed: Feed,
    path: str | Path,
    *,
    compression: str = "zstd",
    compression_level: int = 9,
) -> None:
    """Write all Feed tables as Parquet.

    The output format is determined by the file extension of *path*:

    * **directory** (no extension / existing dir) — one ``.parquet`` file per table.
    * **``.zip``** — a zip archive of ``.parquet`` files (stored without
      extra compression since Parquet already uses *compression* internally).
    * **``.tar``** — a tar archive of ``.parquet`` files (no extra
      compression, single-file distribution).

    Tables are sorted by their schema's
    :attr:`~gtfs_parquet.schema.GtfsFileSchema.sort_keys` before writing.

    Args:
        feed: The feed to write.
        path: Output path (directory, ``.zip``, or ``.tar``).
        compression: Parquet compression codec.
        compression_level: Compression level for the chosen codec.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            for name, df in feed.tables().items():
                data = _prepare_table(name, df, compression, compression_level)
                zf.writestr(f"{name}.parquet", data)

    elif suffix == ".tar":
        with tarfile.open(path, "w") as tf:
            for name, df in feed.tables().items():
                data = _prepare_table(name, df, compression, compression_level)
                info = tarfile.TarInfo(name=f"{name}.parquet")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    else:
        # Default: directory of parquet files.
        path.mkdir(parents=True, exist_ok=True)
        for name, df in feed.tables().items():
            data = _prepare_table(name, df, compression, compression_level)
            (path / f"{name}.parquet").write_bytes(data)


def read_parquet(path: str | Path) -> Feed:
    """Read Parquet tables back into a Feed.

    Accepts a directory, ``.zip`` archive, or ``.tar`` archive of
    ``.parquet`` files.

    Args:
        path: Path to the directory, zip, or tar containing Parquet files.

    Returns:
        A populated :class:`~gtfs_parquet.feed.Feed`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    feed = Feed()

    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            for entry in zf.namelist():
                if not entry.endswith(".parquet"):
                    continue
                table_name = Path(entry).stem
                if hasattr(feed, table_name):
                    setattr(feed, table_name, pl.read_parquet(io.BytesIO(zf.read(entry))))

    elif suffix == ".tar":
        with tarfile.open(path, "r") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".parquet"):
                    continue
                table_name = Path(member.name).stem
                if hasattr(feed, table_name):
                    f = tf.extractfile(member)
                    if f is not None:
                        setattr(feed, table_name, pl.read_parquet(io.BytesIO(f.read())))

    else:
        # Default: directory of parquet files.
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

"""Parse GTFS feeds (zip or directory) into a :class:`~gtfs_parquet.feed.Feed`.

Supports zip files, unzipped directories, and individual CSV files.
All CSV data is read as strings and then cast to typed Polars columns
according to :mod:`gtfs_parquet.schema`.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import polars as pl

from gtfs_parquet.feed import Feed
from gtfs_parquet.schema import (
    ALL_SCHEMAS,
    GtfsFileSchema,
    date_columns,
    get_schema,
    time_columns,
)

logger = logging.getLogger(__name__)

# Zip entries whose uncompressed size exceeds this threshold are streamed
# into chunk files during decompression, so neither the full decompressed
# CSV nor a full Utf8 DataFrame ever exists in memory.
_LARGE_ENTRY_BYTES = 100 * 1024 * 1024  # 100 MB

# Number of CSV rows per chunk file when splitting large entries.
_CHUNK_ROWS = 2_000_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_gtfs(path: str | Path) -> Feed:
    """Parse a GTFS feed from a zip file or directory (auto-detected).

    Args:
        path: Path to a ``.zip`` file or an unzipped GTFS directory.

    Returns:
        A populated :class:`~gtfs_parquet.feed.Feed`.

    Raises:
        ValueError: If the path is neither a directory nor a zip file.
    """
    path = Path(path)
    if path.is_dir():
        return parse_gtfs_dir(path)
    if path.suffix == ".zip" or zipfile.is_zipfile(path):
        return parse_gtfs_zip(path)
    raise ValueError(f"Cannot determine GTFS format for: {path}")


def parse_gtfs_zip(path: str | Path) -> Feed:
    """Parse a GTFS zip file into a Feed.

    Large entries are stream-decompressed into small chunk files and each
    chunk is parsed + typed independently, so peak memory stays close to
    final-data-size + one-chunk overhead.

    Args:
        path: Path to the ``.zip`` file.

    Returns:
        A populated :class:`~gtfs_parquet.feed.Feed`.
    """
    path = Path(path)
    feed = Feed()
    with zipfile.ZipFile(path) as zf:
        names_in_zip = {info.filename: info for info in zf.infolist()}
        for table_name, schema in ALL_SCHEMAS.items():
            fname = schema.file_name
            if fname not in names_in_zip:
                continue
            info = names_in_zip[fname]
            if info.file_size > _LARGE_ENTRY_BYTES:
                df = _read_zip_entry_chunked(zf, fname, schema)
            else:
                df = _read_zip_entry_in_memory(zf, fname, schema)
            setattr(feed, table_name, df)
    return feed


def parse_gtfs_dir(path: str | Path) -> Feed:
    """Parse a GTFS directory into a Feed.

    Args:
        path: Path to a directory containing GTFS ``.txt`` files.

    Returns:
        A populated :class:`~gtfs_parquet.feed.Feed`.
    """
    path = Path(path)
    feed = Feed()
    for table_name, schema in ALL_SCHEMAS.items():
        fpath = path / schema.file_name
        if not fpath.exists():
            continue
        df = _parse_csv_file(fpath, schema)
        setattr(feed, table_name, df)
    return feed


def parse_gtfs_file(source: str | Path | bytes, file_name: str) -> pl.DataFrame:
    """Parse a single GTFS file into a DataFrame.

    Args:
        source: File path or raw CSV bytes.
        file_name: GTFS file name (e.g. ``"stops.txt"``) to look up the schema.

    Returns:
        A typed :class:`polars.DataFrame`.

    Raises:
        ValueError: If *file_name* does not match any known GTFS schema.
    """
    schema = get_schema(file_name)
    if schema is None:
        raise ValueError(f"Unknown GTFS file: {file_name}")
    if isinstance(source, bytes):
        return _parse_csv_bytes(source, schema)
    return _parse_csv_file(Path(source), schema)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _read_zip_entry_chunked(
    zf: zipfile.ZipFile, fname: str, schema: GtfsFileSchema
) -> pl.DataFrame:
    """Stream-decompress a zip entry into chunk files, parse each, concat.

    Reads the zip entry line-by-line via a text wrapper. Every
    ``_CHUNK_ROWS`` lines, the accumulated rows are flushed to a temp file,
    parsed + typed with Polars, and the temp file is deleted — so neither
    the full decompressed CSV nor a full Utf8 DataFrame ever exists.
    """
    chunks: list[pl.DataFrame] = []
    with zf.open(fname) as raw:
        # Handle BOM
        bom = raw.read(3)
        if bom != b"\xef\xbb\xbf":
            raw.seek(0)
        reader = io.TextIOWrapper(raw, encoding="utf-8")
        header = reader.readline()

        buf: list[str] = [header]
        row_count = 0

        for line in reader:
            buf.append(line)
            row_count += 1
            if row_count >= _CHUNK_ROWS:
                chunks.append(_parse_chunk(buf, schema))
                buf = [header]
                row_count = 0

        # Final partial chunk
        if row_count > 0:
            chunks.append(_parse_chunk(buf, schema))

    return pl.concat(chunks)


def _parse_chunk(lines: list[str], schema: GtfsFileSchema) -> pl.DataFrame:
    """Write lines to a temp file, parse with Polars, delete, return typed DF."""
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        df = pl.read_csv(
            tmp_path, infer_schema_length=0, truncate_ragged_lines=True
        )
    finally:
        os.unlink(tmp_path)
    return _apply_schema(df, schema)


def _read_zip_entry_in_memory(
    zf: zipfile.ZipFile, fname: str, schema: GtfsFileSchema
) -> pl.DataFrame:
    """Read a small zip entry into memory and parse."""
    raw = zf.read(fname)
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    df = pl.read_csv(raw, infer_schema_length=0, truncate_ragged_lines=True)
    del raw
    return _apply_schema(df, schema)


def _parse_csv_bytes(raw: bytes, schema: GtfsFileSchema) -> pl.DataFrame:
    """Parse CSV bytes using the given GTFS schema."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    df = pl.read_csv(raw, infer_schema_length=0, truncate_ragged_lines=True)
    return _apply_schema(df, schema)


def _parse_csv_file(path: Path, schema: GtfsFileSchema) -> pl.DataFrame:
    """Parse a CSV file on disk using the given GTFS schema."""
    return _apply_schema(
        pl.read_csv(path, infer_schema_length=0, truncate_ragged_lines=True),
        schema,
    )


def _apply_schema(df: pl.DataFrame, schema: GtfsFileSchema) -> pl.DataFrame:
    """Cast a raw (all-Utf8) DataFrame to the proper GTFS types."""
    df = df.rename({c: c.strip() for c in df.columns})
    exprs = _build_cast_exprs(df.columns, schema)
    return df.select(exprs)


def _build_cast_exprs(
    columns: list[str], schema: GtfsFileSchema
) -> list[pl.Expr]:
    """Build the list of casting expressions for the given columns."""
    exprs: list[pl.Expr] = []
    known_cols = set(schema.columns.keys())
    date_cols = set(date_columns(schema))
    time_cols = set(time_columns(schema))

    for col_name in columns:
        if col_name not in known_cols:
            logger.debug(
                "%s: keeping unknown column %r as Utf8",
                schema.file_name,
                col_name,
            )
            exprs.append(pl.col(col_name))
            continue

        target_dtype = schema.columns[col_name]

        if col_name in date_cols:
            exprs.append(_parse_date_col(col_name))
        elif col_name in time_cols:
            exprs.append(_parse_time_col(col_name))
        elif target_dtype == pl.Utf8:
            exprs.append(pl.col(col_name))
        elif target_dtype.is_float():
            exprs.append(
                pl.col(col_name)
                .str.strip_chars()
                .replace("", None)
                .cast(target_dtype, strict=False)
            )
        elif target_dtype.is_integer():
            exprs.append(
                pl.col(col_name)
                .str.strip_chars()
                .replace("", None)
                .cast(target_dtype, strict=False)
            )
        else:
            exprs.append(pl.col(col_name).cast(target_dtype, strict=False))

    return exprs


def _parse_date_col(col_name: str) -> pl.Expr:
    """Parse a YYYYMMDD string column into pl.Date."""
    return (
        pl.col(col_name)
        .str.strip_chars()
        .replace("", None)
        .str.to_date("%Y%m%d", strict=False)
    )


def _parse_time_col(col_name: str) -> pl.Expr:
    """Parse HH:MM:SS string (can exceed 24h) into pl.Duration — fully vectorized.

    Uses ``split_exact`` into a struct (columnar storage) instead of ``split``
    into a list, which avoids allocating per-row list offset arrays.
    """
    col = pl.col(col_name).str.strip_chars().replace("", None)
    parts = col.str.split_exact(":", 2)
    h = parts.struct.field("field_0").cast(pl.Int64, strict=False)
    m = parts.struct.field("field_1").cast(pl.Int64, strict=False)
    s = parts.struct.field("field_2").cast(pl.Int64, strict=False)
    ms = h * 3_600_000 + m * 60_000 + s * 1_000
    return ms.cast(pl.Duration("ms")).alias(col_name)

"""Convert a GTFS zip to Parquet without loading the whole feed in memory."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import IO, Iterator

import polars as pl

from gtfs_parquet.parse import _apply_schema
from gtfs_parquet.schema import ALL_SCHEMAS, GtfsFileSchema

# Zip entries larger than this are converted block by block
_BLOCK_BYTES = 16 * 1024 * 1024  # 16 MB


def convert_gtfs_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    compression: str = "zstd",
    compression_level: int = 9,
    sort: bool = False,
    block_bytes: int = _BLOCK_BYTES,
) -> dict[str, Path]:
    """Convert a GTFS zip to one Parquet file per table, streaming large files.

    Unlike :func:`~gtfs_parquet.parse_gtfs_zip` followed by
    :func:`~gtfs_parquet.write_parquet`, the feed is never fully in memory:
    tables are converted one at a time, and files larger than *block_bytes*
    are read in blocks that are typed and written separately, then merged into
    one Parquet file by Polars' streaming engine. On the German national feed
    (2.3 GB ``stop_times.txt``) this peaks around 0.5 GB with
    ``POLARS_MAX_THREADS=4``, against about 10 GB for the in-memory path.
    Memory grows with the number of Polars threads.

    Rows keep the order of the source file, which usually compresses as well
    as or better than sorting. With ``sort=True``, tables are sorted by their
    schema's sort keys; large tables are then sorted block by block only.

    Args:
        zip_path: Path to the GTFS ``.zip`` file.
        out_dir: Directory to write ``<table>.parquet`` files to (created if missing).
        compression: Parquet compression codec.
        compression_level: Compression level for the chosen codec.
        sort: Sort rows by the schema's sort keys.
        block_bytes: Size above which a file is converted block by block, and
            the size of those blocks.

    Returns:
        A dict mapping each table name (e.g. ``"stops"``) to its Parquet file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    with zipfile.ZipFile(zip_path) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        for table_name, schema in ALL_SCHEMAS.items():
            info = infos.get(schema.file_name)
            if info is None:
                continue
            out = out_dir / f"{table_name}.parquet"
            if info.file_size <= block_bytes:
                df = _read_block(_strip_bom(zf.read(info)), schema, sort)
                df.write_parquet(
                    out, compression=compression, compression_level=compression_level
                )
            else:
                with zf.open(info) as raw, tempfile.TemporaryDirectory(dir=out_dir) as tmp:
                    parts = []
                    for i, block in enumerate(_csv_blocks(raw, block_bytes)):
                        part = Path(tmp) / f"{i:06}.parquet"
                        _read_block(block, schema, sort).write_parquet(
                            part, compression="uncompressed"
                        )
                        parts.append(part)
                    pl.scan_parquet(parts).sink_parquet(
                        out, compression=compression, compression_level=compression_level
                    )
            written[table_name] = out

    return written


def _read_block(data: bytes, schema: GtfsFileSchema, sort: bool) -> pl.DataFrame:
    """Parse CSV bytes (header included) into a typed DataFrame."""
    df = _apply_schema(
        pl.read_csv(data, infer_schema=False, truncate_ragged_lines=True), schema
    )
    if sort:
        keys = [k for k in schema.sort_keys if k in df.columns]
        if keys:
            df = df.sort(keys)
    return df


def _strip_bom(data: bytes) -> bytes:
    return data.removeprefix(b"\xef\xbb\xbf")


def _csv_blocks(raw: IO[bytes], block_bytes: int) -> Iterator[bytes]:
    """Yield blocks of whole CSV rows, each starting with the header line.

    A block ends at a line break that is not inside a quoted field: CSV escapes
    a quote by doubling it, so a line break is outside quotes when the text
    before it holds an even number of quote characters.
    """
    header = _strip_bom(raw.readline())
    if not header.endswith(b"\n"):
        header += b"\n"
    pending = b""
    while True:
        data = raw.read(block_bytes)
        pending += data
        if not data:
            if pending.strip():
                yield header + pending
            return
        cut = _last_row_end(pending)
        if cut:
            yield header + pending[:cut]
            pending = pending[cut:]


def _last_row_end(data: bytes) -> int:
    """Index just after the last line break outside quotes, or 0 if none."""
    inside = data.count(b'"') % 2 == 1  # state at the end of data
    end = len(data)
    while True:
        newline = data.rfind(b"\n", 0, end)
        if newline == -1:
            return 0
        # Quotes between this line break and the previous candidate flip the state
        inside ^= data.count(b'"', newline + 1, end) % 2 == 1
        if not inside:
            return newline + 1
        end = newline

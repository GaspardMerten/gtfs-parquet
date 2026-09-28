"""Tests for convert_gtfs_zip (streaming conversion)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import pytest

from gtfs_parquet import convert_gtfs_zip, parse_gtfs_zip
from gtfs_parquet.convert import _csv_blocks, _last_row_end

FILES = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\n"
    "A1,Test Agency,https://example.com,Europe/Brussels\n",
    "stops.txt": "stop_id,stop_name,stop_desc,stop_lat,stop_lon\n"
    'S2,"Stop, Two","line one\nline two",50.85,4.35\n'
    "S1,Stop One,,50.84,4.36\n",
    "routes.txt": "route_id,agency_id,route_short_name,route_type\nR1,A1,1,3\n",
    "trips.txt": "route_id,service_id,trip_id\nR1,WD,T2\nR1,WD,T1\n",
    "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
    + "".join(
        f"T{t},{8 + s:02}:00:00,{8 + s:02}:00:30,S{s % 2 + 1},{s}\n"
        for t in (2, 1)
        for s in range(1, 40)
    ),
    "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WD,1,1,1,1,1,0,0,20260101,20261231\n",
}


def _zip(tmp_path: Path, files=FILES, bom=False) -> Path:
    path = tmp_path / "feed.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            data = content.encode()
            zf.writestr(name, (b"\xef\xbb\xbf" + data) if bom else data)
    return path


@pytest.mark.parametrize("block_bytes", [64, 1 << 20])
def test_same_content_as_in_memory_parse(tmp_path: Path, block_bytes: int):
    path = _zip(tmp_path)
    paths = convert_gtfs_zip(path, tmp_path / "out", block_bytes=block_bytes)
    feed = parse_gtfs_zip(path)

    assert set(paths) == set(feed.tables())
    for name, df in feed.tables().items():
        converted = pl.read_parquet(paths[name])
        assert converted.schema == df.schema, name
        assert converted.equals(df), name  # same rows, same (file) order


def test_many_blocks_same_content(tmp_path: Path):
    files = dict(FILES)
    files["stop_times.txt"] = "trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign\n" + "".join(
        f'T{t},{8 + s % 16:02}:{s % 60:02}:00,{8 + s % 16:02}:{s % 60:02}:30,S{s % 2 + 1},{s},"To ""Centre""\nPlatform {t}"\n'
        for t in range(3000)
        for s in range(1, 41)
    )
    path = _zip(tmp_path, files)
    paths = convert_gtfs_zip(path, tmp_path / "out", block_bytes=64 * 1024)
    expected = parse_gtfs_zip(path).stop_times
    assert expected.height == 120_000
    assert pl.read_parquet(paths["stop_times"]).equals(expected)


def test_quoted_newline_survives_block_split(tmp_path: Path):
    paths = convert_gtfs_zip(_zip(tmp_path), tmp_path / "out", block_bytes=64)
    stops = pl.read_parquet(paths["stops"])
    assert stops["stop_desc"].to_list() == ["line one\nline two", None]
    assert stops["stop_name"].to_list() == ["Stop, Two", "Stop One"]


def test_bom(tmp_path: Path):
    paths = convert_gtfs_zip(_zip(tmp_path, bom=True), tmp_path / "out", block_bytes=64)
    assert "trip_id" in pl.read_parquet(paths["stop_times"]).columns
    assert "agency_id" in pl.read_parquet(paths["agency"]).columns


def test_sort(tmp_path: Path):
    paths = convert_gtfs_zip(_zip(tmp_path), tmp_path / "out", sort=True)
    trips = pl.read_parquet(paths["trips"])
    assert trips["trip_id"].to_list() == ["T1", "T2"]
    stop_times = pl.read_parquet(paths["stop_times"])
    assert stop_times["trip_id"][0] == "T1"


def test_no_temporary_files_left(tmp_path: Path):
    out = tmp_path / "out"
    convert_gtfs_zip(_zip(tmp_path), out, block_bytes=64)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{name.removesuffix('.txt')}.parquet" for name in FILES
    )


def test_csv_blocks_keep_every_row():
    rows = [f'{i},"a\nb",x\n'.encode() for i in range(20_000)]
    raw = io.BytesIO(b"id,text,other\n" + b"".join(rows))
    blocks = list(_csv_blocks(raw, 4096))
    assert len(blocks) > 1
    assert all(block.startswith(b"id,text,other\n") for block in blocks)
    total = pl.concat([pl.read_csv(block, infer_schema=False) for block in blocks])
    assert total["id"].to_list() == [str(i) for i in range(20_000)]
    assert set(total["text"]) == {"a\nb"}


def test_unescaped_quote_fails_without_reading_everything():
    good = b"".join(f"{i},x\n".encode() for i in range(1000))
    raw = io.BytesIO(b"id,text\n" + b'0,12" Street\n' + good * 50)
    with pytest.raises(ValueError, match="unescaped quote"):
        for _ in _csv_blocks(raw, 1024):
            pass


def test_cr_only_line_endings(tmp_path: Path):
    files = {
        name: content.replace("\n", "\r") if name == "stop_times.txt" else content
        for name, content in FILES.items()
    }
    paths = convert_gtfs_zip(_zip(tmp_path, files), tmp_path / "out", block_bytes=64)
    stop_times = pl.read_parquet(paths["stop_times"])
    assert stop_times.height == 78
    assert "trip_id" in stop_times.columns


def test_extended_route_types(tmp_path: Path):
    files = dict(FILES)
    files["routes.txt"] = "route_id,agency_id,route_short_name,route_type\nR1,A1,1,3\nR2,A1,2,700\nR3,A1,3,1501\nR4,A1,4,109\n"
    path = _zip(tmp_path, files)
    for routes in (
        pl.read_parquet(convert_gtfs_zip(path, tmp_path / "out")["routes"]),
        parse_gtfs_zip(path).routes,
    ):
        assert routes["route_type"].to_list() == [3, 700, 1501, 109]


def test_last_row_end():
    assert _last_row_end(b"a,b\nc,d") == 4
    assert _last_row_end(b'a,"b\nc",d\ne') == 10
    assert _last_row_end(b'a,"b\nc') == 0
    assert _last_row_end(b'x,"y""z"\n"p\nq') == 9

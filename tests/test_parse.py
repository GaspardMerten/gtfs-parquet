"""Tests for gtfs-parquet: parsing, roundtrip, and real-world feeds."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from gtfs_parquet import Feed, parse_gtfs, parse_gtfs_zip, read_parquet, write_gtfs, write_parquet


# ---------------------------------------------------------------------------
# Fixtures: download real GTFS feeds once per session
# ---------------------------------------------------------------------------

FEED_URLS = {
    # MBTA (Boston) — well-known US feed, ~33 MB
    "mbta": "https://cdn.mbta.com/MBTA_GTFS.zip",
    # FlixBus EU — large European feed, ~54 MB
    "flixbus": "https://gtfs.gis.flix.tech/gtfs_generic_eu.zip",
    # De Lijn (Belgium) — ~200 MB
    "delijn": "https://gtfs.irail.be/de-lijn/de_lijn-gtfs.zip",
}


@pytest.fixture(scope="session")
def feed_cache(tmp_path_factory) -> dict[str, Path]:
    """Download GTFS feeds and cache them for the session."""
    import httpx

    cache_dir = tmp_path_factory.mktemp("gtfs_feeds")
    paths: dict[str, Path] = {}
    for name, url in FEED_URLS.items():
        dest = cache_dir / f"{name}.zip"
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            paths[name] = dest
            print(f"Downloaded {name}: {len(resp.content) / 1e6:.1f} MB")
        except Exception as e:
            print(f"Could not download {name}: {e}")
    if not paths:
        pytest.skip("No GTFS feeds could be downloaded")
    return paths


# ---------------------------------------------------------------------------
# Unit tests with synthetic data
# ---------------------------------------------------------------------------


class TestSyntheticRoundtrip:
    """Test parse/write roundtrip with minimal synthetic GTFS data."""

    def _make_synthetic_gtfs_dir(self, tmp_path: Path) -> Path:
        """Create a minimal valid GTFS directory."""
        d = tmp_path / "gtfs"
        d.mkdir()
        (d / "agency.txt").write_text(
            "agency_id,agency_name,agency_url,agency_timezone\n"
            "A1,Test Agency,https://example.com,America/New_York\n"
        )
        (d / "stops.txt").write_text(
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "S1,Stop One,42.3601,-71.0589\n"
            "S2,Stop Two,42.3611,-71.0578\n"
        )
        (d / "routes.txt").write_text(
            "route_id,agency_id,route_short_name,route_type\n"
            "R1,A1,Red,3\n"
        )
        (d / "trips.txt").write_text(
            "route_id,service_id,trip_id\n"
            "R1,WD,T1\n"
        )
        (d / "stop_times.txt").write_text(
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
            "T1,08:00:00,08:00:00,S1,1\n"
            "T1,25:30:00,25:30:00,S2,2\n"
        )
        (d / "calendar.txt").write_text(
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
            "WD,1,1,1,1,1,0,0,20240101,20241231\n"
        )
        return d

    def test_parse_dir(self, tmp_path: Path):
        d = self._make_synthetic_gtfs_dir(tmp_path)
        feed = parse_gtfs(d)

        assert feed.agency is not None
        assert feed.agency.shape[0] == 1
        assert feed.stops.shape[0] == 2
        assert feed.stop_times.shape[0] == 2

        # Check time parsing: 25:30:00 = 25*3600+30*60 = 91800 seconds
        st = feed.stop_times.sort("stop_sequence")
        t2_arrival = st["arrival_time"][1]
        assert t2_arrival.total_seconds() == 91800

        # Check date parsing
        cal = feed.calendar
        assert cal["start_date"].dtype == pl.Date
        assert str(cal["start_date"][0]) == "2024-01-01"

    def test_parse_zip(self, tmp_path: Path):
        d = self._make_synthetic_gtfs_dir(tmp_path)
        zip_path = tmp_path / "test.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in d.iterdir():
                zf.write(f, f.name)
        feed = parse_gtfs(zip_path)
        assert feed.agency is not None
        assert feed.stop_times.shape[0] == 2

    def test_parquet_roundtrip(self, tmp_path: Path):
        d = self._make_synthetic_gtfs_dir(tmp_path)
        feed = parse_gtfs(d)

        pq_dir = tmp_path / "parquet_out"
        write_parquet(feed, pq_dir)

        feed2 = read_parquet(pq_dir)
        for name, df in feed.tables().items():
            df2 = feed2.tables()[name]
            assert df.shape == df2.shape, f"{name} shape mismatch"
            assert df.columns == df2.columns, f"{name} columns mismatch"

    def test_gtfs_roundtrip(self, tmp_path: Path):
        d = self._make_synthetic_gtfs_dir(tmp_path)
        feed = parse_gtfs(d)

        zip_out = tmp_path / "roundtrip.zip"
        write_gtfs(feed, zip_out)
        feed2 = parse_gtfs_zip(zip_out)

        for name, df in feed.tables().items():
            df2 = feed2.tables()[name]
            assert df.shape == df2.shape, f"{name} shape mismatch after roundtrip"

    def test_validate(self, tmp_path: Path):
        d = self._make_synthetic_gtfs_dir(tmp_path)
        feed = parse_gtfs(d)
        errors = feed.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_missing_required(self):
        feed = Feed()
        errors = feed.validate()
        assert any("agency.txt" in e for e in errors)


# ---------------------------------------------------------------------------
# Integration tests: real feeds
# ---------------------------------------------------------------------------


class TestRealFeeds:
    """Integration tests with real-world GTFS feeds."""

    @pytest.mark.parametrize("feed_name", list(FEED_URLS.keys()))
    def test_parse_real_feed(self, feed_cache: dict[str, Path], feed_name: str):
        if feed_name not in feed_cache:
            pytest.skip(f"{feed_name} feed not available")

        feed = parse_gtfs_zip(feed_cache[feed_name])

        # Every real feed should have these core tables
        assert feed.agency is not None, f"{feed_name}: missing agency"
        assert feed.routes is not None, f"{feed_name}: missing routes"
        assert feed.trips is not None, f"{feed_name}: missing trips"
        assert feed.stop_times is not None, f"{feed_name}: missing stop_times"

        # Should have non-zero rows
        assert feed.agency.shape[0] > 0
        assert feed.routes.shape[0] > 0
        assert feed.trips.shape[0] > 0
        assert feed.stop_times.shape[0] > 0

        # Types should be correct
        assert feed.stop_times["arrival_time"].dtype == pl.Duration("ms")
        if feed.stops is not None and "stop_lat" in feed.stops.columns:
            assert feed.stops["stop_lat"].dtype == pl.Float32

        print(f"\n{feed_name}:")
        print(feed)

    @pytest.mark.parametrize("feed_name", list(FEED_URLS.keys()))
    def test_parquet_roundtrip_real(self, feed_cache: dict[str, Path], feed_name: str, tmp_path: Path):
        if feed_name not in feed_cache:
            pytest.skip(f"{feed_name} feed not available")

        feed = parse_gtfs_zip(feed_cache[feed_name])

        pq_dir = tmp_path / "pq"
        write_parquet(feed, pq_dir)
        feed2 = read_parquet(pq_dir)

        for name, df in feed.tables().items():
            assert name in feed2.tables(), f"{feed_name}: {name} missing after parquet roundtrip"
            df2 = feed2.tables()[name]
            assert df.shape == df2.shape, f"{feed_name}/{name}: shape mismatch {df.shape} vs {df2.shape}"

    @pytest.mark.parametrize("feed_name", list(FEED_URLS.keys()))
    def test_gtfs_roundtrip_real(self, feed_cache: dict[str, Path], feed_name: str, tmp_path: Path):
        if feed_name not in feed_cache:
            pytest.skip(f"{feed_name} feed not available")

        feed = parse_gtfs_zip(feed_cache[feed_name])

        zip_out = tmp_path / "out.zip"
        write_gtfs(feed, zip_out)
        feed2 = parse_gtfs_zip(zip_out)

        for name, df in feed.tables().items():
            assert name in feed2.tables(), f"{feed_name}: {name} missing after GTFS roundtrip"
            df2 = feed2.tables()[name]
            assert df.shape == df2.shape, f"{feed_name}/{name}: shape mismatch {df.shape} vs {df2.shape}"

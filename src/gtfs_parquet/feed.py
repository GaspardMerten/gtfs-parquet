"""Feed class — container for all GTFS tables as Polars DataFrames.

The :class:`Feed` dataclass is the central object in gtfs-parquet.
Each attribute corresponds to a GTFS file and holds either a
:class:`polars.DataFrame` or ``None`` (file not present).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields

import polars as pl

from gtfs_parquet.schema import ALL_SCHEMAS


@dataclass
class Feed:
    """A GTFS feed represented as Polars DataFrames.

    Each attribute corresponds to a GTFS file (without the .txt extension).
    ``None`` means the file was not present in the feed.
    """

    agency: pl.DataFrame | None = None
    stops: pl.DataFrame | None = None
    routes: pl.DataFrame | None = None
    trips: pl.DataFrame | None = None
    stop_times: pl.DataFrame | None = None
    calendar: pl.DataFrame | None = None
    calendar_dates: pl.DataFrame | None = None
    fare_attributes: pl.DataFrame | None = None
    fare_rules: pl.DataFrame | None = None
    timeframes: pl.DataFrame | None = None
    rider_categories: pl.DataFrame | None = None
    fare_media: pl.DataFrame | None = None
    fare_products: pl.DataFrame | None = None
    fare_leg_rules: pl.DataFrame | None = None
    fare_leg_join_rules: pl.DataFrame | None = None
    fare_transfer_rules: pl.DataFrame | None = None
    shapes: pl.DataFrame | None = None
    frequencies: pl.DataFrame | None = None
    transfers: pl.DataFrame | None = None
    pathways: pl.DataFrame | None = None
    levels: pl.DataFrame | None = None
    areas: pl.DataFrame | None = None
    stop_areas: pl.DataFrame | None = None
    networks: pl.DataFrame | None = None
    route_networks: pl.DataFrame | None = None
    location_groups: pl.DataFrame | None = None
    location_group_stops: pl.DataFrame | None = None
    booking_rules: pl.DataFrame | None = None
    translations: pl.DataFrame | None = None
    feed_info: pl.DataFrame | None = None
    attributions: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def tables(self) -> dict[str, pl.DataFrame]:
        """Return a dict of all non-``None`` tables keyed by GTFS file name (without ``.txt``)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def validate(self) -> list[str]:
        """Run basic validation.

        Checks that required files are present and that required columns
        exist in each table.

        Returns:
            A list of error messages (empty means valid).
        """
        errors: list[str] = []
        for name, schema in ALL_SCHEMAS.items():
            if schema.presence == "required" and getattr(self, name, None) is None:
                errors.append(f"Required file {schema.file_name} is missing")
        for name, df in self.tables().items():
            schema = ALL_SCHEMAS.get(name)
            if schema is None:
                continue
            missing = schema.required_columns - set(df.columns)
            if missing:
                errors.append(
                    f"{schema.file_name}: missing required columns: {', '.join(sorted(missing))}"
                )
        return errors

    # ------------------------------------------------------------------
    # Calendar / date operations
    # ------------------------------------------------------------------

    def get_dates(self) -> list[dt.date]:
        """All dates the feed covers, sorted ascending."""
        from gtfs_parquet.ops.calendar import get_dates
        return get_dates(self)

    def get_first_week(self) -> list[dt.date]:
        """First Monday–Sunday week of the feed's date range."""
        from gtfs_parquet.ops.calendar import get_first_week
        return get_first_week(self)

    def get_week(self, k: int) -> list[dt.date]:
        """The k-th Monday–Sunday week (0-indexed) of the feed's date range."""
        from gtfs_parquet.ops.calendar import get_week
        return get_week(self, k)

    def subset_dates(self, dates: list[dt.date]) -> list[dt.date]:
        """Filter dates to those within the feed's range."""
        from gtfs_parquet.ops.calendar import subset_dates
        return subset_dates(self, dates)

    def get_active_services(self, date: dt.date) -> list[str]:
        """Service IDs active on the given date."""
        from gtfs_parquet.ops.calendar import get_active_services
        return get_active_services(self, date)

    def compute_trip_activity(self, dates: list[dt.date]) -> pl.DataFrame:
        """Mark each trip as active (1) or inactive (0) on each date."""
        from gtfs_parquet.ops.calendar import compute_trip_activity
        return compute_trip_activity(self, dates)

    def compute_busiest_date(self, dates: list[dt.date]) -> dt.date:
        """Date with the most active trips."""
        from gtfs_parquet.ops.calendar import compute_busiest_date
        return compute_busiest_date(self, dates)

    # ------------------------------------------------------------------
    # Trip operations
    # ------------------------------------------------------------------

    def get_trips(self, date: dt.date | None = None) -> pl.DataFrame | None:
        """Get trips, optionally filtered to those active on *date*.

        Args:
            date: If given, only trips whose service is active on this date.

        Returns:
            A DataFrame of matching trips, or ``None`` if no trips table.
        """
        from gtfs_parquet.ops.trips import get_trips
        return get_trips(self, date)

    def compute_trip_stats(self, *, route_ids: list[str] | None = None) -> pl.DataFrame:
        """Compute per-trip statistics.

        Computes: *num_stops*, *duration_h*, *distance*, *speed*, *is_loop*, etc.

        Args:
            route_ids: If given, restrict to trips on these routes.

        Returns:
            A DataFrame with one row per trip.
        """
        from gtfs_parquet.ops.trips import compute_trip_stats
        return compute_trip_stats(self, route_ids=route_ids)

    # ------------------------------------------------------------------
    # Route operations
    # ------------------------------------------------------------------

    def get_routes(self, date: dt.date | None = None) -> pl.DataFrame | None:
        """Get routes, optionally filtered by date."""
        from gtfs_parquet.ops.routes import get_routes
        return get_routes(self, date)

    def build_route_timetable(self, route_id: str, dates: list[dt.date]) -> pl.DataFrame:
        """Full timetable for a route on given dates."""
        from gtfs_parquet.ops.routes import build_route_timetable
        return build_route_timetable(self, route_id, dates)

    def compute_route_stats(
        self,
        dates: list[dt.date],
        trip_stats: pl.DataFrame | None = None,
        *,
        split_directions: bool = False,
    ) -> pl.DataFrame:
        """Per-route per-date statistics."""
        from gtfs_parquet.ops.routes import compute_route_stats
        return compute_route_stats(self, dates, trip_stats, split_directions=split_directions)

    # ------------------------------------------------------------------
    # Stop operations
    # ------------------------------------------------------------------

    def get_stops(
        self,
        date: dt.date | None = None,
        route_ids: list[str] | None = None,
        trip_ids: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """Get stops, optionally filtered by date/route/trip."""
        from gtfs_parquet.ops.stops import get_stops
        return get_stops(self, date=date, route_ids=route_ids, trip_ids=trip_ids)

    def get_stop_times(self, date: dt.date | None = None) -> pl.DataFrame | None:
        """Get stop_times, optionally filtered by date."""
        from gtfs_parquet.ops.stops import get_stop_times
        return get_stop_times(self, date)

    def get_start_and_end_times(self, date: dt.date | None = None) -> tuple[int | None, int | None]:
        """(first_departure_seconds, last_arrival_seconds) from stop_times."""
        from gtfs_parquet.ops.stops import get_start_and_end_times
        return get_start_and_end_times(self, date)

    def build_stop_timetable(self, stop_id: str, dates: list[dt.date]) -> pl.DataFrame:
        """Full timetable for a stop on given dates."""
        from gtfs_parquet.ops.stops import build_stop_timetable
        return build_stop_timetable(self, stop_id, dates)

    def compute_stop_activity(self, dates: list[dt.date]) -> pl.DataFrame:
        """Mark each stop as active (1) or not (0) on each date."""
        from gtfs_parquet.ops.stops import compute_stop_activity
        return compute_stop_activity(self, dates)

    def compute_stop_stats(
        self,
        dates: list[dt.date],
        stop_ids: list[str] | None = None,
        *,
        split_directions: bool = False,
    ) -> pl.DataFrame:
        """Per-stop per-date statistics."""
        from gtfs_parquet.ops.stops import compute_stop_stats
        return compute_stop_stats(self, dates, stop_ids, split_directions=split_directions)

    # ------------------------------------------------------------------
    # Network / summary
    # ------------------------------------------------------------------

    def describe(self, sample_date: dt.date | None = None) -> pl.DataFrame:
        """Summary description of the feed."""
        from gtfs_parquet.ops.network import describe
        return describe(self, sample_date)

    def compute_network_stats(
        self,
        dates: list[dt.date],
        trip_stats: pl.DataFrame | None = None,
        *,
        split_route_types: bool = False,
    ) -> pl.DataFrame:
        """Network-wide stats per date."""
        from gtfs_parquet.ops.network import compute_network_stats
        return compute_network_stats(self, dates, trip_stats, split_route_types=split_route_types)

    # ------------------------------------------------------------------
    # Restriction / subsetting
    # ------------------------------------------------------------------

    def restrict_to_trips(self, trip_ids: list[str]) -> Feed:
        """Return a new Feed restricted to the given trips and their dependencies.

        Args:
            trip_ids: Trip IDs to keep.
        """
        from gtfs_parquet.ops.restrict import restrict_to_trips
        return restrict_to_trips(self, trip_ids)

    def restrict_to_routes(self, route_ids: list[str]) -> Feed:
        """Return a new Feed restricted to trips on the given routes.

        Args:
            route_ids: Route IDs to keep.
        """
        from gtfs_parquet.ops.restrict import restrict_to_routes
        return restrict_to_routes(self, route_ids)

    def restrict_to_dates(self, dates: list[dt.date]) -> Feed:
        """Return a new Feed restricted to trips active on any of the given dates.

        Args:
            dates: Dates to keep.
        """
        from gtfs_parquet.ops.restrict import restrict_to_dates
        return restrict_to_dates(self, dates)

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> Feed:
        """Apply all cleaning steps. Returns a new Feed."""
        from gtfs_parquet.ops.clean import clean
        return clean(self)

    def clean_ids(self) -> Feed:
        """Strip whitespace from string ID columns."""
        from gtfs_parquet.ops.clean import clean_ids
        return clean_ids(self)

    def drop_zombies(self) -> Feed:
        """Remove orphan records with no referencing records."""
        from gtfs_parquet.ops.clean import drop_zombies
        return drop_zombies(self)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, df in self.tables().items():
            parts.append(f"  {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        body = "\n".join(parts) if parts else "  (empty)"
        return f"Feed(\n{body}\n)"

"""Feed class — container for all GTFS tables as Polars DataFrames.

The :class:`Feed` dataclass is the central object in gtfs-parquet.
Each attribute corresponds to a GTFS file and holds either a
:class:`polars.DataFrame` or ``None`` (file not present).
"""

from __future__ import annotations

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

    def __repr__(self) -> str:
        parts = []
        for name, df in self.tables().items():
            parts.append(f"  {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        body = "\n".join(parts) if parts else "  (empty)"
        return f"Feed(\n{body}\n)"

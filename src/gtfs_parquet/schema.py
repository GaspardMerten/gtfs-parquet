"""GTFS schema definitions mapped to Polars dtypes.

Based on the `official GTFS Schedule Reference
<https://gtfs.org/documentation/schedule/reference/>`_.

Each GTFS file is represented by a :class:`GtfsFileSchema` instance that
declares column names, Polars dtypes, required columns, file presence,
and optional sort keys used during Parquet writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

# ---------------------------------------------------------------------------
# GTFS type → Polars dtype mapping
# ---------------------------------------------------------------------------
_STR = pl.Utf8
_DATE = pl.Date
_TIME = pl.Duration("ms")
_FLOAT = pl.Float64
_INT = pl.Int32
_ENUM = pl.Int8
# Float32 gives ~11cm precision at the equator — more than enough for transit.
_LAT = pl.Float32
_LON = pl.Float32
# Distances in meters/km: Float32 gives ~7 significant digits — plenty.
_DIST = pl.Float32
# Stop sequences rarely exceed a few hundred.
_SEQ = pl.Int16


@dataclass(frozen=True)
class GtfsFileSchema:
    """Schema for a single GTFS file.

    Attributes:
        file_name: GTFS file name including ``.txt`` extension.
        columns: Mapping of column name to Polars dtype.
        required_columns: Columns that must be present per the GTFS spec.
        presence: Whether the file is required, optional, etc.
        sort_keys: Column names used to sort rows before Parquet writing.
    """

    file_name: str
    columns: dict[str, pl.DataType] = field(default_factory=dict)
    required_columns: set[str] = field(default_factory=set)
    presence: Literal["required", "optional", "conditionally_required", "conditionally_forbidden"] = "optional"
    sort_keys: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# All GTFS file schemas
# ---------------------------------------------------------------------------

AGENCY = GtfsFileSchema(
    file_name="agency.txt",
    presence="required",
    columns={
        "agency_id": _STR,
        "agency_name": _STR,
        "agency_url": _STR,
        "agency_timezone": _STR,
        "agency_lang": _STR,
        "agency_phone": _STR,
        "agency_fare_url": _STR,
        "agency_email": _STR,
    },
    required_columns={"agency_name", "agency_url", "agency_timezone"},
)

STOPS = GtfsFileSchema(
    file_name="stops.txt",
    presence="conditionally_required",
    columns={
        "stop_id": _STR,
        "stop_code": _STR,
        "stop_name": _STR,
        "tts_stop_name": _STR,
        "stop_desc": _STR,
        "stop_lat": _LAT,
        "stop_lon": _LON,
        "zone_id": _STR,
        "stop_url": _STR,
        "location_type": _ENUM,
        "parent_station": _STR,
        "stop_timezone": _STR,
        "wheelchair_boarding": _ENUM,
        "level_id": _STR,
        "platform_code": _STR,
    },
    required_columns={"stop_id"},
    sort_keys=("stop_id",),
)

ROUTES = GtfsFileSchema(
    file_name="routes.txt",
    presence="required",
    columns={
        "route_id": _STR,
        "agency_id": _STR,
        "route_short_name": _STR,
        "route_long_name": _STR,
        "route_desc": _STR,
        "route_type": _ENUM,
        "route_url": _STR,
        "route_color": _STR,
        "route_text_color": _STR,
        "route_sort_order": _INT,
        "continuous_pickup": _ENUM,
        "continuous_drop_off": _ENUM,
        "network_id": _STR,
    },
    required_columns={"route_id", "route_type"},
    sort_keys=("route_id",),
)

TRIPS = GtfsFileSchema(
    file_name="trips.txt",
    presence="required",
    columns={
        "route_id": _STR,
        "service_id": _STR,
        "trip_id": _STR,
        "trip_headsign": _STR,
        "trip_short_name": _STR,
        "direction_id": _ENUM,
        "block_id": _STR,
        "shape_id": _STR,
        "wheelchair_accessible": _ENUM,
        "bikes_allowed": _ENUM,
    },
    required_columns={"route_id", "service_id", "trip_id"},
    sort_keys=("route_id", "service_id", "trip_id"),
)

STOP_TIMES = GtfsFileSchema(
    file_name="stop_times.txt",
    presence="required",
    columns={
        "trip_id": _STR,
        "arrival_time": _TIME,
        "departure_time": _TIME,
        "stop_id": _STR,
        "location_group_id": _STR,
        "location_id": _STR,
        "stop_sequence": _SEQ,
        "stop_headsign": _STR,
        "start_pickup_drop_off_window": _TIME,
        "end_pickup_drop_off_window": _TIME,
        "pickup_type": _ENUM,
        "drop_off_type": _ENUM,
        "continuous_pickup": _ENUM,
        "continuous_drop_off": _ENUM,
        "shape_dist_traveled": _DIST,
        "timepoint": _ENUM,
        "pickup_booking_rule_id": _STR,
        "drop_off_booking_rule_id": _STR,
    },
    required_columns={"trip_id", "stop_sequence"},
    sort_keys=("trip_id", "stop_sequence"),
)

CALENDAR = GtfsFileSchema(
    file_name="calendar.txt",
    presence="conditionally_required",
    columns={
        "service_id": _STR,
        "monday": _ENUM,
        "tuesday": _ENUM,
        "wednesday": _ENUM,
        "thursday": _ENUM,
        "friday": _ENUM,
        "saturday": _ENUM,
        "sunday": _ENUM,
        "start_date": _DATE,
        "end_date": _DATE,
    },
    required_columns={
        "service_id", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "start_date", "end_date",
    },
    sort_keys=("service_id",),
)

CALENDAR_DATES = GtfsFileSchema(
    file_name="calendar_dates.txt",
    presence="conditionally_required",
    columns={
        "service_id": _STR,
        "date": _DATE,
        "exception_type": _ENUM,
    },
    required_columns={"service_id", "date", "exception_type"},
    sort_keys=("service_id", "date"),
)

FARE_ATTRIBUTES = GtfsFileSchema(
    file_name="fare_attributes.txt",
    presence="optional",
    columns={
        "fare_id": _STR,
        "price": _FLOAT,
        "currency_type": _STR,
        "payment_method": _ENUM,
        "transfers": _ENUM,
        "agency_id": _STR,
        "transfer_duration": _INT,
    },
    required_columns={"fare_id", "price", "currency_type", "payment_method", "transfers"},
)

FARE_RULES = GtfsFileSchema(
    file_name="fare_rules.txt",
    presence="optional",
    columns={
        "fare_id": _STR,
        "route_id": _STR,
        "origin_id": _STR,
        "destination_id": _STR,
        "contains_id": _STR,
    },
    required_columns={"fare_id"},
)

TIMEFRAMES = GtfsFileSchema(
    file_name="timeframes.txt",
    presence="optional",
    columns={
        "timeframe_group_id": _STR,
        "start_time": _TIME,
        "end_time": _TIME,
        "service_id": _STR,
    },
    required_columns={"timeframe_group_id", "service_id"},
)

RIDER_CATEGORIES = GtfsFileSchema(
    file_name="rider_categories.txt",
    presence="optional",
    columns={
        "rider_category_id": _STR,
        "rider_category_name": _STR,
        "is_default_fare_category": _ENUM,
        "eligibility_url": _STR,
    },
    required_columns={"rider_category_id", "rider_category_name", "is_default_fare_category"},
)

FARE_MEDIA = GtfsFileSchema(
    file_name="fare_media.txt",
    presence="optional",
    columns={
        "fare_media_id": _STR,
        "fare_media_name": _STR,
        "fare_media_type": _ENUM,
    },
    required_columns={"fare_media_id", "fare_media_type"},
)

FARE_PRODUCTS = GtfsFileSchema(
    file_name="fare_products.txt",
    presence="optional",
    columns={
        "fare_product_id": _STR,
        "fare_product_name": _STR,
        "rider_category_id": _STR,
        "fare_media_id": _STR,
        "amount": _FLOAT,
        "currency": _STR,
    },
    required_columns={"fare_product_id", "amount", "currency"},
)

FARE_LEG_RULES = GtfsFileSchema(
    file_name="fare_leg_rules.txt",
    presence="optional",
    columns={
        "leg_group_id": _STR,
        "network_id": _STR,
        "from_area_id": _STR,
        "to_area_id": _STR,
        "from_timeframe_group_id": _STR,
        "to_timeframe_group_id": _STR,
        "fare_product_id": _STR,
        "rule_priority": _INT,
    },
    required_columns={"fare_product_id"},
)

FARE_LEG_JOIN_RULES = GtfsFileSchema(
    file_name="fare_leg_join_rules.txt",
    presence="optional",
    columns={
        "leg_group_id": _STR,
        "another_leg_group_id": _STR,
        "duration_limit": _INT,
        "fare_transfer_rule_id": _STR,
    },
    required_columns={"leg_group_id", "another_leg_group_id"},
)

FARE_TRANSFER_RULES = GtfsFileSchema(
    file_name="fare_transfer_rules.txt",
    presence="optional",
    columns={
        "from_leg_group_id": _STR,
        "to_leg_group_id": _STR,
        "transfer_count": _INT,
        "duration_limit": _INT,
        "duration_limit_type": _ENUM,
        "fare_transfer_type": _ENUM,
        "fare_product_id": _STR,
    },
    required_columns={"from_leg_group_id", "to_leg_group_id", "transfer_count", "fare_transfer_type"},
)

SHAPES = GtfsFileSchema(
    file_name="shapes.txt",
    presence="optional",
    columns={
        "shape_id": _STR,
        "shape_pt_lat": _LAT,
        "shape_pt_lon": _LON,
        "shape_pt_sequence": _INT,
        "shape_dist_traveled": _DIST,
    },
    required_columns={"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"},
    sort_keys=("shape_id", "shape_pt_sequence"),
)

FREQUENCIES = GtfsFileSchema(
    file_name="frequencies.txt",
    presence="optional",
    columns={
        "trip_id": _STR,
        "start_time": _TIME,
        "end_time": _TIME,
        "headway_secs": _INT,
        "exact_times": _ENUM,
    },
    required_columns={"trip_id", "start_time", "end_time", "headway_secs"},
)

TRANSFERS = GtfsFileSchema(
    file_name="transfers.txt",
    presence="optional",
    columns={
        "from_stop_id": _STR,
        "to_stop_id": _STR,
        "from_route_id": _STR,
        "to_route_id": _STR,
        "from_trip_id": _STR,
        "to_trip_id": _STR,
        "transfer_type": _ENUM,
        "min_transfer_time": _INT,
    },
    required_columns={"transfer_type"},
)

PATHWAYS = GtfsFileSchema(
    file_name="pathways.txt",
    presence="optional",
    columns={
        "pathway_id": _STR,
        "from_stop_id": _STR,
        "to_stop_id": _STR,
        "pathway_mode": _ENUM,
        "is_bidirectional": _ENUM,
        "length": _FLOAT,
        "traversal_time": _INT,
        "stair_count": _INT,
        "max_slope": _FLOAT,
        "min_width": _FLOAT,
        "signposted_as": _STR,
        "reversed_signposted_as": _STR,
    },
    required_columns={"pathway_id", "from_stop_id", "to_stop_id", "pathway_mode", "is_bidirectional"},
)

LEVELS = GtfsFileSchema(
    file_name="levels.txt",
    presence="conditionally_required",
    columns={
        "level_id": _STR,
        "level_index": _FLOAT,
        "level_name": _STR,
    },
    required_columns={"level_id", "level_index"},
)

AREAS = GtfsFileSchema(
    file_name="areas.txt",
    presence="optional",
    columns={
        "area_id": _STR,
        "area_name": _STR,
    },
    required_columns={"area_id"},
)

STOP_AREAS = GtfsFileSchema(
    file_name="stop_areas.txt",
    presence="optional",
    columns={
        "area_id": _STR,
        "stop_id": _STR,
    },
    required_columns={"area_id", "stop_id"},
)

NETWORKS = GtfsFileSchema(
    file_name="networks.txt",
    presence="conditionally_forbidden",
    columns={
        "network_id": _STR,
        "network_name": _STR,
    },
    required_columns={"network_id"},
)

ROUTE_NETWORKS = GtfsFileSchema(
    file_name="route_networks.txt",
    presence="conditionally_forbidden",
    columns={
        "network_id": _STR,
        "route_id": _STR,
    },
    required_columns={"network_id", "route_id"},
)

LOCATION_GROUPS = GtfsFileSchema(
    file_name="location_groups.txt",
    presence="optional",
    columns={
        "location_group_id": _STR,
        "location_group_name": _STR,
    },
    required_columns={"location_group_id", "location_group_name"},
)

LOCATION_GROUP_STOPS = GtfsFileSchema(
    file_name="location_group_stops.txt",
    presence="optional",
    columns={
        "location_group_id": _STR,
        "stop_id": _STR,
    },
    required_columns={"location_group_id", "stop_id"},
)

BOOKING_RULES = GtfsFileSchema(
    file_name="booking_rules.txt",
    presence="optional",
    columns={
        "booking_rule_id": _STR,
        "booking_type": _ENUM,
        "prior_notice_duration_min": _INT,
        "prior_notice_duration_max": _INT,
        "prior_notice_last_day": _INT,
        "prior_notice_last_time": _TIME,
        "prior_notice_start_day": _INT,
        "message": _STR,
        "pickup_message": _STR,
        "drop_off_message": _STR,
    },
    required_columns={"booking_rule_id", "booking_type"},
)

TRANSLATIONS = GtfsFileSchema(
    file_name="translations.txt",
    presence="optional",
    columns={
        "table_name": _STR,
        "field_name": _STR,
        "language": _STR,
        "translation": _STR,
        "record_id": _STR,
        "record_sub_id": _STR,
        "is_record_id_referenced": _ENUM,
    },
    required_columns={"table_name", "field_name", "language", "translation"},
)

FEED_INFO = GtfsFileSchema(
    file_name="feed_info.txt",
    presence="conditionally_required",
    columns={
        "feed_publisher_name": _STR,
        "feed_publisher_url": _STR,
        "feed_lang": _STR,
        "default_lang": _STR,
        "feed_start_date": _DATE,
        "feed_end_date": _DATE,
        "feed_version": _STR,
        "feed_contact_email": _STR,
        "feed_contact_url": _STR,
    },
    required_columns={"feed_publisher_name", "feed_publisher_url", "feed_lang"},
)

ATTRIBUTIONS = GtfsFileSchema(
    file_name="attributions.txt",
    presence="optional",
    columns={
        "attribution_id": _STR,
        "agency_id": _STR,
        "route_id": _STR,
        "trip_id": _STR,
        "organization_name": _STR,
        "is_producer": _ENUM,
        "is_operator": _ENUM,
        "is_authority": _ENUM,
        "attribution_url": _STR,
        "attribution_email": _STR,
        "attribution_phone": _STR,
    },
    required_columns={"organization_name"},
)

# ---------------------------------------------------------------------------
# Registry: file_name (without .txt) → schema
# ---------------------------------------------------------------------------
ALL_SCHEMAS: dict[str, GtfsFileSchema] = {
    s.file_name.removesuffix(".txt"): s
    for s in [
        AGENCY, STOPS, ROUTES, TRIPS, STOP_TIMES,
        CALENDAR, CALENDAR_DATES,
        FARE_ATTRIBUTES, FARE_RULES,
        TIMEFRAMES, RIDER_CATEGORIES, FARE_MEDIA, FARE_PRODUCTS,
        FARE_LEG_RULES, FARE_LEG_JOIN_RULES, FARE_TRANSFER_RULES,
        SHAPES, FREQUENCIES, TRANSFERS, PATHWAYS, LEVELS,
        AREAS, STOP_AREAS, NETWORKS, ROUTE_NETWORKS,
        LOCATION_GROUPS, LOCATION_GROUP_STOPS,
        BOOKING_RULES, TRANSLATIONS, FEED_INFO, ATTRIBUTIONS,
    ]
}


def get_schema(file_name: str) -> GtfsFileSchema | None:
    """Look up a schema by GTFS file name.

    Args:
        file_name: GTFS file name, with or without the ``.txt`` extension.

    Returns:
        The matching schema, or ``None`` if not found.
    """
    key = file_name.removesuffix(".txt")
    return ALL_SCHEMAS.get(key)


def time_columns(schema: GtfsFileSchema) -> list[str]:
    """Return column names that use the Duration (time) type.

    Args:
        schema: A GTFS file schema.
    """
    return [name for name, dtype in schema.columns.items() if dtype == _TIME]


def date_columns(schema: GtfsFileSchema) -> list[str]:
    """Return column names that use the Date type.

    Args:
        schema: A GTFS file schema.
    """
    return [name for name, dtype in schema.columns.items() if dtype == _DATE]

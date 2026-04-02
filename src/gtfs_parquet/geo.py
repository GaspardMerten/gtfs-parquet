"""Vectorized geographic operations on Polars columns."""

import polars as pl

from gtfs_parquet.constants import EARTH_RADIUS_M


def haversine_m(
    lat1: pl.Expr, lon1: pl.Expr,
    lat2: pl.Expr, lon2: pl.Expr,
) -> pl.Expr:
    """Haversine distance in meters between two WGS-84 point columns.

    All inputs and the return value are Polars expressions, so this
    operates in a fully vectorized fashion.

    Args:
        lat1: Latitude of the first point (degrees).
        lon1: Longitude of the first point (degrees).
        lat2: Latitude of the second point (degrees).
        lon2: Longitude of the second point (degrees).

    Returns:
        A Polars expression yielding the distance in meters.
        ``null`` propagates if any input coordinate is ``null``.
    """
    dlat = (lat2 - lat1).radians()
    dlon = (lon2 - lon1).radians()
    a = (
        (dlat / 2).sin().pow(2)
        + lat1.radians().cos() * lat2.radians().cos()
        * (dlon / 2).sin().pow(2)
    )
    return EARTH_RADIUS_M * 2.0 * a.sqrt().arcsin()

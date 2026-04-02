"""Shared constants for gtfs-parquet.

All constants are module-level and importable directly.
"""

import polars as pl

# Earth radius in meters (WGS84 mean)
EARTH_RADIUS_M = 6_371_000

# Time unit conversions
MS_PER_SECOND = 1_000
MS_PER_MINUTE = 60_000
MS_PER_HOUR = 3_600_000
SECONDS_PER_HOUR = 3_600

# Loop detection threshold: a trip is considered a loop when start and end
# stops are closer than this distance. Matches gtfs-kit's convention.
LOOP_DISTANCE_THRESHOLD_M = 400

# Distance units supported for conversions
DIST_UNITS = ("ft", "mi", "m", "km")

"""Read a NeTEx timetable (European profile, e.g. EPIP or the Italian profile) as GTFS tables.

The file is read in one streaming pass: each NeTEx object is reduced to the few
fields GTFS needs and cleared, so a national timetable of several hundred MB
stays well under 1 GB of memory. The tables are typed with the same schemas as
a parsed GTFS feed.

Mapping:

- ``Operator`` (or ``Authority``) → agency
- ``StopPlace`` → station (``location_type`` 1), its ``Quay`` → stop; a
  ``ScheduledStopPoint`` without a stop assignment becomes a stop itself
- ``Line`` → route; ``TransportMode`` gives the basic GTFS ``route_type``
- ``ServiceJourney`` → trip, its ``Name`` (or public code) as ``trip_short_name``
- ``TimetabledPassingTime`` → stop time, day offsets as times past 24:00
- ``DayType`` + ``DayTypeAssignment`` + ``OperatingPeriod`` /
  ``UicOperatingPeriod`` / ``OperatingDay`` → calendar_dates (one service per
  distinct set of dates)
- ``ServiceLink`` geometries along the journey pattern → shapes
"""

from __future__ import annotations

import gzip
import io
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import IO, Iterator
from xml.etree.ElementTree import Element, iterparse

import polars as pl

from gtfs_parquet.feed import Feed
from gtfs_parquet.parse import _apply_schema
from gtfs_parquet.schema import ALL_SCHEMAS

logger = logging.getLogger(__name__)

# Basic GTFS route types; modes with no GTFS equivalent fall back to bus
ROUTE_TYPES = {
    "tram": 0,
    "metro": 1,
    "rail": 2,
    "bus": 3,
    "coach": 3,
    "trolleyBus": 11,
    "water": 4,
    "ferry": 4,
    "cableway": 6,
    "telecabin": 6,
    "funicular": 7,
    "air": 1100,
    "taxi": 1500,
}

WEEKDAYS = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}
DAY_GROUPS = {
    "Weekdays": {0, 1, 2, 3, 4},
    "Weekend": {5, 6},
    "Everyday": set(range(7)),
}

# Objects read as a whole (and then cleared); everything they hold is read from them
_OBJECTS = {
    "Operator", "Authority", "StopPlace", "ScheduledStopPoint",
    "PassengerStopAssignment", "Line", "Route", "ServiceJourneyPattern",
    "JourneyPattern", "ServiceLink", "ServiceJourney", "DayType",
    "DayTypeAssignment", "OperatingPeriod", "UicOperatingPeriod",
    "OperatingDay", "DestinationDisplay", "FrameDefaults",
}


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def _child(el: Element, name: str) -> Element | None:
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el: Element | None, *path: str) -> str | None:
    for name in path:
        if el is None:
            return None
        el = _child(el, name)
    if el is None or el.text is None:
        return None
    return el.text.strip() or None


def _ref(el: Element | None, *path: str) -> str | None:
    for name in path:
        if el is None:
            return None
        el = _child(el, name)
    return None if el is None else el.get("ref")


def _location(el: Element) -> tuple[str | None, str | None]:
    """Longitude and latitude of a Centroid/Location or Location child."""
    loc = _child(el, "Centroid")
    loc = _child(loc, "Location") if loc is not None else _child(el, "Location")
    return _text(loc, "Longitude"), _text(loc, "Latitude")


def _false(value: str | None) -> bool:
    return value is not None and value.lower() == "false"


@dataclass
class _Pattern:
    line: str | None
    route: str | None
    destination: str | None
    # stop point in journey pattern id -> (order, scheduled stop point, boarding, alighting)
    points: dict[str, tuple[int, str, bool, bool]] = field(default_factory=dict)
    links: list[tuple[int, str]] = field(default_factory=list)  # (order, service link)


@dataclass
class _Journey:
    id: str
    name: str | None
    pattern: str | None
    line: str | None
    operator: str | None
    mode: str | None
    day_types: tuple[str, ...]
    # (stop point in journey pattern, arrival, departure) as "HH:MM:SS" past 24h
    times: list[tuple[str, str | None, str | None]]


class _Reader:
    def __init__(self) -> None:
        self.timezone: str | None = None
        self.operators: dict[str, tuple[str | None, str | None]] = {}
        self.stop_places: list[dict] = []
        self.stop_points: dict[str, tuple[str | None, str | None, str | None]] = {}
        self.assignments: dict[str, str] = {}  # scheduled stop point -> quay or stop place
        self.lines: dict[str, dict] = {}
        self.routes: dict[str, str | None] = {}  # route -> line
        self.patterns: dict[str, _Pattern] = {}
        self.links: dict[str, list[float]] = {}  # lat, lon, lat, lon, ...
        self.journeys: list[_Journey] = []
        self.day_weekdays: dict[str, set[int]] = {}
        # day type -> [(available, period or operating day or date)]
        self.day_assignments: dict[str, list[tuple[bool, str, str]]] = defaultdict(list)
        self.periods: dict[str, tuple[date, date, str | None]] = {}
        self.operating_days: dict[str, date] = {}
        self.destinations: dict[str, str | None] = {}

    def read(self, source: IO[bytes]) -> None:
        stack: list[Element] = []
        for event, el in iterparse(source, events=("start", "end")):
            if event == "start":
                stack.append(el)
                continue
            stack.pop()
            name = _local(el.tag)
            if name in _OBJECTS:
                getattr(self, f"_{name}")(el)
                el.clear()
                # Drop the cleared object from its parent too, so long lists don't pile up
                if stack:
                    try:
                        stack[-1].remove(el)
                    except ValueError:
                        pass

    def _FrameDefaults(self, el: Element) -> None:
        self.timezone = self.timezone or _text(el, "DefaultLocale", "TimeZone")

    def _Operator(self, el: Element) -> None:
        self.operators[el.get("id")] = (
            _text(el, "Name") or _text(el, "ShortName"),
            _text(el, "ContactDetails", "Url"),
        )

    _Authority = _Operator

    def _StopPlace(self, el: Element) -> None:
        lon, lat = _location(el)
        place = {
            "id": el.get("id"),
            "name": _text(el, "Name"),
            "code": _text(el, "PublicCode") or _text(el, "PrivateCode"),
            "lon": lon,
            "lat": lat,
            "quays": [],
        }
        quays = _child(el, "quays")
        for q in quays if quays is not None else []:
            if _local(q.tag) != "Quay":
                continue
            qlon, qlat = _location(q)
            place["quays"].append(
                {
                    "id": q.get("id"),
                    "name": _text(q, "Name"),
                    "platform": _text(q, "PublicCode"),
                    "lon": qlon or lon,
                    "lat": qlat or lat,
                }
            )
        self.stop_places.append(place)

    def _ScheduledStopPoint(self, el: Element) -> None:
        lon, lat = _location(el)
        self.stop_points[el.get("id")] = (_text(el, "Name"), lon, lat)

    def _PassengerStopAssignment(self, el: Element) -> None:
        point = _ref(el, "ScheduledStopPointRef")
        target = _ref(el, "QuayRef") or _ref(el, "StopPlaceRef")
        if point and target:
            self.assignments[point] = target

    def _Line(self, el: Element) -> None:
        self.lines[el.get("id")] = {
            "name": _text(el, "Name"),
            "short": _text(el, "ShortName") or _text(el, "PublicCode"),
            "mode": _text(el, "TransportMode"),
            "operator": _ref(el, "OperatorRef") or _ref(el, "AuthorityRef"),
            "colour": _text(el, "Presentation", "Colour"),
            "text_colour": _text(el, "Presentation", "TextColour"),
        }

    def _Route(self, el: Element) -> None:
        self.routes[el.get("id")] = _ref(el, "LineRef")

    def _ServiceJourneyPattern(self, el: Element) -> None:
        pattern = _Pattern(
            line=_ref(el, "RouteView", "LineRef"),
            route=_ref(el, "RouteRef"),
            destination=_ref(el, "DestinationDisplayRef"),
        )
        points = _child(el, "pointsInSequence")
        for p in points if points is not None else []:
            if _local(p.tag) != "StopPointInJourneyPattern":
                continue
            order = int(p.get("order") or len(pattern.points) + 1)
            stop = _ref(p, "ScheduledStopPointRef")
            if stop:
                pattern.points[p.get("id")] = (
                    order,
                    stop,
                    not _false(_text(p, "ForBoarding")),
                    not _false(_text(p, "ForAlighting")),
                )
            link = _ref(p, "OnwardServiceLinkRef")
            if link:
                pattern.links.append((order, link))
        links = _child(el, "linksInSequence")
        for p in links if links is not None else []:
            link = _ref(p, "ServiceLinkRef")
            if link:
                pattern.links.append((int(p.get("order") or len(pattern.links) + 1), link))
        self.patterns[el.get("id")] = pattern

    _JourneyPattern = _ServiceJourneyPattern

    def _ServiceLink(self, el: Element) -> None:
        for sub in el.iter():
            if _local(sub.tag) == "posList" and sub.text:
                # GML in EPSG:4326 lists latitude first
                self.links[el.get("id")] = [float(v) for v in sub.text.split()]
                return
            if _local(sub.tag) == "LineString":
                coords = [
                    float(v)
                    for pos in sub.iter()
                    if _local(pos.tag) == "pos" and pos.text
                    for v in pos.text.split()
                ]
                if coords:
                    self.links[el.get("id")] = coords
                    return

    def _ServiceJourney(self, el: Element) -> None:
        day_types = _child(el, "dayTypes")
        refs = tuple(
            r.get("ref")
            for r in (day_types if day_types is not None else [])
            if _local(r.tag) == "DayTypeRef"
        )
        times = []
        passing = _child(el, "passingTimes")
        for t in passing if passing is not None else []:
            if _local(t.tag) != "TimetabledPassingTime":
                continue
            point = _ref(t, "StopPointInJourneyPatternRef")
            if point is None:
                continue
            times.append(
                (
                    point,
                    _past_midnight(_text(t, "ArrivalTime"), _text(t, "ArrivalDayOffset")),
                    _past_midnight(_text(t, "DepartureTime"), _text(t, "DepartureDayOffset")),
                )
            )
        self.journeys.append(
            _Journey(
                id=el.get("id"),
                name=_text(el, "Name") or _text(el, "PublicCode") or _text(el, "PrivateCode"),
                pattern=_ref(el, "ServiceJourneyPatternRef") or _ref(el, "JourneyPatternRef"),
                line=_ref(el, "LineRef"),
                operator=_ref(el, "OperatorRef"),
                mode=_text(el, "TransportMode"),
                day_types=refs,
                times=times,
            )
        )

    def _DayType(self, el: Element) -> None:
        days: set[int] = set()
        for sub in el.iter():
            if _local(sub.tag) == "DaysOfWeek" and sub.text:
                for word in sub.text.split():
                    if word in WEEKDAYS:
                        days.add(WEEKDAYS[word])
                    days |= DAY_GROUPS.get(word, set())
        if days:
            self.day_weekdays[el.get("id")] = days

    def _DayTypeAssignment(self, el: Element) -> None:
        day_type = _ref(el, "DayTypeRef")
        if day_type is None:
            return
        available = not _false(_text(el, "isAvailable"))
        if (ref := _ref(el, "OperatingPeriodRef") or _ref(el, "UicOperatingPeriodRef")):
            self.day_assignments[day_type].append((available, "period", ref))
        elif (ref := _ref(el, "OperatingDayRef")):
            self.day_assignments[day_type].append((available, "day", ref))
        elif (value := _text(el, "Date")):
            self.day_assignments[day_type].append((available, "date", value[:10]))

    def _OperatingPeriod(self, el: Element) -> None:
        start = _text(el, "FromDate")
        end = _text(el, "ToDate")
        start_day = self.operating_days.get(_ref(el, "FromOperatingDayRef") or "")
        end_day = self.operating_days.get(_ref(el, "ToOperatingDayRef") or "")
        start_date = date.fromisoformat(start[:10]) if start else start_day
        end_date = date.fromisoformat(end[:10]) if end else end_day
        if start_date and end_date:
            self.periods[el.get("id")] = (start_date, end_date, _text(el, "ValidDayBits"))

    _UicOperatingPeriod = _OperatingPeriod

    def _OperatingDay(self, el: Element) -> None:
        value = _text(el, "CalendarDate")
        if value:
            self.operating_days[el.get("id")] = date.fromisoformat(value[:10])

    def _DestinationDisplay(self, el: Element) -> None:
        self.destinations[el.get("id")] = _text(el, "FrontText") or _text(el, "Name")

    def dates(self, day_type: str) -> set[date]:
        """The dates a day type runs on (its weekdays applied to its periods)."""
        weekdays = self.day_weekdays.get(day_type, set(range(7)))
        on: set[date] = set()
        off: set[date] = set()
        for available, kind, ref in self.day_assignments.get(day_type, []):
            days: set[date] = set()
            if kind == "period" and ref in self.periods:
                start, end, bits = self.periods[ref]
                n = (end - start).days + 1
                for i in range(n):
                    if bits is None or (i < len(bits) and bits[i] == "1"):
                        days.add(start + timedelta(days=i))
                days = {d for d in days if d.weekday() in weekdays}
            elif kind == "day" and ref in self.operating_days:
                days.add(self.operating_days[ref])
            elif kind == "date":
                days.add(date.fromisoformat(ref))
            (on if available else off).update(days)
        return on - off


def _past_midnight(time: str | None, offset: str | None) -> str | None:
    """'HH:MM:SS' plus a day offset, as GTFS time (hours past 24 on later days)."""
    if not time:
        return None
    h, m, s = (time.split(":") + ["0", "0"])[:3]
    hours = int(h) + 24 * int(offset or 0)
    return f"{hours:02d}:{int(m):02d}:{int(float(s)):02d}"


def _sources(path: Path) -> Iterator[IO[bytes]]:
    """The XML documents in a .xml, .xml.gz or .zip file (shared files first)."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            yield f
    elif magic == b"PK\x03\x04":
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                (n for n in zf.namelist() if n.lower().endswith(".xml")),
                # Nordic and French exports put the shared objects in files starting with "_"
                key=lambda n: (not Path(n).name.startswith("_"), n),
            )
            for name in names:
                with zf.open(name) as f:
                    yield f
    else:
        with open(path, "rb") as f:
            yield f


def _frame(rows: dict[str, list], table: str) -> pl.DataFrame:
    """Typed table from string columns, as if read from a GTFS file."""
    df = pl.DataFrame(rows, schema={c: pl.Utf8 for c in rows})
    return _apply_schema(df, ALL_SCHEMAS[table])


def parse_netex(path: str | Path, *, timezone: str | None = None) -> Feed:
    """Read a NeTEx timetable into a :class:`Feed` of GTFS tables.

    Args:
        path: A NeTEx ``.xml`` file, gzipped (``.xml.gz``), or a ``.zip`` of
            XML files (as Nordic or French exports are).
        timezone: Time zone of the timetable; defaults to the file's
            ``FrameDefaults`` time zone, then UTC.

    Returns:
        A Feed with agency, stops, routes, trips, stop_times, calendar_dates
        and, when the service links have geometry, shapes.
    """
    reader = _Reader()
    for source in _sources(Path(path)):
        reader.read(source)
    tz = timezone or reader.timezone
    if tz is None:
        logger.warning("No time zone in the NeTEx file: using UTC")
        tz = "UTC"
    return _build(reader, tz)


def _build(r: _Reader, tz: str) -> Feed:
    # Agency
    default_operator = next(iter(r.operators), None)
    agency = {"agency_id": [], "agency_name": [], "agency_url": [], "agency_timezone": []}
    for op_id, (name, url) in r.operators.items():
        agency["agency_id"].append(op_id)
        agency["agency_name"].append(name or op_id)
        agency["agency_url"].append(url)
        agency["agency_timezone"].append(tz)

    # Stops: stations and their quays, then stop points not assigned to either
    stops = {k: [] for k in (
        "stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon",
        "location_type", "parent_station", "platform_code",
    )}

    def add_stop(sid, code, name, lat, lon, location_type, parent, platform):
        for k, v in zip(stops, (sid, code, name, lat, lon, location_type, parent, platform)):
            stops[k].append(v)

    known = set()
    for place in r.stop_places:
        add_stop(place["id"], place["code"], place["name"], place["lat"], place["lon"],
                 "1" if place["quays"] else "0", None, None)
        known.add(place["id"])
        for q in place["quays"]:
            add_stop(q["id"], place["code"], q["name"] or place["name"], q["lat"], q["lon"],
                     "0", place["id"], q["platform"])
            known.add(q["id"])

    def stop_of(point: str) -> str:
        target = r.assignments.get(point)
        return target if target in known else point

    for point, (name, lon, lat) in r.stop_points.items():
        if stop_of(point) == point:
            add_stop(point, None, name, lat, lon, "0", None, None)

    # Routes
    routes = {k: [] for k in (
        "route_id", "agency_id", "route_short_name", "route_long_name",
        "route_type", "route_color", "route_text_color",
    )}
    route_type: dict[str, int] = {}
    for line_id, line in r.lines.items():
        rtype = ROUTE_TYPES.get(line["mode"] or "", 3)
        route_type[line_id] = rtype
        for k, v in zip(routes, (
            line_id, line["operator"] or default_operator, line["short"],
            line["name"], str(rtype), line["colour"], line["text_colour"],
        )):
            routes[k].append(v)

    # Services: one per distinct set of dates
    services: dict[frozenset, str] = {}
    by_day_types: dict[tuple, str | None] = {}
    calendar = {"service_id": [], "date": [], "exception_type": []}

    def service_of(day_types: tuple[str, ...]) -> str | None:
        if day_types in by_day_types:
            return by_day_types[day_types]
        days = frozenset().union(*(r.dates(d) for d in day_types)) if day_types else frozenset()
        sid = None
        if days:
            sid = services.get(days)
            if sid is None:
                sid = day_types[0] if len(day_types) == 1 else "+".join(day_types)
                services[days] = sid
                for d in sorted(days):
                    calendar["service_id"].append(sid)
                    calendar["date"].append(d.strftime("%Y%m%d"))
                    calendar["exception_type"].append("1")
        by_day_types[day_types] = sid
        return sid

    # Shapes: one per distinct sequence of service links with geometry
    shapes = {"shape_id": [], "shape_pt_lat": [], "shape_pt_lon": [], "shape_pt_sequence": []}
    shape_ids: dict[tuple[str, ...], str] = {}
    pattern_shape: dict[str, str | None] = {}

    def shape_of(pattern_id: str) -> str | None:
        if pattern_id in pattern_shape:
            return pattern_shape[pattern_id]
        pattern = r.patterns.get(pattern_id)
        links = tuple(l for _, l in sorted(pattern.links) if l in r.links) if pattern else ()
        sid = None
        if links:
            sid = shape_ids.get(links)
            if sid is None:
                sid = pattern_id
                shape_ids[links] = sid
                seq, last = 0, None
                for link in links:
                    coords = r.links[link]
                    for i in range(0, len(coords) - 1, 2):
                        point = (coords[i], coords[i + 1])
                        if point == last:
                            continue
                        last = point
                        seq += 1
                        shapes["shape_id"].append(sid)
                        shapes["shape_pt_lat"].append(repr(point[0]))
                        shapes["shape_pt_lon"].append(repr(point[1]))
                        shapes["shape_pt_sequence"].append(str(seq))
        pattern_shape[pattern_id] = sid
        return sid

    trips = {k: [] for k in (
        "route_id", "service_id", "trip_id", "trip_headsign", "trip_short_name", "shape_id",
    )}
    stop_times = {k: [] for k in (
        "trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
        "pickup_type", "drop_off_type",
    )}
    skipped = 0
    for j in r.journeys:
        pattern = r.patterns.get(j.pattern) if j.pattern else None
        line = j.line or (pattern and (pattern.line or r.routes.get(pattern.route)))
        service = service_of(j.day_types)
        if pattern is None or line is None or service is None or not j.times:
            skipped += 1
            continue
        if line not in r.lines:
            # A journey on a line not described: keep it with what the journey says
            r.lines[line] = {}
            rtype = ROUTE_TYPES.get(j.mode or "", 3)
            for k, v in zip(routes, (line, j.operator or default_operator, None, None,
                                     str(rtype), None, None)):
                routes[k].append(v)
        calls = sorted(
            (pattern.points[p] + (arr, dep) for p, arr, dep in j.times if p in pattern.points),
            key=lambda c: c[0],
        )
        if not calls:
            skipped += 1
            continue
        last_stop = r.stop_points.get(calls[-1][1], (None,))[0]
        trips["route_id"].append(line)
        trips["service_id"].append(service)
        trips["trip_id"].append(j.id)
        trips["trip_headsign"].append(r.destinations.get(pattern.destination) or last_stop)
        trips["trip_short_name"].append(j.name)
        trips["shape_id"].append(shape_of(j.pattern))
        for order, point, boarding, alighting, arr, dep in calls:
            stop_times["trip_id"].append(j.id)
            stop_times["arrival_time"].append(arr or dep)
            stop_times["departure_time"].append(dep or arr)
            stop_times["stop_id"].append(stop_of(point))
            stop_times["stop_sequence"].append(str(order))
            stop_times["pickup_type"].append("0" if boarding else "1")
            stop_times["drop_off_type"].append("0" if alighting else "1")
    if skipped:
        logger.warning("%d service journeys without pattern, line, dates or times skipped", skipped)

    return Feed(
        agency=_frame(agency, "agency"),
        stops=_frame(stops, "stops"),
        routes=_frame(routes, "routes"),
        trips=_frame(trips, "trips"),
        stop_times=_frame(stop_times, "stop_times"),
        calendar_dates=_frame(calendar, "calendar_dates"),
        shapes=_frame(shapes, "shapes") if shapes["shape_id"] else None,
    )


def convert_netex(
    path: str | Path,
    out_dir: str | Path,
    *,
    timezone: str | None = None,
    compression: str = "zstd",
    compression_level: int = 9,
) -> dict[str, Path]:
    """Convert a NeTEx timetable to one Parquet file per GTFS table.

    The output matches :func:`~gtfs_parquet.convert_gtfs_zip`'s, so a NeTEx
    timetable can be stored and queried like any GTFS feed.

    Returns:
        A dict mapping each table name (e.g. ``"stops"``) to its Parquet file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in parse_netex(path, timezone=timezone).tables().items():
        out = out_dir / f"{name}.parquet"
        df.write_parquet(out, compression=compression, compression_level=compression_level)
        written[name] = out
    return written

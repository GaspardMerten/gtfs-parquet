"""Tests for the NeTEx reader."""

from __future__ import annotations

import gzip
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from gtfs_parquet import convert_netex, parse_netex

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<PublicationDelivery xmlns="http://www.netex.org.uk/netex" xmlns:gml="http://www.opengis.net/gml/3.2">
 <dataObjects><CompositeFrame id="C">
  <FrameDefaults><DefaultLocale><TimeZone>Europe/Rome</TimeZone></DefaultLocale></FrameDefaults>
  <frames>
   <ResourceFrame id="R"><organisations>
    <Operator id="OP"><Name>Rail Co</Name><ContactDetails><Url>https://rail.example</Url></ContactDetails></Operator>
   </organisations></ResourceFrame>
   <SiteFrame id="S"><stopPlaces>
    <StopPlace id="SP:A"><Name>Alpha</Name><Centroid><Location><Longitude>9.1</Longitude><Latitude>45.4</Latitude></Location></Centroid>
     <PrivateCode>1001</PrivateCode>
     <quays><Quay id="Q:A"><Name>Alpha</Name><Centroid><Location><Longitude>9.1</Longitude><Latitude>45.4</Latitude></Location></Centroid></Quay></quays>
    </StopPlace>
    <StopPlace id="SP:B"><Name>Beta</Name><Centroid><Location><Longitude>9.2</Longitude><Latitude>45.5</Latitude></Location></Centroid>
     <quays><Quay id="Q:B"><Name>Beta</Name></Quay></quays>
    </StopPlace>
   </stopPlaces></SiteFrame>
   <ServiceFrame id="SF">
    <lines>
     <Line id="L:REG"><Name>Regionale</Name><ShortName>REG</ShortName><TransportMode>rail</TransportMode></Line>
     <Line id="L:BUS"><Name>Autobus</Name><ShortName>BUS</ShortName><TransportMode>bus</TransportMode></Line>
    </lines>
    <scheduledStopPoints>
     <ScheduledStopPoint id="SSP:A"><Name>Alpha</Name></ScheduledStopPoint>
     <ScheduledStopPoint id="SSP:B"><Name>Beta</Name></ScheduledStopPoint>
     <ScheduledStopPoint id="SSP:C"><Name>Gamma</Name><Location><Longitude>9.3</Longitude><Latitude>45.6</Latitude></Location></ScheduledStopPoint>
    </scheduledStopPoints>
    <serviceLinks>
     <ServiceLink id="SL:AB"><FromPointRef ref="SSP:A"/><ToPointRef ref="SSP:B"/>
      <projections><LinkSequenceProjection id="P1"><gml:LineString gml:id="G1"><gml:posList>45.4 9.1 45.45 9.15 45.5 9.2</gml:posList></gml:LineString></LinkSequenceProjection></projections>
     </ServiceLink>
     <ServiceLink id="SL:BC"><FromPointRef ref="SSP:B"/><ToPointRef ref="SSP:C"/>
      <projections><LinkSequenceProjection id="P2"><gml:LineString gml:id="G2"><gml:posList>45.5 9.2 45.6 9.3</gml:posList></gml:LineString></LinkSequenceProjection></projections>
     </ServiceLink>
    </serviceLinks>
    <stopAssignments>
     <PassengerStopAssignment id="PSA:A" order="1"><ScheduledStopPointRef ref="SSP:A"/><StopPlaceRef ref="SP:A"/><QuayRef ref="Q:A"/></PassengerStopAssignment>
     <PassengerStopAssignment id="PSA:B" order="1"><ScheduledStopPointRef ref="SSP:B"/><StopPlaceRef ref="SP:B"/><QuayRef ref="Q:B"/></PassengerStopAssignment>
    </stopAssignments>
    <journeyPatterns>
     <ServiceJourneyPattern id="JP:1"><RouteView><LineRef ref="L:REG"/></RouteView>
      <pointsInSequence>
       <StopPointInJourneyPattern id="P:1:1" order="1"><ScheduledStopPointRef ref="SSP:A"/><OnwardServiceLinkRef ref="SL:AB"/><ForAlighting>false</ForAlighting></StopPointInJourneyPattern>
       <StopPointInJourneyPattern id="P:1:2" order="2"><ScheduledStopPointRef ref="SSP:B"/><OnwardServiceLinkRef ref="SL:BC"/></StopPointInJourneyPattern>
       <StopPointInJourneyPattern id="P:1:3" order="3"><ScheduledStopPointRef ref="SSP:C"/><ForBoarding>false</ForBoarding></StopPointInJourneyPattern>
      </pointsInSequence>
     </ServiceJourneyPattern>
     <ServiceJourneyPattern id="JP:2"><RouteView><LineRef ref="L:BUS"/></RouteView>
      <pointsInSequence>
       <StopPointInJourneyPattern id="P:2:1" order="1"><ScheduledStopPointRef ref="SSP:B"/></StopPointInJourneyPattern>
       <StopPointInJourneyPattern id="P:2:2" order="2"><ScheduledStopPointRef ref="SSP:A"/></StopPointInJourneyPattern>
      </pointsInSequence>
     </ServiceJourneyPattern>
    </journeyPatterns>
   </ServiceFrame>
   <TimetableFrame id="TF"><vehicleJourneys>
    <ServiceJourney id="SJ:1"><Name>10201</Name><dayTypes><DayTypeRef ref="DT:1"/></dayTypes>
     <ServiceJourneyPatternRef ref="JP:1"/><OperatorRef ref="OP"/>
     <passingTimes>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:1"/><DepartureTime>23:50:00</DepartureTime></TimetabledPassingTime>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:2"/><ArrivalTime>23:58:00</ArrivalTime><DepartureTime>00:01:00</DepartureTime><DepartureDayOffset>1</DepartureDayOffset></TimetabledPassingTime>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:3"/><ArrivalTime>00:20:00</ArrivalTime><ArrivalDayOffset>1</ArrivalDayOffset></TimetabledPassingTime>
     </passingTimes>
    </ServiceJourney>
    <ServiceJourney id="SJ:2"><Name>10203</Name><dayTypes><DayTypeRef ref="DT:1"/></dayTypes>
     <ServiceJourneyPatternRef ref="JP:1"/>
     <passingTimes>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:1"/><DepartureTime>08:00:00</DepartureTime></TimetabledPassingTime>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:2"/><ArrivalTime>08:10:00</ArrivalTime><DepartureTime>08:11:00</DepartureTime></TimetabledPassingTime>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:1:3"/><ArrivalTime>08:30:00</ArrivalTime></TimetabledPassingTime>
     </passingTimes>
    </ServiceJourney>
    <ServiceJourney id="SJ:3"><Name>B1</Name><dayTypes><DayTypeRef ref="DT:2"/></dayTypes>
     <ServiceJourneyPatternRef ref="JP:2"/>
     <passingTimes>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:2:1"/><DepartureTime>09:00:00</DepartureTime></TimetabledPassingTime>
      <TimetabledPassingTime><StopPointInJourneyPatternRef ref="P:2:2"/><ArrivalTime>09:20:00</ArrivalTime></TimetabledPassingTime>
     </passingTimes>
    </ServiceJourney>
   </vehicleJourneys></TimetableFrame>
   <ServiceCalendarFrame id="CF">
    <dayTypes>
     <DayType id="DT:1"><properties><PropertyOfDay><DaysOfWeek>Sunday Monday Tuesday Wednesday Thursday Friday Saturday</DaysOfWeek></PropertyOfDay></properties></DayType>
     <DayType id="DT:2"><properties><PropertyOfDay><DaysOfWeek>Weekdays</DaysOfWeek></PropertyOfDay></properties></DayType>
    </dayTypes>
    <operatingPeriods>
     <UicOperatingPeriod id="OP:1"><FromDate>2026-10-05T00:00:00</FromDate><ToDate>2026-10-11T00:00:00</ToDate><ValidDayBits>1010100</ValidDayBits></UicOperatingPeriod>
     <OperatingPeriod id="OP:2"><FromDate>2026-10-05T00:00:00</FromDate><ToDate>2026-10-11T00:00:00</ToDate></OperatingPeriod>
    </operatingPeriods>
    <dayTypeAssignments>
     <DayTypeAssignment id="DTA:1" order="1"><OperatingPeriodRef ref="OP:1"/><DayTypeRef ref="DT:1"/></DayTypeAssignment>
     <DayTypeAssignment id="DTA:2" order="1"><OperatingPeriodRef ref="OP:2"/><DayTypeRef ref="DT:2"/></DayTypeAssignment>
     <DayTypeAssignment id="DTA:3" order="2"><Date>2026-10-07</Date><DayTypeRef ref="DT:2"/><isAvailable>false</isAvailable></DayTypeAssignment>
    </dayTypeAssignments>
   </ServiceCalendarFrame>
  </frames>
 </CompositeFrame></dataObjects>
</PublicationDelivery>
"""


def _sample(tmp_path: Path, kind: str = "xml") -> Path:
    if kind == "gz":
        path = tmp_path / "netex.xml.gz"
        path.write_bytes(gzip.compress(SAMPLE.encode()))
    elif kind == "zip":
        path = tmp_path / "netex.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("line.xml", SAMPLE)
    else:
        path = tmp_path / "netex.xml"
        path.write_text(SAMPLE)
    return path


def test_tables(tmp_path):
    feed = parse_netex(_sample(tmp_path))
    assert feed.agency.row(0, named=True)["agency_timezone"] == "Europe/Rome"
    assert dict(feed.routes.select("route_short_name", "route_type").iter_rows()) == {"REG": 2, "BUS": 3}

    stops = feed.stops.sort("stop_id")
    assert stops["stop_id"].to_list() == ["Q:A", "Q:B", "SP:A", "SP:B", "SSP:C"]
    # A quay without coordinates takes its station's
    assert stops.filter(pl.col("stop_id") == "Q:B")["stop_lat"].item() == pl.Series([45.5], dtype=pl.Float32).item()
    assert stops.filter(pl.col("stop_id") == "Q:A")["parent_station"].item() == "SP:A"

    trips = feed.trips.sort("trip_id")
    assert trips["trip_short_name"].to_list() == ["10201", "10203", "B1"]
    assert trips["trip_headsign"].to_list() == ["Gamma", "Gamma", "Alpha"]
    assert trips["shape_id"].to_list() == ["JP:1", "JP:1", None]


def test_stop_times(tmp_path):
    st = parse_netex(_sample(tmp_path)).stop_times.filter(pl.col("trip_id") == "SJ:1")
    hours = lambda c: [d.total_seconds() / 3600 for d in st[c]]
    assert st["stop_id"].to_list() == ["Q:A", "Q:B", "SSP:C"]
    assert hours("arrival_time") == [23 + 50 / 60, 23 + 58 / 60, 24 + 20 / 60]
    assert hours("departure_time") == [23 + 50 / 60, 24 + 1 / 60, 24 + 20 / 60]
    assert st["drop_off_type"].to_list() == [1, 0, 0]
    assert st["pickup_type"].to_list() == [0, 0, 1]
    assert st.schema["arrival_time"] == pl.Duration("ms")


def test_calendar(tmp_path):
    feed = parse_netex(_sample(tmp_path))
    days = {
        sid: sorted(g["date"].to_list())
        for (sid,), g in feed.calendar_dates.group_by("service_id")
    }
    start = date(2026, 10, 5)
    assert days["DT:1"] == [start, start + timedelta(2), start + timedelta(4)]
    # Weekdays of the period, minus the day marked unavailable
    assert days["DT:2"] == [start, start + timedelta(1), start + timedelta(3), start + timedelta(4)]


def test_shapes(tmp_path):
    shapes = parse_netex(_sample(tmp_path)).shapes.sort("shape_pt_sequence")
    # Links joined in pattern order, the shared point once, latitude first in GML
    assert [round(v, 2) for v in shapes["shape_pt_lat"]] == [45.4, 45.45, 45.5, 45.6]
    assert [round(v, 2) for v in shapes["shape_pt_lon"]] == [9.1, 9.15, 9.2, 9.3]


def test_formats_and_convert(tmp_path):
    for kind in ("gz", "zip"):
        assert parse_netex(_sample(tmp_path, kind)).trips.height == 3
    out = convert_netex(_sample(tmp_path), tmp_path / "out", timezone="Europe/Brussels")
    assert set(out) == {"agency", "stops", "routes", "trips", "stop_times", "calendar_dates", "shapes"}
    assert pl.read_parquet(out["agency"])["agency_timezone"].item() == "Europe/Brussels"

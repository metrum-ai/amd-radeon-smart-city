# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""All Pydantic v2 request/response models for the FastAPI application."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated list wrapper."""

    model_config = ConfigDict(from_attributes=True)

    items: List[T]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


class ZoneCreate(BaseModel):
    """Zone definition for stream creation."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    name: str
    zone_type: str
    detect_class: str = "person"
    polygon: List[List[float]]
    threshold_critical: int = 200


class StreamCreate(BaseModel):
    """Request to add a new RTSP stream."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    location_name: str
    lat: float
    lon: float
    codec: str = "h264"
    width: int = 1280
    height: int = 720
    gpu_id: int = -1
    zones: List[ZoneCreate] = Field(default_factory=list)


class StreamResponse(BaseModel):
    """Current state of a registered stream."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    location_name: str
    city: Optional[str] = None
    lat: float
    lon: float
    status: str
    gpu_id: int
    current_severity: Literal["SAFE", "CRITICAL"]
    person_count: int
    fps: float
    latency_ms: float
    last_updated: datetime


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class AlertResponse(BaseModel):
    """A crowd or zone violation alert."""

    model_config = ConfigDict(from_attributes=True)

    alert_id: str
    timestamp: datetime
    stream_id: int
    zone_id: str
    zone_name: str
    zone_type: str
    location_name: str
    lat: float
    lon: float
    person_count: int
    threshold: int
    severity: Literal["SAFE", "CRITICAL"]
    violation_type: Optional[Literal["crowd_density"]] = None
    description: str
    is_auto_popup: bool
    status: str
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""

    model_config = ConfigDict(from_attributes=True)

    reason: Optional[str] = None
    operator_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class CameraMarker(BaseModel):
    """Camera position and live severity for the city map."""

    model_config = ConfigDict(from_attributes=True)

    stream_id: int
    location_name: str
    lat: float
    lon: float
    severity: Literal["SAFE", "CRITICAL"]
    zone_type: str
    violation_type: Optional[str] = None
    person_count: int
    threshold: int
    is_auto_popup: bool
    stream_status: str
    location_tag: Optional[str] = None
    preview_video_url: Optional[str] = None
    webrtc_whep_url: Optional[str] = None
    marker_color: Optional[str] = None
    marker_radius: Optional[int] = None
    marker_pulse: Optional[bool] = None
    tooltip_width: Optional[int] = None
    tooltip_aspect_ratio: Optional[str] = None


class MapDataResponse(BaseModel):
    """All camera markers with aggregate counters."""

    model_config = ConfigDict(from_attributes=True)

    cameras: List[CameraMarker]
    timestamp: datetime
    active_critical_count: int


class HotspotMapCity(BaseModel):
    """Map viewport option configured for the city selector."""

    model_config = ConfigDict(from_attributes=True)

    label: str
    center: List[float]
    zoom: int


class HotspotSubLocation(BaseModel):
    """Sub-location option fixed under a configurable main location."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    label: str


class HotspotMainLocation(BaseModel):
    """Main location configuration with fixed sub-locations."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    label: str
    center: List[float]
    zoom: int
    sub_locations: List[HotspotSubLocation]


class HotspotSeverityStyle(BaseModel):
    """Visual style for hotspot markers by severity."""

    model_config = ConfigDict(from_attributes=True)

    color: str
    radius: int
    pulse: bool


class HotspotTooltipConfig(BaseModel):
    """Tooltip video card layout settings."""

    model_config = ConfigDict(from_attributes=True)

    width: int
    aspect_ratio: str


class HotspotConfigResponse(BaseModel):
    """Config driving city views and hotspot visual behaviour."""

    model_config = ConfigDict(from_attributes=True)

    cities: List[HotspotMapCity]
    main_locations: List[HotspotMainLocation]
    styles: dict[str, HotspotSeverityStyle]
    tooltip: HotspotTooltipConfig


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class ZoneDensity(BaseModel):
    """Per-zone density snapshot."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    zone_name: str
    zone_type: str
    person_count: int
    density_score: float
    severity: Literal["SAFE", "CRITICAL"]
    violation_type: Optional[str] = None


class DensityResponse(BaseModel):
    """Density snapshot for all zones on a stream."""

    model_config = ConfigDict(from_attributes=True)

    stream_id: int
    zones: List[ZoneDensity]
    timestamp: datetime


class HeatmapResponse(BaseModel):
    """Base64-encoded JPEG heatmap for a stream."""

    model_config = ConfigDict(from_attributes=True)

    stream_id: int
    image_b64: str
    width: int
    height: int
    timestamp: datetime


class PatternItem(BaseModel):
    """A single recurring crowd pattern."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    zone_name: str
    zone_type: str
    violation_type: str
    pattern_description: str
    frequency: str
    peak_day: str
    avg_count: float
    occurrences: int


class PatternResponse(BaseModel):
    """List of recurring patterns."""

    model_config = ConfigDict(from_attributes=True)

    patterns: List[PatternItem]


class KPIResponse(BaseModel):
    """Live dashboard KPI values."""

    model_config = ConfigDict(from_attributes=True)

    active_critical_zones: int
    peak_crowd_count: int
    peak_crowd_location: str
    streams_live: int
    streams_total: int
    detection_latency_p95: float
    timestamp: datetime


class WeeklyHeatmapResponse(BaseModel):
    """7×24 average crowd density grid (Mon=0 … Sun=6, hour 0–23).

    Values are 0–100 representing average density percentage.
    """

    model_config = ConfigDict(from_attributes=True)

    grid: List[List[float]]
    generated_at: datetime


class TrendZone(BaseModel):
    """Per-zone time-series trend data."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    zone_name: str
    location_name: str
    data: List[int]


class TrendResponse(BaseModel):
    """Daily peak crowd counts per zone for a selected time range."""

    model_config = ConfigDict(from_attributes=True)

    zones: List[TrendZone]
    labels: List[str]
    range: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class ReportRequest(BaseModel):
    """Incident report generation request.

    Accepts either an explicit ``hours`` lookback window or an ISO-date
    ``from_date`` / ``to_date`` range.  When both dates are supplied the
    ``hours`` field is ignored; the effective window is computed from the
    date range.

    ``location_name`` is optional — if omitted it is resolved automatically
    from ``streams_metadata.yaml`` using ``zone_id``.
    """

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    hours: int = 24
    location_name: Optional[str] = Field(
        default=None,
        description="Human-readable location name (auto-resolved if omitted).",
    )
    from_date: Optional[str] = Field(
        default=None,
        description="Start of report window (ISO date, e.g. '2026-01-01').",
    )
    to_date: Optional[str] = Field(
        default=None,
        description="End of report window (ISO date, e.g. '2026-02-01').",
    )


class ReportResponse(BaseModel):
    """Generated report response."""

    model_config = ConfigDict(from_attributes=True)

    report_id: str
    content: str
    zone_id: str
    generated_at: datetime


class AgentTraceStep(BaseModel):
    """Single trace entry emitted by one report agent."""

    model_config = ConfigDict(from_attributes=True)

    agent: str
    tool: Optional[str] = None
    summary: str
    timestamp: datetime
    input_preview: Optional[str] = Field(
        default=None,
        description="First 200 chars of the input handed to this agent/tool.",
    )


class ReportJobResponse(BaseModel):
    """Async report job status response."""

    model_config = ConfigDict(from_attributes=True)

    report_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    zone_id: str
    location_name: Optional[str] = Field(
        default=None,
        description="Human-readable location name resolved from streams_metadata.",
    )
    city: Optional[str] = Field(default=None, description="City / district label.")
    lat: Optional[float] = Field(default=None, description="Latitude of the zone.")
    lon: Optional[float] = Field(default=None, description="Longitude of the zone.")
    submitted_at: datetime
    generated_at: Optional[datetime] = None
    error: Optional[str] = None
    content: Optional[str] = None
    agent_trace: Optional[List[AgentTraceStep]] = None


# ---------------------------------------------------------------------------

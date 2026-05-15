# Created by Metrum AI for AMD

"""Dashboard map data routes."""
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

from smart_city.api.models import (
    CameraMarker,
    HotspotConfigResponse,
    HotspotMainLocation,
    HotspotMapCity,
    HotspotSeverityStyle,
    HotspotSubLocation,
    HotspotTooltipConfig,
    MapDataResponse,
)

router = APIRouter(tags=["dashboard"])


@lru_cache(maxsize=1)
def _load_hotspot_config() -> dict[str, Any]:
    """Load hotspot and replay mapping config from config/hotspots.yaml."""
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "hotspots.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _location_tag(location_name: str) -> str:
    """Map location names into three hotspot regions."""
    cfg = _load_hotspot_config()
    tags = cfg.get("location_tags", {})
    name = location_name.lower()
    for tag, item in tags.items():
        keywords = item.get("keywords", [])
        if any(keyword.lower() in name for keyword in keywords):
            return str(tag)
    return "west"


def _preview_video_url(
    stream_id: int, location_name: str, base_url: str
) -> str:
    """Return a stable replay clip URL for each stream/location."""
    cfg = _load_hotspot_config()
    tag = _location_tag(location_name)
    tags = cfg.get("location_tags", {})
    pool = tags.get(tag, {}).get("video_pool", [])
    if not pool:
        pool = tags.get("west", {}).get("video_pool", ["crowd_1.mp4"])
    filename = pool[(stream_id - 1) % len(pool)]
    return f"{base_url.rstrip('/')}/videos/{filename}"


def _webrtc_whep_url(stream_id: int, suffix: str = "") -> str:
    """Build WHEP URL for the currently available MediaMTX stream path."""
    base = os.getenv("SCRATCH_WEBRTC_BASE_URL", "/webrtc")
    return f"{base.rstrip('/')}/cam{stream_id}{suffix}/whep"


def _preferred_stream_suffix(data: dict[str, Any]) -> str:
    """Return the active preview suffix.

    Density-enabled streams use the paired output (YOLO+Density side-by-side).
    This reduces WebRTC connections from 2 to 1 per camera, preventing
    MediaMTX backpressure issues with many concurrent viewers.
    """
    if data.get("paired_published", False):
        return "_paired"
    return "_yolo"


def _style_for(severity: str) -> dict[str, Any]:
    """Return marker display style for SAFE/CRITICAL."""
    cfg = _load_hotspot_config()
    styles = cfg.get("hotspot_style", {})
    key = "critical" if severity == "CRITICAL" else "safe"
    return styles.get(key, {"color": "#22C55E", "radius": 6, "pulse": False})


def _tooltip_cfg() -> dict[str, Any]:
    """Return tooltip layout config."""
    cfg = _load_hotspot_config()
    return cfg.get("tooltip", {"width": 220, "aspect_ratio": "16/9"})


@lru_cache(maxsize=1)
def _load_stream_city_index() -> tuple[dict[str, str], dict[str, str]]:
    """Load zone->city and location->city mappings from stream metadata."""
    from smart_city.core.stream_allocation import (  # noqa: PLC0415
        apply_city_allocation,
    )

    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "streams_metadata.yaml"
    )
    zone_to_city: dict[str, str] = {}
    location_to_city: dict[str, str] = {}
    if not cfg_path.exists():
        return zone_to_city, location_to_city
    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    for stream in apply_city_allocation(list(loaded.get("streams", []))):
        city = str(stream.get("city", "")).strip()
        zone_id = str(stream.get("zone_id", "")).strip()
        if not zone_id:
            stream_id = stream.get("id")
            if stream_id is not None:
                zone_id = f"cam{stream_id}_main"
        location_name = str(stream.get("location_name", "")).strip()
        if city and zone_id:
            zone_to_city[zone_id] = city
        if city and location_name:
            location_to_city[location_name] = city
    return zone_to_city, location_to_city


def _camera_matches_city(data: dict, city: str) -> bool:
    """Return True when a live camera cache entry belongs to city."""
    zone_to_city, location_to_city = _load_stream_city_index()
    for zone in data.get("zones", []):
        zone_id = str(zone.get("zone_id", "")).strip()
        if zone_id and zone_to_city.get(zone_id) == city:
            return True
    location_name = str(data.get("location_name", "")).strip()
    if location_to_city.get(location_name) == city:
        return True
    # Fallback for synthesised streams (beyond YAML pool size): city is
    # stored directly on the density_cache entry by _resolve_stream_cfg.
    return str(data.get("city", "")).strip() == city


@router.get("/dashboard/map", response_model=MapDataResponse)
async def get_map_data(
    request: Request,
    city: Optional[str] = None,
) -> MapDataResponse:
    """Return all camera markers with live severity for the city map."""
    cache = getattr(request.app.state, "density_cache", {})
    base_url = str(request.base_url).rstrip("/")
    markers = []
    cache_items = list(cache.items())
    if city:
        cache_items = [
            (stream_id, data)
            for stream_id, data in cache_items
            if _camera_matches_city(data, city)
        ]
    for stream_id, data in cache_items:
        style = _style_for(data.get("severity", "SAFE"))
        tooltip = _tooltip_cfg()
        markers.append(CameraMarker(
            stream_id=stream_id,
            location_name=data.get("location_name", ""),
            lat=data.get("lat", 0.0),
            lon=data.get("lon", 0.0),
            severity=data.get("severity", "SAFE"),
            zone_type=data.get("zone_type", "crowd_density"),
            violation_type=data.get("violation_type"),
            person_count=data.get("person_count", 0),
            threshold=data.get("threshold", 200),
            is_auto_popup=data.get("severity") == "CRITICAL",
            stream_status=data.get("status", "LIVE"),
            location_tag=_location_tag(data.get("location_name", "")),
            preview_video_url=_preview_video_url(
                stream_id=stream_id,
                location_name=data.get("location_name", ""),
                base_url=base_url,
            ),
            webrtc_whep_url=_webrtc_whep_url(
                stream_id,
                _preferred_stream_suffix(data),
            ),
            marker_color=style.get("color"),
            marker_radius=int(style.get("radius", 6)),
            marker_pulse=bool(style.get("pulse", False)),
            tooltip_width=int(tooltip.get("width", 220)),
            tooltip_aspect_ratio=str(tooltip.get("aspect_ratio", "16/9")),
        ))
    critical = [m for m in markers if m.severity == "CRITICAL"]
    return MapDataResponse(
        cameras=markers,
        timestamp=datetime.now(tz=timezone.utc),
        active_critical_count=len(critical),
    )


@router.get("/dashboard/auto-popup")
async def get_auto_popup(
    request: Request,
    city: Optional[str] = None,
):
    """Return only streams currently triggering auto-popup."""
    cache = getattr(request.app.state, "density_cache", {})
    base_url = str(request.base_url).rstrip("/")
    cache_items = list(cache.items())
    if city:
        cache_items = [
            (sid, d)
            for sid, d in cache_items
            if _camera_matches_city(d, city)
        ]
    return [
        CameraMarker(
            stream_id=sid,
            location_name=d.get("location_name", ""),
            lat=d.get("lat", 0.0),
            lon=d.get("lon", 0.0),
            severity="CRITICAL",
            zone_type=d.get("zone_type", "crowd_density"),
            violation_type=d.get("violation_type"),
            person_count=d.get("person_count", 0),
            threshold=d.get("threshold", 200),
            is_auto_popup=True,
            stream_status="LIVE",
            location_tag=_location_tag(d.get("location_name", "")),
            preview_video_url=_preview_video_url(
                stream_id=sid,
                location_name=d.get("location_name", ""),
                base_url=base_url,
            ),
            webrtc_whep_url=_webrtc_whep_url(
                sid,
                _preferred_stream_suffix(d),
            ),
            marker_color=_style_for("CRITICAL").get("color"),
            marker_radius=int(_style_for("CRITICAL").get("radius", 10)),
            marker_pulse=bool(_style_for("CRITICAL").get("pulse", True)),
            tooltip_width=int(_tooltip_cfg().get("width", 220)),
            tooltip_aspect_ratio=str(
                _tooltip_cfg().get("aspect_ratio", "16/9")
            ),
        )
        for sid, d in cache_items
        if d.get("severity") == "CRITICAL"
    ]


@router.get("/dashboard/hotspot-config", response_model=HotspotConfigResponse)
async def get_hotspot_config() -> HotspotConfigResponse:
    """Return configured map coordinates and hotspot display options."""
    cfg = _load_hotspot_config()
    cities = [
        HotspotMapCity(
            label=str(item.get("label", "Austin Downtown, TX")),
            center=list(item.get("center", [30.2672, -97.7431])),
            zoom=int(item.get("zoom", 13)),
        )
        for item in cfg.get("cities", [])
    ]
    if not cities:
        cities = [
            HotspotMapCity(
                label="Austin Downtown, TX",
                center=[30.2672, -97.7431],
                zoom=13,
            )
        ]
    main_locations = [
        HotspotMainLocation(
            id=str(item.get("id", "austin_downtown")),
            label=str(item.get("label", "Austin Downtown, TX")),
            center=list(item.get("center", [30.2672, -97.7431])),
            zoom=int(item.get("zoom", 13)),
            sub_locations=[
                HotspotSubLocation(
                    zone_id=str(sub.get("zone_id", "")),
                    label=str(sub.get("label", sub.get("zone_id", ""))),
                )
                for sub in item.get("sub_locations", [])
                if str(sub.get("zone_id", "")).strip()
            ],
        )
        for item in cfg.get("main_locations", [])
    ]
    if not main_locations:
        main_locations = [
            HotspotMainLocation(
                id="austin_downtown",
                label="Austin Downtown, TX",
                center=[30.2672, -97.7431],
                zoom=13,
                sub_locations=[],
            )
        ]
    styles = {
        key: HotspotSeverityStyle(
            color=str(val.get("color", "#22C55E")),
            radius=int(val.get("radius", 6)),
            pulse=bool(val.get("pulse", False)),
        )
        for key, val in cfg.get("hotspot_style", {}).items()
    }
    if "critical" not in styles:
        styles["critical"] = HotspotSeverityStyle(
            color="#ED1C24", radius=10, pulse=True
        )
    if "safe" not in styles:
        styles["safe"] = HotspotSeverityStyle(
            color="#22C55E", radius=6, pulse=False
        )
    tooltip_raw = cfg.get("tooltip", {})
    tooltip = HotspotTooltipConfig(
        width=int(tooltip_raw.get("width", 220)),
        aspect_ratio=str(tooltip_raw.get("aspect_ratio", "16/9")),
    )
    return HotspotConfigResponse(
        cities=cities,
        main_locations=main_locations,
        styles=styles,
        tooltip=tooltip,
    )


_STREAMS_META_FILE = os.environ.get(
    "STREAMS_METADATA_FILE", "/app/smart_city/config/streams_metadata.yaml"
)

_CITY_ID_MAP: dict[str, str] = {
    "Austin Downtown, TX": "austin_downtown",
    "Dallas Downtown, TX": "dallas_downtown",
    "Houston Downtown, TX": "houston_downtown",
}


@lru_cache(maxsize=1)
def _load_streams_metadata() -> dict[str, list[dict]]:
    """Return streams grouped by city_id, keyed by city label → id mapping.

    Returns:
        Dict mapping city_id (e.g. ``austin_downtown``) to an ordered list
        of ``{zone_id, label}`` dicts loaded from streams_metadata.yaml.
    """
    from smart_city.core.stream_allocation import (  # noqa: PLC0415
        apply_city_allocation,
    )

    try:
        with open(_STREAMS_META_FILE, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        streams = raw.get("streams", raw) if isinstance(raw, dict) else raw
        if not isinstance(streams, list):
            return {}
        streams = apply_city_allocation(list(streams))
        grouped: dict[str, list[dict]] = {}
        seen: set[str] = set()
        for s in streams:
            city_label = s.get("city", "")
            city_id = _CITY_ID_MAP.get(city_label, "")
            if not city_id:
                continue
            zone_id = s.get("zone_id") or (
                f"cam{s['id']}_main" if s.get("id") else None
            )
            if not zone_id or zone_id in seen:
                continue
            seen.add(zone_id)
            grouped.setdefault(city_id, []).append(
                {
                    "zone_id": zone_id,
                    "label": s.get("location_name", zone_id),
                }
            )
        return grouped
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        logger.warning(
            "Could not load streams metadata from %s: %s",
            _STREAMS_META_FILE, exc, exc_info=True,
        )
        return {}


@router.get("/dashboard/locations")
async def get_locations_by_city(
    request: Request,
    city_id: Optional[str] = None,
) -> dict:
    """Return stream locations grouped by city_id from streams_metadata.yaml.

    Also includes synthesised streams (beyond the YAML pool size) from the
    live density_cache so the location dropdown stays complete when
    STREAM_COUNT exceeds the number of YAML entries.

    Args:
        city_id: Optional filter, e.g. ``austin_downtown``.  Returns all
            cities when omitted.

    Returns:
        Dict mapping city_id to list of ``{zone_id, label}`` objects.
    """
    # Deep-copy so we can append without corrupting the lru_cache result.
    base = _load_streams_metadata()
    data: dict[str, list[dict]] = {k: list(v) for k, v in base.items()}

    # Augment with synthesised streams from the live cache.
    cache: dict = getattr(request.app.state, "density_cache", {})
    for sid, stream_data in sorted(cache.items()):
        city_label = str(stream_data.get("city", "")).strip()
        cid = _CITY_ID_MAP.get(city_label, "")
        if not cid:
            continue
        zone_id = str(stream_data.get("zone_id", "")).strip() or f"cam{sid}_main"
        # Skip if already present from YAML.
        if any(e["zone_id"] == zone_id for e in data.get(cid, [])):
            continue
        data.setdefault(cid, []).append({
            "zone_id": zone_id,
            "label": str(stream_data.get("location_name", zone_id)).strip(),
        })

    if city_id:
        return {city_id: data.get(city_id, [])}
    return data

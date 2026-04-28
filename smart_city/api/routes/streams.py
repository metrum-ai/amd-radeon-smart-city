# Created by Metrum AI for AMD

"""Stream management routes."""

import logging
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request, status

from smart_city.api.models import StreamCreate, StreamResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["streams"])


def _cache_to_response(sid: int, data: dict) -> StreamResponse:
    # Extract short city name (e.g. "Austin" from "Austin Downtown, TX")
    full_city: str = data.get("city", "") or ""
    short_city = full_city.split()[0] if full_city else None
    return StreamResponse(
        id=sid,
        location_name=data.get("location_name", ""),
        city=short_city,
        lat=data.get("lat", 0.0),
        lon=data.get("lon", 0.0),
        status=data.get("status", "LIVE"),
        gpu_id=0,
        current_severity=data.get("severity", "SAFE"),
        person_count=data.get("person_count", 0),
        fps=data.get("fps", 0.0),
        latency_ms=data.get("latency_ms", 0.0),
        last_updated=datetime.now(tz=timezone.utc),
    )


@router.get("/streams", response_model=List[StreamResponse])
async def list_streams(request: Request) -> List[StreamResponse]:
    """Return current status of all registered streams."""
    cache: dict = getattr(request.app.state, "density_cache", {})
    return [_cache_to_response(sid, data) for sid, data in sorted(cache.items())]


@router.get("/streams/{stream_id}", response_model=StreamResponse)
async def get_stream(stream_id: int, request: Request) -> StreamResponse:
    """Return current status of a single stream by ID."""
    cache: dict = getattr(request.app.state, "density_cache", {})
    data = cache.get(stream_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    return _cache_to_response(stream_id, data)


@router.post("/streams", response_model=StreamResponse,
             status_code=status.HTTP_201_CREATED)
async def add_stream(body: StreamCreate, request: Request) -> StreamResponse:
    """Add a new RTSP camera stream (admin only)."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    try:
        pipeline.add_stream(body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return StreamResponse(
        id=body.id,
        location_name=body.location_name,
        lat=body.lat,
        lon=body.lon,
        status="STARTING",
        gpu_id=body.gpu_id,
        current_severity="SAFE",
        person_count=0,
        fps=0.0,
        latency_ms=0.0,
        last_updated=datetime.now(tz=timezone.utc),
    )


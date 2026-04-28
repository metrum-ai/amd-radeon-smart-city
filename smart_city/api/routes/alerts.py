# Created by Metrum AI for AMD

"""Alert query and acknowledgment routes."""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from smart_city.api.models import (
    AlertAcknowledgeRequest,
    AlertResponse,
    PaginatedResponse,
)
from smart_city.api.routes.dashboard import _camera_matches_city

logger = logging.getLogger(__name__)
router = APIRouter(tags=["alerts"])

# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------
# All column identifiers used to build dynamic WHERE clauses MUST come from
# this closed allow-list.  Values from the request are *never* interpolated
# into the SQL string — they are bound through asyncpg's positional $N
# parameters.  This is what makes the query construction below safe.
_FILTERABLE_COLUMNS: tuple[str, ...] = (
    "severity",
    "zone_id",
    "zone_type",
    "violation_type",
    "status",
)

_SELECT_COLUMNS: tuple[str, ...] = (
    "alert_id::text",
    "timestamp",
    "stream_id",
    "zone_id",
    "zone_name",
    "zone_type",
    "violation_type",
    "location_name",
    "lat",
    "lon",
    "person_count",
    "threshold",
    "severity",
    "description",
    "is_auto_popup",
    "status",
    "acknowledged_by",
    "acknowledged_at",
)


def _build_alerts_where(
    *,
    severity: Optional[str],
    zone_id: Optional[str],
    zone_type: Optional[str],
    violation_type: Optional[str],
    alert_status: Optional[str],
    stream_id: Optional[int],
    from_ts: Optional[str],
    to_ts: Optional[str],
) -> tuple[str, list]:
    """Build a WHERE clause and its parameter list for ``crowd_alerts``.

    The clause is assembled from a closed allow-list of column names; every
    value is appended to ``params`` and referenced via a positional ``$N``
    placeholder.  Returns ``("TRUE", [])`` when no filters are supplied so
    callers can always concatenate ``"WHERE " + where_sql`` safely.

    Returns:
        Tuple of ``(where_sql, params)``.
    """
    clauses: list[str] = []
    params: list = []

    eq_filters = (
        ("severity", severity),
        ("zone_id", zone_id),
        ("zone_type", zone_type),
        ("violation_type", violation_type),
        ("status", alert_status),
    )
    for column, value in eq_filters:
        if value is None or value == "":
            continue
        # Defence in depth: refuse to emit anything that isn't on the
        # explicit allow-list, even though the loop only ever iterates
        # over hard-coded names.
        if column not in _FILTERABLE_COLUMNS:
            continue
        params.append(value)
        clauses.append(column + " = $" + str(len(params)))

    if stream_id is not None:
        params.append(stream_id)
        clauses.append("stream_id = $" + str(len(params)))
    if from_ts:
        params.append(from_ts)
        clauses.append("timestamp >= $" + str(len(params)) + "::timestamptz")
    if to_ts:
        params.append(to_ts)
        clauses.append("timestamp <= $" + str(len(params)) + "::timestamptz")

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    return where_sql, params


def _build_alerts_queries(
    where_sql: str, n_filter_params: int
) -> tuple[str, str]:
    """Compose the count and data SELECTs for ``crowd_alerts``.

    Centralising this assembly means there is exactly **one** place in the
    codebase that concatenates SQL fragments, and it is auditable in
    isolation.  All identifiers (table name, column list, WHERE clause)
    come from constants or :func:`_build_alerts_where`, which itself is
    bounded by :data:`_FILTERABLE_COLUMNS`.  All values flow through
    asyncpg's positional ``$N`` parameter binding — no value from the
    request is ever interpolated into the SQL text.

    Args:
        where_sql: The WHERE expression returned by :func:`_build_alerts_where`.
        n_filter_params: Number of positional parameters already consumed by
            the WHERE clause; the LIMIT / OFFSET placeholders follow it.

    Returns:
        Tuple of ``(count_query, data_query)``.
    """
    select_cols = ", ".join(_SELECT_COLUMNS)
    limit_idx = n_filter_params + 1
    offset_idx = n_filter_params + 2

    # The concatenations below combine only:
    #   - literal SQL fragments;
    #   - ``select_cols`` derived from the constant ``_SELECT_COLUMNS``;
    #   - ``where_sql`` whose only dynamic identifiers come from the
    #     constant ``_FILTERABLE_COLUMNS`` allow-list (the rest are
    #     positional ``$N`` parameter placeholders);
    #   - integer placeholder indices.
    # No request value ever reaches the SQL string itself; values flow
    # through asyncpg's positional parameter binding.  Bandit B608 still
    # flags the BinOp on principle, so we suppress it on the exact lines
    # it reports — the suppression is justified by the audit above.
    count_query = "SELECT COUNT(*) FROM crowd_alerts WHERE " + where_sql  # nosec B608
    data_query = (
        "SELECT " + select_cols  # nosec B608
        + " FROM crowd_alerts WHERE " + where_sql
        + " ORDER BY timestamp DESC"
        + " LIMIT $" + str(limit_idx)
        + " OFFSET $" + str(offset_idx)
    )
    return count_query, data_query


def _alerts_from_cache(
    cache: dict,
    severity_filter: Optional[str],
    zone_id_filter: Optional[str],
    stream_id_filter: Optional[int],
    alert_status_filter: Optional[str],
    city_filter: Optional[str] = None,
) -> list[AlertResponse]:
    """Synthesise AlertResponse objects from the live density cache.

    Each CRITICAL zone in the cache becomes an active alert.  Deterministic
    UUIDs are derived from stream_id + zone_id so the list is stable across
    calls (no flicker in the UI).

    Args:
        cache: app.state.density_cache mapping stream_id → stream data.
        severity_filter: Optional severity string to filter by.
        zone_id_filter: Optional zone_id to filter by.
        stream_id_filter: Optional stream_id to filter by.
        alert_status_filter: Optional status to filter by.
        city_filter: Optional city label to restrict alerts to one city.

    Returns:
        List of AlertResponse for the matching zones.
    """
    now = datetime.now(tz=timezone.utc)
    items: list[AlertResponse] = []
    for sid, data in cache.items():
        if stream_id_filter is not None and sid != stream_id_filter:
            continue
        if city_filter and not _camera_matches_city(data, city_filter):
            continue
        for z in data.get("zones", []):
            zid = z.get("zone_id", "")
            sev = z.get("severity", "SAFE")
            if sev != "CRITICAL":
                continue  # only surface fired alerts, not normal zones
            if severity_filter and sev != severity_filter.upper():
                continue
            if zone_id_filter and zid != zone_id_filter:
                continue
            if alert_status_filter and alert_status_filter != "active":
                continue
            seed = f"{sid}:{zid}"
            alert_uuid = str(
                uuid.UUID(
                    hashlib.md5(seed.encode(), usedforsecurity=False).hexdigest()
                )
            )
            items.append(
                AlertResponse(
                    alert_id=alert_uuid,
                    timestamp=now,
                    stream_id=int(sid),
                    zone_id=zid,
                    zone_name=z.get("zone_name") or zid,
                    zone_type=z.get("zone_type") or "crowd_density",
                    location_name=data.get("location_name") or "",
                    lat=float(data.get("lat") or 0.0),
                    lon=float(data.get("lon") or 0.0),
                    person_count=int(z.get("person_count") or 0),
                    threshold=int(z.get("threshold") or 0),
                    severity=sev,
                    violation_type=z.get("violation_type"),
                    description=(
                        f"Live: {z.get('person_count', 0)} persons "
                        f"detected in {z.get('zone_name', zid)}"
                    ),
                    is_auto_popup=True,
                    status="active",
                    acknowledged_by=None,
                    acknowledged_at=None,
                )
            )
    return items


@router.get(
    "/alerts",
    response_model=PaginatedResponse[AlertResponse],
)
async def list_alerts(
    request: Request,
    severity: Optional[str] = None,
    zone_id: Optional[str] = None,
    zone_type: Optional[str] = None,
    violation_type: Optional[str] = None,
    stream_id: Optional[int] = None,
    city: Optional[str] = Query(None, description="City label to filter alerts"),
    alert_status: Optional[str] = Query(None, alias="status"),
    from_ts: Optional[str] = Query(None, description="ISO datetime lower bound"),
    to_ts: Optional[str] = Query(None, description="ISO datetime upper bound"),
    limit: int = 50,
    offset: int = 0,
) -> PaginatedResponse[AlertResponse]:
    """Return paginated historical alerts with optional filters."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        cache: dict = getattr(request.app.state, "density_cache", {})
        items = _alerts_from_cache(
            cache, severity, zone_id, stream_id, alert_status, city
        )
        page = items[offset: offset + limit]
        return PaginatedResponse(
            items=page, total=len(items), limit=limit, offset=offset
        )

    where_sql, params = _build_alerts_where(
        severity=severity,
        zone_id=zone_id,
        zone_type=zone_type,
        violation_type=violation_type,
        alert_status=alert_status,
        stream_id=stream_id,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    count_q, data_q = _build_alerts_queries(where_sql, len(params))
    params_data = params + [limit, offset]

    async with pool.acquire() as conn:
        total = await conn.fetchval(count_q, *params)
        rows = await conn.fetch(data_q, *params_data)

    items = [
        AlertResponse(
            alert_id=str(r["alert_id"]),
            timestamp=r["timestamp"],
            stream_id=r["stream_id"],
            zone_id=r["zone_id"],
            zone_name=r["zone_name"] or "",
            zone_type=r["zone_type"] or "",
            location_name=r["location_name"] or "",
            lat=r["lat"] or 0.0,
            lon=r["lon"] or 0.0,
            person_count=r["person_count"] or 0,
            threshold=r["threshold"] or 0,
            severity=r["severity"] or "SAFE",
            violation_type=r["violation_type"],
            description=r["description"] or "",
            is_auto_popup=r["is_auto_popup"] or False,
            status=r["status"] or "active",
            acknowledged_by=r["acknowledged_by"],
            acknowledged_at=r["acknowledged_at"],
        )
        for r in rows
    ]
    return PaginatedResponse(
        items=items, total=total or 0, limit=limit, offset=offset
    )


@router.patch("/alerts/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: str,
    body: AlertAcknowledgeRequest,
    request: Request,
) -> AlertResponse:
    """Acknowledge an active alert (operator+)."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="DB unavailable.")
    operator = body.operator_id or "operator"
    now = datetime.now(tz=timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE crowd_alerts SET status='acknowledged', "
            "acknowledged_by=$1, acknowledged_at=$2, acknowledge_reason=$3 "
            "WHERE alert_id=$4::uuid",
            operator, now, body.reason, alert_id,
        )
        row = await conn.fetchrow(
            "SELECT alert_id::text, timestamp, stream_id, zone_id, zone_name, "
            "zone_type, violation_type, location_name, lat, lon, person_count, "
            "threshold, severity, description, is_auto_popup, status, "
            "acknowledged_by, acknowledged_at FROM crowd_alerts "
            "WHERE alert_id=$1::uuid ORDER BY timestamp DESC LIMIT 1",
            alert_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return AlertResponse(
        alert_id=str(row["alert_id"]),
        timestamp=row["timestamp"],
        stream_id=row["stream_id"],
        zone_id=row["zone_id"],
        zone_name=row["zone_name"] or "",
        zone_type=row["zone_type"] or "",
        location_name=row["location_name"] or "",
        lat=row["lat"] or 0.0,
        lon=row["lon"] or 0.0,
        person_count=row["person_count"] or 0,
        threshold=row["threshold"] or 0,
        severity=row["severity"] or "SAFE",
        violation_type=row["violation_type"],
        description=row["description"] or "",
        is_auto_popup=row["is_auto_popup"] or False,
        status=row["status"] or "acknowledged",
        acknowledged_by=row["acknowledged_by"],
        acknowledged_at=row["acknowledged_at"],
    )

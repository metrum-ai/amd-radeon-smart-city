# Created by Metrum AI for AMD

"""Tool execution handlers for GAIA-backed multi-agent reporting."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


def _resolve_window(
    args: dict[str, Any],
) -> tuple[datetime, datetime]:
    """Return (start_ts, end_ts) as UTC datetimes from args.

    Priority:
    1. ``from_ts`` / ``to_ts`` absolute ISO date strings (injected by the
       agent when the user supplied a date range).
    2. ``hours`` relative lookback from NOW() (default 24).
    """
    now = datetime.now(timezone.utc)
    from_ts_raw = args.get("from_ts")
    to_ts_raw = args.get("to_ts")
    if from_ts_raw and to_ts_raw:
        try:
            start = datetime.fromisoformat(str(from_ts_raw)).replace(
                tzinfo=timezone.utc
            )
            # to_ts is end-of-day for date-only strings.
            end = datetime.fromisoformat(str(to_ts_raw)).replace(
                tzinfo=timezone.utc
            ) + timedelta(days=1)
            return start, min(end, now)
        except ValueError:
            pass
    hours = int(args.get("hours", 24))
    hours = max(1, min(hours, 720))
    return now - timedelta(hours=hours), now


async def execute_tool(
    *,
    tool_name: str,
    args: dict[str, Any],
    db_pool: Optional[Any],
    vector_store: Optional[Any],
    embedder: Optional[Any],
) -> str:
    """Execute one tool call and return JSON string output."""
    if tool_name == "get_zone_alerts":
        return await _get_zone_alerts(db_pool=db_pool, args=args)
    if tool_name == "get_density_trends":
        return await _get_density_trends(db_pool=db_pool, args=args)
    if tool_name == "get_historical_patterns":
        return await _get_historical_patterns(db_pool=db_pool, args=args)
    if tool_name == "search_sop_documents":
        return _search_sop_documents(
            vector_store=vector_store,
            embedder=embedder,
            args=args,
        )
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


async def _get_zone_alerts(*, db_pool: Optional[Any], args: dict[str, Any]) -> str:
    """Return recent zone alert rows from TimescaleDB.

    When ``crowd_alerts`` returns no rows (SAFE zone or empty window), falls
    back to ``crowd_counts`` so the investigator always has observation data.
    """
    if db_pool is None:
        return json.dumps({"rows": [], "note": "db_pool unavailable"})

    zone_id = str(args.get("zone_id", ""))
    start, end = _resolve_window(args)

    alert_query = (
        "SELECT timestamp, zone_name, zone_type, violation_type, "
        "person_count, threshold, severity, description "
        "FROM crowd_alerts "
        "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
        "ORDER BY timestamp DESC LIMIT 50"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(alert_query, zone_id, start, end)

    if rows:
        payload = [
            {
                "timestamp": str(row["timestamp"]),
                "zone_name": row["zone_name"],
                "zone_type": row["zone_type"],
                "violation_type": row["violation_type"],
                "person_count": row["person_count"],
                "threshold": row["threshold"],
                "severity": row["severity"],
                "description": row["description"],
            }
            for row in rows
        ]
        return json.dumps({"rows": payload})

    # No threshold violations in window — query crowd_counts for observations.
    counts_query = (
        "SELECT timestamp, zone_name, person_count, density_score, severity "
        "FROM crowd_counts "
        "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
        "ORDER BY timestamp DESC LIMIT 50"
    )
    async with db_pool.acquire() as conn:
        count_rows = await conn.fetch(counts_query, zone_id, start, end)

    if not count_rows:
        return json.dumps({
            "rows": [],
            "note": (
                f"No alerts or observations recorded for zone '{zone_id}' "
                f"between {start.strftime('%Y-%m-%d %H:%M')} UTC and "
                f"{end.strftime('%Y-%m-%d %H:%M')} UTC."
            ),
        })

    payload = [
        {
            "timestamp": str(row["timestamp"]),
            "zone_name": row["zone_name"],
            "zone_type": None,
            "violation_type": None,
            "person_count": row["person_count"],
            "threshold": None,
            "severity": row["severity"],
            "description": (
                f"Observation: {row['person_count']} persons, "
                f"density {float(row['density_score'] or 0):.2f} — SAFE"
            ),
        }
        for row in count_rows
    ]
    return json.dumps({
        "rows": payload,
        "note": "No threshold violations; showing crowd-count observations.",
    })


async def _get_density_trends(
    *, db_pool: Optional[Any], args: dict[str, Any]
) -> str:
    """Return hourly density aggregates for a zone."""
    if db_pool is None:
        return json.dumps({"rows": [], "note": "db_pool unavailable"})

    zone_id = str(args.get("zone_id", ""))
    start, end = _resolve_window(args)

    query = (
        "SELECT date_trunc('hour', timestamp) AS hour_bucket, "
        "AVG(person_count) AS avg_count, MAX(person_count) AS peak_count, "
        "AVG(density_score) AS avg_density "
        "FROM crowd_counts "
        "WHERE zone_id=$1 AND timestamp >= $2 AND timestamp < $3 "
        "GROUP BY hour_bucket "
        "ORDER BY hour_bucket ASC "
        "LIMIT 240"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, zone_id, start, end)

    payload = [
        {
            "hour_bucket": str(row["hour_bucket"]),
            "avg_count": float(row["avg_count"] or 0.0),
            "peak_count": int(row["peak_count"] or 0),
            "avg_density": float(row["avg_density"] or 0.0),
        }
        for row in rows
    ]
    return json.dumps({"rows": payload})


_PATTERN_DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


async def _get_historical_patterns(
    *, db_pool: Optional[Any], args: dict[str, Any]
) -> str:
    """Return recurring patterns for a zone.

    Primary: recurring_patterns table.
    Fallback: derives patterns from crowd_counts when the table has no rows
    for the requested zone (fresh install / insufficient alert history).
    """
    if db_pool is None:
        return json.dumps({"rows": [], "note": "db_pool unavailable"})

    zone_id = str(args.get("zone_id", ""))
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT zone_name, pattern_description, frequency, peak_day, "
            "peak_hour_start, peak_hour_end, avg_count, occurrences, last_updated "
            "FROM recurring_patterns "
            "WHERE zone_id=$1 "
            "ORDER BY occurrences DESC, last_updated DESC "
            "LIMIT 20",
            zone_id,
        )

        if not rows:
            rows = await conn.fetch(
                """
                SELECT
                    zone_id, zone_name,
                    EXTRACT(DOW FROM timestamp)::int  AS dow,
                    EXTRACT(HOUR FROM timestamp)::int AS hour,
                    COUNT(*)                          AS occurrences,
                    AVG(person_count)                 AS avg_count
                FROM crowd_counts
                WHERE zone_id = $1
                  AND timestamp >= NOW() - INTERVAL '30 days'
                  AND person_count > 5
                GROUP BY zone_id, zone_name, dow, hour
                HAVING AVG(person_count) > 10
                ORDER BY avg_count DESC
                LIMIT 20
                """,
                zone_id,
            )
            payload = []
            for row in rows:
                dow = int(row["dow"])
                hour = int(row["hour"])
                day_name = _PATTERN_DAYS[dow % 7]
                avg = float(row["avg_count"] or 0.0)
                payload.append({
                    "zone_name": row["zone_name"] or zone_id,
                    "pattern_description": (
                        f"Peak crowd activity on {day_name} around {hour:02d}:00 "
                        f"(avg {avg:.0f} persons)"
                    ),
                    "frequency": f"Weekly on {day_name} {hour:02d}:00",
                    "peak_day": day_name,
                    "peak_hour_start": hour,
                    "peak_hour_end": min(hour + 1, 23),
                    "avg_count": avg,
                    "occurrences": int(row["occurrences"] or 0),
                    "last_updated": None,
                })
            return json.dumps({"rows": payload, "source": "crowd_counts"})

    payload = [
        {
            "zone_name": row["zone_name"],
            "pattern_description": row["pattern_description"],
            "frequency": row["frequency"],
            "peak_day": row["peak_day"],
            "peak_hour_start": row["peak_hour_start"],
            "peak_hour_end": row["peak_hour_end"],
            "avg_count": float(row["avg_count"] or 0.0),
            "occurrences": int(row["occurrences"] or 0),
            "last_updated": str(row["last_updated"]),
        }
        for row in rows
    ]
    return json.dumps({"rows": payload})


def _search_sop_documents(
    *,
    vector_store: Optional[Any],
    embedder: Optional[Any],
    args: dict[str, Any],
) -> str:
    """Return top-k vector search hits for policy query."""
    if vector_store is None or embedder is None:
        return json.dumps({"rows": [], "note": "vector tools unavailable"})

    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"rows": []})

    query_embedding = embedder.embed_one(query)
    hits = vector_store.search(query_embedding, top_k=5)
    payload = [
        {
            "text": hit.get("text", ""),
            "metadata": hit.get("metadata", "{}"),
            "score": float(hit.get("score", 0.0)),
        }
        for hit in hits
    ]
    return json.dumps({"rows": payload})

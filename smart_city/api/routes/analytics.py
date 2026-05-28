# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Analytics endpoints: density, heatmap, patterns, KPIs, weekly heatmap, trends."""
import base64
import datetime as dt
import logging
import random
from builtins import range as builtins_range
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set

from fastapi import APIRouter, Query, Request
import yaml
from smart_city.api.models import (
    DensityResponse, HeatmapResponse, KPIResponse, PatternItem,
    PatternResponse, TrendResponse, TrendZone, WeeklyHeatmapResponse,
    ZoneDensity,
)
from smart_city.core.stream_allocation import apply_city_allocation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analytics"])

# Normalized hourly crowd weights: 0-1 relative to daily peak.
# Pattern: morning rush 7-9, lunch 12-13, evening rush 17-19.
_HOURLY_WEIGHTS = [
    0.04, 0.03, 0.03, 0.03, 0.04, 0.06,   # 0-5
    0.12, 0.48, 0.88, 0.96, 0.76, 0.62,   # 6-11
    0.72, 0.76, 0.66, 0.62, 0.72, 0.92,   # 12-17
    0.96, 0.82, 0.66, 0.50, 0.32, 0.15,   # 18-23
]


def _simulate_hourly_profile(
    current_count: int, now_hour: int
) -> List[int]:
    """Simulate 24-h hourly counts anchored to current_count at now_hour.

    Args:
        current_count: Live person count for the current hour.
        now_hour: Hour of day (0-23) for current_count.

    Returns:
        List of 24 integer counts.
    """
    now_w = _HOURLY_WEIGHTS[now_hour]
    scale = (
        current_count / now_w
        if now_w > 0 and current_count > 0
        else max(current_count, 10)
    )
    result = []
    for h in range(24):
        v = scale * _HOURLY_WEIGHTS[h]
        noise = (random.random() - 0.5) * 0.15 * max(v, 1)
        result.append(max(0, round(v + noise)))
    return result


def _simulate_daily_history(
    today_count: int, n_days: int, now_dow: int
) -> List[int]:
    """Simulate n_days of daily peak counts ending with today_count.

    Weekend days get ~70 % of weekday counts. Noise ±20 %.

    Args:
        today_count: Today's live person count (last element).
        n_days: Total number of days to generate.
        now_dow: Today's weekday index Mon=0 … Sun=6.

    Returns:
        List of n_days integers, last element == today_count.
    """
    result = []
    for i in range(n_days):
        days_ago = n_days - 1 - i
        if days_ago == 0:
            result.append(today_count)
            continue
        dow = (now_dow - days_ago) % 7
        is_weekend = dow >= 5
        wknd_factor = 0.70 if is_weekend else 1.0
        noise = (random.random() - 0.5) * 0.40
        v = today_count * wknd_factor * (1.0 + noise)
        result.append(max(0, round(v)))
    return result


def _simulate_weekly_heatmap(baseline_pct: float) -> List[List[float]]:
    """Generate a realistic 7×24 weekly heatmap seeded from live density.

    Args:
        baseline_pct: Live average density percentage (0–100) used to
            scale the simulated grid.

    Returns:
        7×24 grid of density percentages (0–100).
    """
    scale = max(baseline_pct / 55.0, 0.1)  # 55 is typical midday %
    grid: List[List[float]] = []
    for day in range(7):
        row: List[float] = []
        is_weekday = day < 5
        is_friday = day == 4
        is_weekend = day >= 5
        for h in range(24):
            if h < 5:
                base = 4.0
            elif h < 7:
                base = 13.0
            elif is_weekday and 7 <= h <= 10:
                base = 88.0 - abs(h - 9) * 22.0
                if is_friday:
                    base *= 0.92
            elif 12 <= h <= 13:
                base = 62.0 if is_weekend else 54.0
            elif is_weekday and 17 <= h <= 20:
                base = 80.0 - abs(h - 18) * 14.0
                if is_friday:
                    base = min(96.0, base * 1.22)
            elif is_weekend and 10 <= h <= 14:
                base = 58.0 + random.random() * 14.0
            elif is_weekend and 18 <= h <= 22:
                base = 86.0 - abs(h - 20) * 11.0
            elif 6 <= h < 22:
                base = 22.0 + random.random() * 16.0
            else:
                base = 8.0
            v = max(3.0, min(100.0, base * scale + random.random() * 4 - 2))
            row.append(round(v, 1))
        grid.append(row)
    return grid


@router.get("/analytics/density/{stream_id}", response_model=DensityResponse)
async def get_density(stream_id: int, request: Request) -> DensityResponse:
    """Return current per-zone density for a stream."""
    cache = getattr(request.app.state, "density_cache", {})
    data = cache.get(stream_id)
    if data is None:
        return DensityResponse(
            stream_id=stream_id, zones=[], timestamp=datetime.now(tz=timezone.utc)
        )
    zones = [
        ZoneDensity(
            zone_id=z["zone_id"],
            zone_name=z["zone_name"],
            zone_type=z["zone_type"],
            person_count=z["person_count"],
            density_score=z["density_score"],
            severity=z["severity"],
            violation_type=z.get("violation_type"),
        )
        for z in data.get("zones", [])
    ]
    return DensityResponse(
        stream_id=stream_id,
        zones=zones,
        timestamp=datetime.now(tz=timezone.utc),
    )


@router.get("/analytics/heatmap/{stream_id}", response_model=HeatmapResponse)
async def get_heatmap(stream_id: int, request: Request) -> HeatmapResponse:
    """Return the latest base64 JPEG heatmap for a stream."""
    heatmap_cache = getattr(request.app.state, "heatmap_cache", {})
    b64 = heatmap_cache.get(stream_id, "")
    return HeatmapResponse(
        stream_id=stream_id,
        image_b64=b64,
        width=1280,
        height=720,
        timestamp=datetime.now(tz=timezone.utc),
    )


@router.get("/analytics/kpis", response_model=KPIResponse)
async def get_kpis(
    request: Request,
    city: Optional[str] = Query(
        default=None,
        description="City label, e.g. Dallas Downtown, TX",
    ),
) -> KPIResponse:
    """Return live KPI values computed from the density cache."""
    cache: dict = getattr(request.app.state, "density_cache", {})
    allowed_zone_ids = _zone_ids_for_city(city)
    stream_values = list(cache.values())
    if city and allowed_zone_ids is not None:
        stream_values = [
            d for d in stream_values
            if _stream_matches_city(d, city, allowed_zone_ids)
        ]
    active_critical = sum(
        1 for d in stream_values if d.get("severity") == "CRITICAL"
    )
    peak_count = 0
    peak_location = ""
    for data in stream_values:
        if data.get("person_count", 0) > peak_count:
            peak_count = data["person_count"]
            peak_location = data.get("location_name", "")

    latencies = sorted(
        d["latency_ms"]
        for d in stream_values
        if isinstance(d.get("latency_ms"), (int, float)) and d["latency_ms"] > 0
    )
    if latencies:
        idx = max(0, int(len(latencies) * 0.95) - 1)
        p95_latency = round(latencies[idx], 1)
    else:
        p95_latency = 0.0

    return KPIResponse(
        active_critical_zones=active_critical,
        peak_crowd_count=peak_count,
        peak_crowd_location=peak_location,
        streams_live=len(stream_values),
        streams_total=len(stream_values),
        detection_latency_p95=p95_latency,
        timestamp=datetime.now(tz=timezone.utc),
    )


def _fill_heatmap_gaps(grid: List[List[float]]) -> List[List[float]]:
    """Fill zero cells in a 7×24 density grid using weighted hourly profile.

    When the DB has data for some hours but not others (e.g. fresh seed),
    zeros create blank columns on the chart.  This fills them using the
    hourly weight profile scaled to each day's non-zero average.

    Args:
        grid: 7×24 list of density percentages, possibly containing zeros.

    Returns:
        7×24 grid with no zero entries.
    """
    result = [row[:] for row in grid]
    for d in range(7):
        non_zero = [v for v in result[d] if v > 0]
        day_avg = sum(non_zero) / len(non_zero) if non_zero else 0.0
        if day_avg == 0:
            day_avg = 40.0
        for h in range(24):
            if result[d][h] == 0:
                result[d][h] = round(
                    max(1.0, day_avg * _HOURLY_WEIGHTS[h] / 0.5), 1
                )
    return result


@router.get("/analytics/heatmap-weekly", response_model=WeeklyHeatmapResponse)
async def get_weekly_heatmap(
    request: Request,
    city: Optional[str] = Query(
        default=None,
        description="City label, e.g. Dallas Downtown, TX",
    ),
) -> WeeklyHeatmapResponse:
    """Return a 7×24 density grid averaged over the last 30 days.

    Rows are Mon(0)…Sun(6); columns are hours 0–23.
    Values are 0–100 (density percentage).  When no DB is available,
    returns a simulated grid seeded from the live density cache so the
    frontend never receives an all-zero grid.
    """
    pool = getattr(request.app.state, "db_pool", None)
    cache: dict = getattr(request.app.state, "density_cache", {})
    allowed_zone_ids = _zone_ids_for_city(city)
    now = datetime.now(tz=timezone.utc)
    mon_dow = now.weekday() % 7
    hour = now.hour

    live_scores: List[float] = []
    cache_values = list(cache.values())
    if city and allowed_zone_ids is not None:
        cache_values = [
            d for d in cache_values
            if _stream_matches_city(d, city, allowed_zone_ids)
        ]
    for data in cache_values:
        for z in data.get("zones", []):
            score = z.get("density_score")
            if score is not None:
                live_scores.append(float(score) * 100.0)
    live_avg = (
        round(sum(live_scores) / len(live_scores), 1) if live_scores else 0.0
    )

    if pool is not None:
        grid: List[List[float]] = [[0.0] * 24 for _ in range(7)]
        cutoff = now - timedelta(days=30)
        if city and allowed_zone_ids is not None and allowed_zone_ids:
            rows = await pool.fetch(
                """
                SELECT
                    ((EXTRACT(DOW FROM timestamp)::int + 6) % 7) AS mon_dow,
                    EXTRACT(HOUR FROM timestamp)::int AS hour,
                    AVG(density_score * 100) AS avg_density
                FROM crowd_counts
                WHERE density_score IS NOT NULL
                  AND timestamp >= $1
                  AND zone_id = ANY($2)
                GROUP BY mon_dow, hour
                """,
                cutoff,
                list(allowed_zone_ids),
            )
        else:
            rows = await pool.fetch(
                """
                SELECT
                    ((EXTRACT(DOW FROM timestamp)::int + 6) % 7) AS mon_dow,
                    EXTRACT(HOUR FROM timestamp)::int AS hour,
                    AVG(density_score * 100) AS avg_density
                FROM crowd_counts
                WHERE density_score IS NOT NULL
                  AND timestamp >= $1
                GROUP BY mon_dow, hour
                """,
                cutoff,
            )
        for r in rows:
            d = int(r["mon_dow"])
            h = int(r["hour"])
            grid[d][h] = round(float(r["avg_density"] or 0.0), 1)
        # If DB returned no historical rows, seed the grid from live data.
        has_db_data = any(v > 0 for row in grid for v in row)
        if not has_db_data:
            baseline = live_avg if live_avg > 0 else 40.0
            grid = _simulate_weekly_heatmap(baseline)
        else:
            # Fill zero cells using surrounding-hour average so the heatmap
            # has no blank gaps from sparse data.
            grid = _fill_heatmap_gaps(grid)
    else:
        # Simulate a realistic weekly pattern seeded from live density.
        baseline = live_avg if live_avg > 0 else 40.0
        grid = _simulate_weekly_heatmap(baseline)

    # Always blend live cache into current weekday/hour cell so the chart
    # reflects the real-time reading regardless of DB coverage.
    if live_avg > 0:
        grid[mon_dow][hour] = live_avg

    return WeeklyHeatmapResponse(
        grid=grid,
        generated_at=now,
    )


_RANGE_DAYS: dict = {"day": 1, "week": 7, "month": 30, "quarter": 90}


@lru_cache(maxsize=1)
def _load_stream_city_index() -> tuple[dict[str, str], dict[str, str]]:
    """Load zone->city and location->city mappings from stream metadata."""
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


def _zone_ids_for_city(city: Optional[str]) -> Optional[Set[str]]:
    """Return zone IDs for the given city label (None => no filter)."""
    if not city:
        return None
    zone_to_city, _ = _load_stream_city_index()
    return {
        zone_id for zone_id, city_name in zone_to_city.items()
        if city_name == city
    }


def _stream_matches_city(
    stream_data: dict, city: str, allowed_zone_ids: Set[str]
) -> bool:
    """Return True when a stream belongs to the selected city."""
    if not allowed_zone_ids:
        return False
    zones = stream_data.get("zones", [])
    for zone in zones:
        zone_id = str(zone.get("zone_id", "")).strip()
        if zone_id in allowed_zone_ids:
            return True
    _, location_to_city = _load_stream_city_index()
    location_name = str(stream_data.get("location_name", "")).strip()
    if location_to_city.get(location_name) == city:
        return True
    # Fallback for synthesised streams beyond YAML pool size: city is
    # stored directly on the density_cache entry by _resolve_stream_cfg.
    return str(stream_data.get("city", "")).strip() == city


def _zones_from_cache(cache: dict) -> List[dict]:
    """Extract a flat list of zone dicts from the density cache.

    Each zone dict is augmented with the parent stream's ``city`` field
    so that synthesised streams (beyond YAML pool size) can be matched
    by city when zone_id is not in the static YAML index.

    Args:
        cache: app.state.density_cache mapping stream_id → stream data.

    Returns:
        List of zone dicts each containing zone_id, zone_name,
        person_count, density_score, and city.
    """
    zones: List[dict] = []
    for data in cache.values():
        city = str(data.get("city", "")).strip()
        for z in data.get("zones", []):
            if z.get("zone_id"):
                zones.append({**z, "city": city})
    return zones


@router.get("/analytics/trends", response_model=TrendResponse)
async def get_trends(
    request: Request,
    range: str = Query(  # noqa: A002
        "month", description="day | week | month | quarter"
    ),
    city: Optional[str] = Query(
        default=None,
        description="City label, e.g. Dallas Downtown, TX",
    ),
) -> TrendResponse:
    """Return daily peak crowd counts per zone for the chosen time range.

    When no DB is available the endpoint returns simulated data seeded
    from the live density cache so the frontend always displays data.
    """
    range_key = range
    pool = getattr(request.app.state, "db_pool", None)
    now_utc = datetime.now(tz=timezone.utc)
    cache: dict = getattr(request.app.state, "density_cache", {})
    allowed_zone_ids = _zone_ids_for_city(city)

    if range_key == "day":
        hour_range = [
            (now_utc - timedelta(hours=23 - i)).replace(
                minute=0, second=0, microsecond=0
            )
            for i in builtins_range(24)
        ]
        labels = [h.strftime("%H:%M") for h in hour_range]

        if pool is None:
            # Build simulated hourly data from live cache.
            now_hour = now_utc.hour
            live_zones = _zones_from_cache(cache)
            if city and allowed_zone_ids is not None:
                live_zones = [
                    z for z in live_zones
                    if str(z.get("zone_id", "")).strip() in allowed_zone_ids
                    or str(z.get("city", "")).strip() == city
                ]
            if not live_zones:
                return TrendResponse(
                    zones=[],
                    labels=labels,
                    range=range_key,
                    generated_at=now_utc,
                )
            trend_zones = [
                TrendZone(
                    zone_id=z["zone_id"],
                    zone_name=z.get("zone_name") or z["zone_id"],
                    location_name=z.get("zone_name") or z["zone_id"],
                    data=_simulate_hourly_profile(
                        int(z.get("person_count", 0)), now_hour
                    ),
                )
                for z in live_zones
                if z.get("person_count", 0) >= 0
            ]
            return TrendResponse(
                zones=trend_zones,
                labels=labels,
                range=range_key,
                generated_at=now_utc,
            )

        cutoff = now_utc - timedelta(hours=24)
        if city and allowed_zone_ids is not None and allowed_zone_ids:
            rows = await pool.fetch(
                """
                SELECT
                    zone_id,
                    MAX(zone_name) AS zone_name,
                    DATE_TRUNC('hour', timestamp) AS hr,
                    MAX(person_count) AS person_count
                FROM crowd_counts
                WHERE timestamp >= $1
                  AND person_count IS NOT NULL
                  AND zone_id = ANY($2)
                GROUP BY zone_id, DATE_TRUNC('hour', timestamp)
                ORDER BY zone_id, hr ASC
                """,
                cutoff,
                list(allowed_zone_ids),
            )
        else:
            rows = await pool.fetch(
                """
                SELECT
                    zone_id,
                    MAX(zone_name) AS zone_name,
                    DATE_TRUNC('hour', timestamp) AS hr,
                    MAX(person_count) AS person_count
                FROM crowd_counts
                WHERE timestamp >= $1
                  AND person_count IS NOT NULL
                GROUP BY zone_id, DATE_TRUNC('hour', timestamp)
                ORDER BY zone_id, hr ASC
                """,
                cutoff,
            )

        zone_hours: dict = defaultdict(dict)
        zone_names: dict = {}
        for r in rows:
            zid = r["zone_id"]
            zone_hours[zid][r["hr"]] = int(r["person_count"] or 0)
            zone_names[zid] = r["zone_name"] or zid

        current_hour = now_utc.replace(minute=0, second=0, microsecond=0)
        for data in cache.values():
            for z in data.get("zones", []):
                zid = z.get("zone_id")
                if not zid:
                    continue
                if city and allowed_zone_ids is not None:
                    if str(zid).strip() not in allowed_zone_ids:
                        continue
                zone_names.setdefault(zid, z.get("zone_name") or zid)
                zone_hours[zid][current_hour] = int(
                    z.get("person_count", 0)
                )

        zones = []
        for zid, name in sorted(zone_names.items()):
            data_pts = [zone_hours[zid].get(h, 0) for h in hour_range]
            if any(v > 0 for v in data_pts):
                zones.append(
                    TrendZone(
                        zone_id=zid,
                        zone_name=name,
                        location_name=name,
                        data=data_pts,
                    )
                )

        return TrendResponse(
            zones=zones,
            labels=labels,
            range=range_key,
            generated_at=now_utc,
        )

    # -- Multi-day ranges (week / month / quarter) --------------------------
    days = _RANGE_DAYS.get(range_key, 30)
    today = dt.date.today()
    date_range = [
        today - dt.timedelta(days=days - 1 - i)
        for i in builtins_range(days)
    ]
    if range_key == "week":
        labels = [d.strftime("%a %d") for d in date_range]
    elif range_key == "month":
        labels = [d.strftime("%b %d") for d in date_range]
    else:
        labels = [d.strftime("%b %d") for d in date_range]

    if pool is None:
        # Simulate history seeded from live cache values.
        live_zones = _zones_from_cache(cache)
        if city and allowed_zone_ids is not None:
            live_zones = [
                z for z in live_zones
                if str(z.get("zone_id", "")).strip() in allowed_zone_ids
                or str(z.get("city", "")).strip() == city
            ]
        now_dow = now_utc.weekday()
        if not live_zones:
            return TrendResponse(
                zones=[],
                labels=labels,
                range=range_key,
                generated_at=now_utc,
            )
        trend_zones = [
            TrendZone(
                zone_id=z["zone_id"],
                zone_name=z.get("zone_name") or z["zone_id"],
                location_name=z.get("zone_name") or z["zone_id"],
                data=_simulate_daily_history(
                    int(z.get("person_count", 0)), days, now_dow
                ),
            )
            for z in live_zones
        ]
        return TrendResponse(
            zones=trend_zones,
            labels=labels,
            range=range_key,
            generated_at=now_utc,
        )

    cutoff = now_utc - timedelta(days=days)
    if city and allowed_zone_ids is not None and allowed_zone_ids:
        rows = await pool.fetch(
            """
            SELECT
                zone_id,
                MAX(zone_name) AS zone_name,
                DATE_TRUNC('day', timestamp)::date AS day,
                MAX(person_count) AS person_count
            FROM crowd_counts
            WHERE timestamp >= $1
              AND person_count IS NOT NULL
              AND zone_id = ANY($2)
            GROUP BY zone_id, DATE_TRUNC('day', timestamp)::date
            ORDER BY zone_id, day ASC
            """,
            cutoff,
            list(allowed_zone_ids),
        )
    else:
        rows = await pool.fetch(
            """
            SELECT
                zone_id,
                MAX(zone_name) AS zone_name,
                DATE_TRUNC('day', timestamp)::date AS day,
                MAX(person_count) AS person_count
            FROM crowd_counts
            WHERE timestamp >= $1
              AND person_count IS NOT NULL
            GROUP BY zone_id, DATE_TRUNC('day', timestamp)::date
            ORDER BY zone_id, day ASC
            """,
            cutoff,
        )

    zone_days: dict = defaultdict(dict)
    zone_names: dict = {}
    for r in rows:
        zid = r["zone_id"]
        zone_days[zid][r["day"]] = int(r["person_count"] or 0)
        zone_names[zid] = r["zone_name"] or zid

    if not zone_names:
        # DB has no historical data yet — simulate from live cache.
        live_zones = _zones_from_cache(cache)
        if city and allowed_zone_ids is not None:
            live_zones = [
                z for z in live_zones
                if str(z.get("zone_id", "")).strip() in allowed_zone_ids
                or str(z.get("city", "")).strip() == city
            ]
        now_dow = now_utc.weekday()
        trend_zones = [
            TrendZone(
                zone_id=z["zone_id"],
                zone_name=z.get("zone_name") or z["zone_id"],
                location_name=z.get("zone_name") or z["zone_id"],
                data=_simulate_daily_history(
                    int(z.get("person_count", 0)), days, now_dow
                ),
            )
            for z in live_zones
        ]
        return TrendResponse(
            zones=trend_zones,
            labels=labels,
            range=range_key,
            generated_at=now_utc,
        )

    zones = []
    for zid, name in sorted(zone_names.items()):
        data_pts = [zone_days[zid].get(d, 0) for d in date_range]
        if any(v > 0 for v in data_pts):
            zones.append(
                TrendZone(
                    zone_id=zid,
                    zone_name=name,
                    location_name=name,
                    data=data_pts,
                )
            )

    return TrendResponse(
        zones=zones,
        labels=labels,
        range=range_key,
        generated_at=now_utc,
    )


@router.get("/analytics/patterns", response_model=PatternResponse)
async def get_patterns(request: Request) -> PatternResponse:
    """Return detected recurring crowd patterns.

    Primary source: recurring_patterns table (populated by HistoricalPatternAnalyzer
    after 3+ CRITICAL alerts in the same hour/day slot, running every 6 hours).

    Fallback: when that table is empty (fresh install or not enough alert history),
    derives patterns from crowd_counts — the per-stream count time-series written
    every 2 s by the real-time push loop.  This gives useful results immediately
    without waiting for CRITICAL alert accumulation.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        return PatternResponse(patterns=[])

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT zone_id, zone_name, pattern_description, frequency, "
            "peak_day, avg_count, occurrences FROM recurring_patterns "
            "ORDER BY occurrences DESC LIMIT 50"
        )

        if rows:
            return PatternResponse(
                patterns=[
                    PatternItem(
                        zone_id=r["zone_id"],
                        zone_name=r["zone_name"] or "",
                        zone_type="crowd_density",
                        violation_type="crowd_density",
                        pattern_description=r["pattern_description"] or "",
                        frequency=r["frequency"] or "",
                        peak_day=r["peak_day"] or "",
                        avg_count=float(r["avg_count"] or 0),
                        occurrences=int(r["occurrences"] or 0),
                    )
                    for r in rows
                ]
            )

        # Fallback: derive patterns from crowd_counts time-series.
        # Groups by zone + day-of-week + hour, returns slots where the average
        # count exceeds 20 persons — meaningful even after just a few hours of
        # data collection.
        _DAYS = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        fallback_rows = await conn.fetch(
            """
            SELECT
                zone_id,
                zone_name,
                EXTRACT(DOW FROM timestamp)::int  AS dow,
                EXTRACT(HOUR FROM timestamp)::int AS hour,
                COUNT(*)                          AS occurrences,
                AVG(person_count)                 AS avg_count
            FROM crowd_counts
            WHERE timestamp >= NOW() - INTERVAL '30 days'
              AND person_count > 5
            GROUP BY zone_id, zone_name, dow, hour
            HAVING AVG(person_count) > 20
            ORDER BY avg_count DESC
            LIMIT 50
            """
        )

    patterns = []
    for r in fallback_rows:
        dow = int(r["dow"])
        day_name = _DAYS[dow % 7]
        hour = int(r["hour"])
        avg = float(r["avg_count"] or 0)
        occ = int(r["occurrences"] or 0)
        patterns.append(
            PatternItem(
                zone_id=r["zone_id"],
                zone_name=r["zone_name"] or r["zone_id"],
                zone_type="crowd_density",
                violation_type="crowd_density",
                pattern_description=(
                    f"Peak crowd activity on {day_name} around {hour:02d}:00 "
                    f"(avg {avg:.0f} persons)"
                ),
                frequency=f"Weekly on {day_name} {hour:02d}:00",
                peak_day=day_name,
                avg_count=avg,
                occurrences=occ,
            )
        )
    return PatternResponse(patterns=patterns)

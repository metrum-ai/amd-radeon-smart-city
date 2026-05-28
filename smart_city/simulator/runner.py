# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Batch seed script: populates TimescaleDB with 2 months of synthetic crowd data.

Usage (via docker compose):
    docker compose --profile seed run --rm simulator

Usage (local):
    DATABASE_URL=postgresql://... python -m smart_city.simulator.runner

The script:
- Generates hourly crowd counts for the current month and the previous month.
- Skips today: for today it only backfills hours that have no real pipeline data.
- Applies a 60-minute per-zone cooldown before firing a CRITICAL alert row.
- Clears existing is_simulated rows in the date window before inserting.
- Updates the simulator_state singleton at the end.
"""

import argparse
import asyncio
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import yaml

from smart_city.core.stream_allocation import apply_city_allocation
from smart_city.simulator.data_gen import (
    ZoneSpec,
    _crowd_count,
    _density_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_zones_from_metadata() -> list[ZoneSpec]:
    """Load ZoneSpec list from config/streams_metadata.yaml.

    Falls back to 3 hard-coded zones if the file is unavailabl
    Returns:
        List of ZoneSpec objects for all configured streams.
    """
    _cfg_path = (
        Path(__file__).resolve().parents[1] / "config" / "streams_metadata.yaml"
    )
    try:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        zones = []
        for s in apply_city_allocation(list(_cfg.get("streams", []))):
            sid = int(s["id"])
            zone_id = s.get("zone_id") or f"cam{sid}_main"
            zone_type = s.get("zone_type", "crowd_density")
            loc = s.get("location_name", f"Camera {sid}")
            peak = 250
            zones.append(ZoneSpec(
                stream_id=sid,
                zone_id=zone_id,
                zone_name=loc,
                zone_type=zone_type,
                threshold_critical=int(s.get("threshold_critical", 200)),
                location_name=loc,
                lat=float(s.get("lat", 0.0)),
                lon=float(s.get("lon", 0.0)),
                peak_capacity=peak,
                city=str(s.get("city", "")),
            ))
        return zones
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        logger.warning(
            "streams_metadata.yaml unavailable: %s — using 3 defaults",
            exc,
            exc_info=True,
        )
        return [
            ZoneSpec(
                1, "congress_plaza_main", "Congress Avenue Main Plaza",
                "crowd_density", 200, "Congress Avenue Main Plaza",
                30.2672, -97.7431, 300,
            ),
            ZoneSpec(
                2, "discovery_green_main", "Discovery Green Park",
                "crowd_density", 200, "Discovery Green Park",
                29.7604, -95.3698, 250,
            ),
        ]


_DEFAULT_ZONES: list[ZoneSpec] = _load_zones_from_metadata()

_VIOLATION_MAP: dict[str, str] = {
    "crowd_density": "crowd_density",
}

BATCH_SIZE = 1_000
ALERT_COOLDOWN = timedelta(minutes=60)


async def _flush_counts(conn: asyncpg.Connection, rows: list) -> None:
    await conn.executemany(
        """
        INSERT INTO crowd_counts
            (timestamp, stream_id, zone_id, zone_name,
             person_count, density_score, severity, is_simulated,
             city, lat, lon)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
        rows,
    )


async def _flush_alerts(conn: asyncpg.Connection, rows: list) -> None:
    await conn.executemany(
        """
        INSERT INTO crowd_alerts
            (timestamp, stream_id, zone_id, zone_name, zone_type,
             violation_type, location_name, lat, lon,
             person_count, threshold, severity, description,
             is_auto_popup, status, is_simulated)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
        """,
        rows,
    )


def _two_month_window(now: datetime) -> tuple[datetime, datetime]:
    """Return (start_of_previous_month, start_of_today).

    Args:
        now: Current UTC datetime.

    Returns:
        Tuple of (window_start, today_midnight_utc).
    """
    today_midnight = now.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # First day of current month
    first_of_current = today_midnight.replace(day=1)
    # First day of previous month
    if first_of_current.month == 1:
        first_of_previous = first_of_current.replace(
            year=first_of_current.year - 1, month=12
        )
    else:
        first_of_previous = first_of_current.replace(
            month=first_of_current.month - 1
        )
    return first_of_previous, today_midnight


async def _fetch_today_covered_hours(
    pool: asyncpg.Pool,
    today_start: datetime,
    today_end: datetime,
) -> set[tuple[int, int]]:
    """Return (stream_id, hour) pairs that already have real data today.

    Args:
        pool: asyncpg connection pool.
        today_start: Start of today (UTC midnight).
        today_end: End of today (UTC).

    Returns:
        Set of (stream_id, hour) tuples that already have rows.
    """
    rows = await pool.fetch(
        """
        SELECT DISTINCT stream_id,
               EXTRACT(HOUR FROM timestamp)::int AS hour
        FROM crowd_counts
        WHERE timestamp >= $1
          AND timestamp < $2
          AND is_simulated = FALSE
        """,
        today_start,
        today_end,
    )
    return {(int(r["stream_id"]), int(r["hour"])) for r in rows}


async def seed(
    db_url: str,
    retire_after: int,
) -> None:
    """Generate and persist synthetic data for the 2-month window.

    Args:
        db_url: asyncpg-compatible DSN.
        retire_after: Days after which simulator retires to real-data mode.
    """
    pool = await asyncpg.create_pool(dsn=db_url, min_size=2, max_size=5)
    try:
        await _run_seed(pool, retire_after)
    finally:
        await pool.close()


async def _run_seed(pool: asyncpg.Pool, retire_after: int) -> None:
    now = datetime.now(tz=timezone.utc)
    window_start, today_midnight = _two_month_window(now)
    step = timedelta(hours=1)
    rng = random.Random(42)

    total_days = (today_midnight - window_start).days
    logger.info(
        "Seeding %d days (%s → %s) at 1-hr intervals × %d zones...",
        total_days,
        window_start.date(),
        today_midnight.date(),
        len(_DEFAULT_ZONES),
    )

    # Clear previous simulated rows in the historical window (not today)
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM crowd_counts "
            "WHERE is_simulated = TRUE "
            "AND timestamp >= $1 AND timestamp < $2",
            window_start, today_midnight,
        )
        await conn.execute(
            "DELETE FROM crowd_alerts "
            "WHERE is_simulated = TRUE "
            "AND timestamp >= $1 AND timestamp < $2",
            window_start, today_midnight,
        )
        logger.info("Cleared existing simulated data in historical window.")

    # ----------------------------------------------------------------
    # Phase 1: historical hourly data (previous month + current month
    #          up to but NOT including today)
    # ----------------------------------------------------------------
    count_rows: list = []
    alert_rows: list = []
    alert_cooldowns: dict[str, datetime] = {}
    total_counts = 0
    total_alerts = 0
    total_steps = int(
        (today_midnight - window_start).total_seconds() / step.total_seconds()
    )
    step_num = 0
    logged_pct = -1

    t = window_start
    while t < today_midnight:
        for zone in _DEFAULT_ZONES:
            count = _crowd_count(t, zone, rng)
            density = _density_score(count, zone.peak_capacity)
            severity = (
                "CRITICAL" if count >= zone.threshold_critical else "SAFE"
            )

            count_rows.append((
                t, zone.stream_id, zone.zone_id, zone.zone_name,
                count, density, severity, True,
                zone.city or None, zone.lat or None, zone.lon or None,
            ))

            if severity == "CRITICAL":
                last = alert_cooldowns.get(zone.zone_id)
                if last is None or (t - last) >= ALERT_COOLDOWN:
                    vtype = _VIOLATION_MAP.get(zone.zone_type, "crowd_density")
                    desc = f"CRITICAL alert in zone '{zone.zone_name}'."
                    alert_rows.append((
                        t, zone.stream_id, zone.zone_id, zone.zone_name,
                        zone.zone_type, vtype, zone.location_name,
                        zone.lat, zone.lon, count, zone.threshold_critical,
                        "CRITICAL", desc, True, "active", True,
                    ))
                    alert_cooldowns[zone.zone_id] = t
                    total_alerts += 1

        total_counts += len(_DEFAULT_ZONES)

        if len(count_rows) >= BATCH_SIZE:
            async with pool.acquire() as conn:
                await _flush_counts(conn, count_rows)
            count_rows = []

        if len(alert_rows) >= BATCH_SIZE:
            async with pool.acquire() as conn:
                await _flush_alerts(conn, alert_rows)
            alert_rows = []

        t += step
        step_num += 1
        pct = int(100 * step_num / max(total_steps, 1))
        if pct >= logged_pct + 10:
            logger.info("Historical progress: %d%% (%d rows)", pct, total_counts)
            logged_pct = pct

    # Flush remaining historical rows
    if count_rows:
        async with pool.acquire() as conn:
            await _flush_counts(conn, count_rows)
        count_rows = []
    if alert_rows:
        async with pool.acquire() as conn:
            await _flush_alerts(conn, alert_rows)
        alert_rows = []

    # ----------------------------------------------------------------
    # Phase 2: today backfill — only hours missing real pipeline data
    # ----------------------------------------------------------------
    covered = await _fetch_today_covered_hours(pool, today_midnight, now)
    current_hour = now.hour
    backfill_count = 0

    for hour_offset in range(current_hour + 1):
        t_hour = today_midnight + timedelta(hours=hour_offset)
        for zone in _DEFAULT_ZONES:
            if (zone.stream_id, hour_offset) in covered:
                continue
            count = _crowd_count(t_hour, zone, rng)
            density = _density_score(count, zone.peak_capacity)
            severity = (
                "CRITICAL" if count >= zone.threshold_critical else "SAFE"
            )
            count_rows.append((
                t_hour, zone.stream_id, zone.zone_id, zone.zone_name,
                count, density, severity, True,
                zone.city or None, zone.lat or None, zone.lon or None,
            ))
            backfill_count += 1

            if severity == "CRITICAL":
                last = alert_cooldowns.get(zone.zone_id)
                if last is None or (t_hour - last) >= ALERT_COOLDOWN:
                    vtype = _VIOLATION_MAP.get(zone.zone_type, "crowd_density")
                    desc = f"CRITICAL alert in zone '{zone.zone_name}'."
                    alert_rows.append((
                        t_hour, zone.stream_id, zone.zone_id, zone.zone_name,
                        zone.zone_type, vtype, zone.location_name,
                        zone.lat, zone.lon, count, zone.threshold_critical,
                        "CRITICAL", desc, True, "active", True,
                    ))
                    alert_cooldowns[zone.zone_id] = t_hour
                    total_alerts += 1

    if count_rows:
        async with pool.acquire() as conn:
            await _flush_counts(conn, count_rows)
    if alert_rows:
        async with pool.acquire() as conn:
            await _flush_alerts(conn, alert_rows)

    logger.info(
        "Today backfill: %d rows added (%d hours had real data, skipped).",
        backfill_count, len(covered),
    )

    # Update simulator_state singleton
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO simulator_state
                (state_id, mode, seeded_days, seed_start, seed_end,
                 retire_after_days)
            VALUES (1, 'active', $1, $2, $3, $4)
            ON CONFLICT (state_id) DO UPDATE SET
                mode = 'active',
                seeded_days = $1,
                seed_start = $2,
                seed_end = $3,
                retire_after_days = $4,
                last_checked = NOW()
            """,
            total_days, window_start, now, retire_after,
        )

    logger.info(
        "Seed complete: %d historical count rows, %d alert rows inserted.",
        total_counts + backfill_count, total_alerts,
    )


def main() -> None:
    """Parse CLI args and run the seed coroutine."""
    parser = argparse.ArgumentParser(
        description=(
            "Seed TimescaleDB with synthetic crowd data "
            "(current month + previous month, hourly)."
        )
    )
    parser.add_argument(
        "--retire-after", type=int,
        default=int(os.environ.get("SIMULATE_RETIRE_AFTER_DAYS", "7")),
        help="Days before retiring simulator (default 7)",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL environment variable must be set")
    asyncio.run(seed(db_url, args.retire_after))


if __name__ == "__main__":
    main()

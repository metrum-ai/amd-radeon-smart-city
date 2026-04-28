# Created by Metrum AI for AMD

"""Historical pattern analyser: detects recurring crowd density patterns."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class RecurringPattern:
    """A detected recurring crowd/violation pattern.

    Attributes:
        zone_id: Zone identifier.
        zone_name: Human-readable zone name.
        zone_type: Zone type string.
        violation_type: Always "crowd_density".
        pattern_description: Human-readable pattern description.
        frequency: e.g. "Weekly on Monday 08:00–09:00".
        peak_day: Day of week string (e.g. "Monday").
        peak_hour_start: Starting hour (0–23).
        peak_hour_end: Ending hour (0–23).
        avg_count: Average count during peak.
        occurrences: Number of times this pattern was observed.
    """

    zone_id: str
    zone_name: str
    zone_type: str
    violation_type: str
    pattern_description: str
    frequency: str
    peak_day: str
    peak_hour_start: int
    peak_hour_end: int
    avg_count: float
    occurrences: int


_DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


class HistoricalPatternAnalyzer:
    """Queries TimescaleDB to detect recurring crowd patterns.

    Args:
        db_pool: asyncpg connection pool.
    """

    def __init__(self, db_pool) -> None:
        """Initialise with a database pool reference."""
        self._pool = db_pool

    async def analyze_weekly_patterns(
        self, lookback_days: int = 30
    ) -> List[RecurringPattern]:
        """Detect recurring patterns from alerts + crowd_counts.

        Primary source: CRITICAL alerts (>= 3 occurrences per zone/day/hour).
        Secondary source: crowd_counts time-series, used to fill any zone/day/hour
        slots not covered by alert-based patterns — ensures recurring_patterns is
        populated immediately on fresh installs without waiting for alert history.

        Args:
            lookback_days: How many days to look back.

        Returns:
            Merged list of RecurringPattern instances, alert-derived entries first.
        """
        alert_query = """
            SELECT
                zone_id,
                zone_name,
                EXTRACT(DOW FROM timestamp) AS dow,
                EXTRACT(HOUR FROM timestamp) AS hour,
                COUNT(*) AS occurrences,
                AVG(person_count) AS avg_count
            FROM crowd_alerts
            WHERE
                severity = 'CRITICAL'
                AND timestamp >= NOW() - ($1 * INTERVAL '1 day')
            GROUP BY zone_id, zone_name, dow, hour
            HAVING COUNT(*) >= 3
            ORDER BY occurrences DESC
        """
        counts_query = """
            SELECT
                zone_id,
                zone_name,
                EXTRACT(DOW FROM timestamp)::int  AS dow,
                EXTRACT(HOUR FROM timestamp)::int AS hour,
                COUNT(*)                          AS occurrences,
                AVG(person_count)                 AS avg_count
            FROM crowd_counts
            WHERE
                timestamp >= NOW() - ($1 * INTERVAL '1 day')
                AND person_count > 5
            GROUP BY zone_id, zone_name, dow, hour
            HAVING COUNT(*) >= 3 AND AVG(person_count) > 10
            ORDER BY avg_count DESC
            LIMIT 100
        """

        patterns: List[RecurringPattern] = []
        seen: set = set()

        def _build(row, description: str, frequency: str) -> RecurringPattern:
            dow = int(row["dow"])
            hour = int(row["hour"])
            day_name = _DAYS[dow % 7]
            return RecurringPattern(
                zone_id=row["zone_id"],
                zone_name=row["zone_name"] or row["zone_id"],
                zone_type="crowd_density",
                violation_type="crowd_density",
                pattern_description=description.format(
                    day=day_name, hour=hour, avg=float(row["avg_count"] or 0)
                ),
                frequency=frequency.format(day=day_name, hour=hour),
                peak_day=day_name,
                peak_hour_start=hour,
                peak_hour_end=min(hour + 1, 23),
                avg_count=float(row["avg_count"] or 0),
                occurrences=int(row["occurrences"]),
            )

        try:
            async with self._pool.acquire() as conn:
                alert_rows = await conn.fetch(alert_query, lookback_days)
                counts_rows = await conn.fetch(counts_query, lookback_days)

            for row in alert_rows:
                p = _build(
                    row,
                    "Recurring CRITICAL alert on {day} around {hour:02d}:00",
                    "Weekly on {day} {hour:02d}:00",
                )
                key = (p.zone_id, p.peak_day, p.peak_hour_start)
                seen.add(key)
                patterns.append(p)

            for row in counts_rows:
                dow = int(row["dow"])
                hour = int(row["hour"])
                day_name = _DAYS[dow % 7]
                key = (row["zone_id"], day_name, hour)
                if key in seen:
                    continue
                seen.add(key)
                p = _build(
                    row,
                    "Peak crowd activity on {day} around {hour:02d}:00 (avg {avg:.0f} persons)",
                    "Weekly on {day} {hour:02d}:00",
                )
                patterns.append(p)

        except (asyncpg.PostgresError, OSError, ValueError) as exc:
            logger.error(
                "Pattern analysis query failed: %s", exc, exc_info=True
            )

        return patterns

    async def upsert_patterns(
        self, patterns: List[RecurringPattern]
    ) -> None:
        """Upsert detected patterns into the recurring_patterns table.

        Args:
            patterns: List of RecurringPattern instances to persist.
        """
        if not patterns:
            return
        query = """
            INSERT INTO recurring_patterns
                (zone_id, zone_name, pattern_description, frequency,
                 peak_day, peak_hour_start, peak_hour_end,
                 avg_count, occurrences, last_updated)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (zone_id, peak_day, peak_hour_start) DO UPDATE SET
                pattern_description = EXCLUDED.pattern_description,
                frequency           = EXCLUDED.frequency,
                occurrences         = EXCLUDED.occurrences,
                avg_count           = EXCLUDED.avg_count,
                last_updated        = NOW()
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    query,
                    [
                        (
                            p.zone_id, p.zone_name, p.pattern_description,
                            p.frequency, p.peak_day, p.peak_hour_start,
                            p.peak_hour_end, p.avg_count, p.occurrences,
                        )
                        for p in patterns
                    ],
                )
        except (asyncpg.PostgresError, OSError, ValueError) as exc:
            logger.error("Pattern upsert failed: %s", exc, exc_info=True)

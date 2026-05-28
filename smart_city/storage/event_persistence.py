# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Async batch event writer for crowd_counts and crowd_alerts tables."""

import asyncio
import logging
from typing import List, Set

import asyncpg

logger = logging.getLogger(__name__)


class EventPersistence:
    """Async batch writer for high-throughput event persistence.

    Accumulates density counts and alerts in memory and flushes to
    TimescaleDB in bulk batches to avoid per-event round-trips.

    Args:
        db_pool: asyncpg connection pool.
        flush_interval_s: Seconds between automatic flushes.
        batch_size: Flush early if buffer reaches this size.
    """

    def __init__(
        self,
        db_pool,
        flush_interval_s: float = 1.0,
        batch_size: int = 100,
    ) -> None:
        """Initialise with empty buffers."""
        self._pool = db_pool
        self._flush_interval = flush_interval_s
        self._batch_size = batch_size
        self._count_buffer: List = []
        self._alert_buffer: List = []
        self._running = False
        self._lock = asyncio.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()

    def submit_count(
        self,
        density_result,
        *,
        city: str = "",
        lat: float = 0.0,
        lon: float = 0.0,
        fps: float = 0.0
    ) -> None:
        """Buffer a crowd count record for batch insertion.

        Args:
            density_result: CrowdDensityResult instance.
            city: City name (e.g. "Austin", "Dallas", "Houston").
            lat: Latitude of the stream location.
            lon: Longitude of the stream location.
            fps: Current frames-per-second for the stream.
        """
        from datetime import datetime, timezone

        self._count_buffer.append(
            (
                datetime.fromtimestamp(
                    density_result.timestamp, tz=timezone.utc
                ),
                density_result.stream_id,
                density_result.zone_id,
                density_result.zone_name,
                density_result.person_count,
                density_result.density_score,
                density_result.severity,
                city or None,
                lat or None,
                lon or None,
                fps or None,
            )
        )
        if len(self._count_buffer) >= self._batch_size:
            task = asyncio.create_task(self._flush_counts())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

       
    def submit_alert(self, alert) -> None:
        """Buffer an alert for batch insertion.

        Args:
            alert: CrowdAlert instance.
        """
        self._alert_buffer.append(alert)
        if len(self._alert_buffer) >= self._batch_size:
            task = asyncio.create_task(self._flush_alerts())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def run(self) -> None:
        """Periodic flush loop — runs until stop() is called."""
        self._running = True
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush_counts()
            await self._flush_alerts()

    async def flush(self) -> None:
        """Force immediate flush of all buffers."""
        await self._flush_counts()
        await self._flush_alerts()

    def stop(self) -> None:
        """Stop the flush loop."""
        self._running = False

    async def _flush_counts(self) -> None:
        """Flush the count buffer to TimescaleDB."""
        if not self._count_buffer or self._pool is None:
            return
        async with self._lock:
            batch = self._count_buffer[:]
            self._count_buffer.clear()
        query = """
            INSERT INTO crowd_counts
                (timestamp, stream_id, zone_id, zone_name,
                 person_count, density_score, severity,
                 city, lat, lon, fps)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, batch)
        except (asyncpg.PostgresError, OSError, ValueError) as exc:
            logger.error("Count flush failed: %s", exc, exc_info=True)

    async def _flush_alerts(self) -> None:
        """Flush the alert buffer to TimescaleDB."""
        if not self._alert_buffer or self._pool is None:
            return
        async with self._lock:
            batch = self._alert_buffer[:]
            self._alert_buffer.clear()
        query = """
            INSERT INTO crowd_alerts
                (alert_id, timestamp, stream_id, zone_id, zone_name,
                 zone_type, violation_type, location_name, lat, lon,
                 person_count, threshold, severity, description,
                 is_auto_popup, status)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                 $11, $12, $13, $14, $15, $16)
            ON CONFLICT DO NOTHING
        """
        rows = [
            (
                alert.alert_id,
                alert.timestamp,
                alert.stream_id,
                alert.zone_id,
                alert.zone_name,
                alert.zone_type,
                alert.violation_type,
                alert.location_name,
                alert.lat,
                alert.lon,
                alert.person_count,
                alert.threshold,
                alert.severity,
                alert.description,
                alert.is_auto_popup,
                alert.status,
            )
            for alert in batch
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(query, rows)
        except (asyncpg.PostgresError, OSError, ValueError) as exc:
            logger.error("Alert flush failed: %s", exc, exc_info=True)

# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Simulator lifecycle mode management.

Tracks whether displayed data is fully simulated, retiring (mix of
simulated + real), or fully replaced by live sensor data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)


class SimulatorMode(str, Enum):
    """Lifecycle mode for the simulator data source."""

    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"


@dataclass
class SimulatorStatus:
    """Snapshot of the current simulator lifecycle state.

    Attributes:
        mode: Current lifecycle mode.
        seeded_days: Number of days of simulated data seeded.
        seed_start: Timestamp of the earliest simulated record.
        seed_end: Timestamp of the latest simulated record.
        real_data_from: Timestamp of the earliest real sensor record.
        retire_after_days: Days of real data required to retire the
            simulator.
        real_days_covered: Fractional days of real data accumulated.
        retirement_pct: Fraction of retire_after_days covered (0–1).
        last_checked: When this status was last computed.
    """

    mode: SimulatorMode = SimulatorMode.ACTIVE
    seeded_days: Optional[int] = None
    seed_start: Optional[datetime] = None
    seed_end: Optional[datetime] = None
    real_data_from: Optional[datetime] = None
    retire_after_days: int = 7
    real_days_covered: float = 0.0
    retirement_pct: float = 0.0
    last_checked: Optional[datetime] = None


class ModeManager:
    """Determines and updates the simulator lifecycle mode.

    Queries TimescaleDB to measure how much real sensor data has
    accumulated and promotes the mode from ACTIVE → RETIRING → RETIRED
    as coverage grows.

    Attributes:
        _db_pool: asyncpg connection pool or single connection.
        _retire_after_days: Days of real data required to retire.
    """

    def __init__(self, db_pool: Any, retire_after_days: int = 7) -> None:
        """Initialise the ModeManager.

        Args:
            db_pool: asyncpg connection pool used to query crowd_counts.
            retire_after_days: Days of real data to trigger retirement.
        """
        self._db_pool = db_pool
        self._retire_after_days = retire_after_days

    async def get_status(self) -> SimulatorStatus:
        """Compute the current simulator status without updating the DB.

        Returns:
            SimulatorStatus populated from DB counts.
        """
        return await self._compute_status()

    async def check_and_update(self) -> SimulatorStatus:
        """Compute status and log a promotion if the mode has changed.

        Returns:
            Current SimulatorStatus.
        """
        status = await self._compute_status()
        logger.info(
            "Simulator mode: %s (%.1f / %d real-data days)",
            status.mode.value,
            status.real_days_covered,
            status.retire_after_days,
        )
        return status

    async def _compute_status(self) -> SimulatorStatus:
        """Query the DB and derive the lifecycle mode.

        Returns:
            SimulatorStatus based on real vs simulated record counts.
        """
        now = datetime.now(tz=timezone.utc)
        status = SimulatorStatus(
            retire_after_days=self._retire_after_days,
            last_checked=now,
        )

        if self._db_pool is None:
            return status

        try:
            # Acquire works for both asyncpg Pool and single Connection
            if hasattr(self._db_pool, "acquire"):
                async with self._db_pool.acquire() as conn:
                    return await self._query_status(conn, status, now)
            else:
                return await self._query_status(
                    self._db_pool, status, now
                )
        except (
            AttributeError,
            asyncpg.PostgresError,
            OSError,
            RuntimeError,
            TypeError,
        ) as exc:
            logger.warning("ModeManager DB query failed: %s", exc)
            return status

    async def _query_status(
        self,
        conn: Any,
        status: SimulatorStatus,
        now: datetime,
    ) -> SimulatorStatus:
        """Execute DB queries to fill in the SimulatorStatus.

        Args:
            conn: asyncpg connection.
            status: Partially populated status to fill in.
            now: Current UTC time.

        Returns:
            Updated SimulatorStatus.
        """
        # Earliest and latest records indicate seeded range
        row = await conn.fetchrow(
            "SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts "
            "FROM crowd_counts"
        )
        if row and row["min_ts"]:
            status.seed_start = row["min_ts"]
            status.seed_end = row["max_ts"]
            if row["min_ts"] and row["max_ts"]:
                delta = (row["max_ts"] - row["min_ts"]).total_seconds()
                status.seeded_days = max(0, int(delta / 86400))

        # Real data starts after seed_end; count days since then
        # For now, treat everything as simulated if no real source marker
        status.real_days_covered = 0.0
        status.retirement_pct = 0.0
        status.mode = SimulatorMode.ACTIVE

        if self._retire_after_days > 0:
            pct = status.real_days_covered / self._retire_after_days
            status.retirement_pct = round(min(1.0, pct), 4)
            if pct >= 1.0:
                status.mode = SimulatorMode.RETIRED
            elif pct > 0:
                status.mode = SimulatorMode.RETIRING

        return status

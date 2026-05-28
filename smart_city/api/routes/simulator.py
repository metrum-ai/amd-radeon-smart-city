# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Simulator status API route.

Exposes the current simulator mode so the dashboard can show a banner
indicating whether displayed data is simulated or real.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from smart_city.simulator.mode_manager import ModeManager, SimulatorMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulator", tags=["simulator"])


class SimulatorStatusResponse(BaseModel):
    """Response schema for the simulator status endpoint."""

    mode: SimulatorMode
    seeded_days: Optional[int]
    seed_start: Optional[str]
    seed_end: Optional[str]
    real_data_from: Optional[str]
    retire_after_days: int
    real_days_covered: float
    retirement_pct: float
    last_checked: Optional[str]
    message: str


def _ts(dt) -> Optional[str]:
    """Serialise datetime to ISO-8601 string or None.

    Args:
        dt: datetime object or None.

    Returns:
        ISO-8601 string or None.
    """
    return dt.isoformat() if dt else None


@router.get(
    "/status",
    response_model=SimulatorStatusResponse,
    summary="Get simulator mode and real-data coverage",
)
async def get_simulator_status(
    request: Request,
) -> SimulatorStatusResponse:
    """Return the current simulator lifecycle state.

    The dashboard uses this to show a 'Simulated data' badge when
    ``mode == 'active'``, a progress indicator when ``mode == 'retiring'``,
    and nothing when ``mode == 'retired'``.

    Returns:
        SimulatorStatusResponse with mode and coverage metrics.
    """
    db = request.app.state.db

    retire_after = int(
        getattr(request.app.state, "sim_retire_after_days", 7)
    )
    mgr = ModeManager(db, retire_after_days=retire_after)
    status = await mgr.get_status()

    messages: Dict[SimulatorMode, str] = {
        SimulatorMode.ACTIVE: (
            "Displaying simulated historical data. "
            "Real data will gradually replace it."
        ),
        SimulatorMode.RETIRING: (
            f"Real data accumulating "
            f"({status.real_days_covered:.1f} / "
            f"{status.retire_after_days} days). "
            "Simulated data still supplements recent gaps."
        ),
        SimulatorMode.RETIRED: (
            "All displayed data is from live sensors. "
            "Simulated seed has been fully superseded."
        ),
    }

    return SimulatorStatusResponse(
        mode=status.mode,
        seeded_days=status.seeded_days,
        seed_start=_ts(status.seed_start),
        seed_end=_ts(status.seed_end),
        real_data_from=_ts(status.real_data_from),
        retire_after_days=status.retire_after_days,
        real_days_covered=round(status.real_days_covered, 2),
        retirement_pct=round(status.retirement_pct, 4),
        last_checked=_ts(status.last_checked),
        message=messages[status.mode],
    )

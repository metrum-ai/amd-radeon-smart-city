# Created by Metrum AI for AMD

"""Threshold alert engine: converts analytics results into CrowdAlert events."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from smart_city.analytics.crowd.density_analyzer import CrowdDensityResult

logger = logging.getLogger(__name__)


@dataclass
class CrowdAlert:
    """A fired crowd or zone violation alert.

    Attributes:
        alert_id: UUID string uniquely identifying this alert.
        timestamp: UTC datetime when the alert was generated.
        stream_id: Source stream identifier.
        zone_id: Zone identifier string.
        zone_name: Human-readable zone name.
        zone_type: Always "crowd_density".
        location_name: Human-readable location label.
        lat: Latitude for map rendering.
        lon: Longitude for map rendering.
        person_count: Person count at alert time.
        threshold: Configured threshold_critical for the zone.
        severity: "SAFE" or "CRITICAL".
        violation_type: Always "crowd_density".
        description: Human-readable summary.
        is_auto_popup: Always True for CRITICAL alerts.
        status: "active" | "acknowledged" | "dismissed".
        acknowledged_by: Username of acknowledging operator; None if unacknowledged.
        acknowledged_at: UTC datetime of acknowledgment; None if unacknowledged.
        acknowledge_reason: Operator note; None if unacknowledged.
    """

    alert_id: str
    timestamp: datetime
    stream_id: int
    zone_id: str
    zone_name: str
    zone_type: str
    location_name: str
    lat: float
    lon: float
    person_count: int
    threshold: int
    severity: str
    violation_type: Optional[str]
    description: str
    is_auto_popup: bool
    status: str = "active"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledge_reason: Optional[str] = None


class ThresholdAlertEngine:
    """Converts analytics results into CrowdAlert events with cooldown.

    Args:
        cooldown_seconds: Minimum seconds between alerts for the same zone.
    """

    def __init__(self, cooldown_seconds: float = 60.0) -> None:
        """Initialise with an empty cooldown tracker."""
        self._cooldown_seconds = cooldown_seconds
        self._last_alert_time: Dict[str, float] = {}
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        """Register a callback to be called with each new CrowdAlert.

        Args:
            callback: Function with signature callback(alert: CrowdAlert).
        """
        self._callbacks.append(callback)

    def evaluate(
        self,
        result: CrowdDensityResult,
        stream_config: Dict,
    ) -> Optional[CrowdAlert]:
        """Evaluate a result and fire an alert if criteria are met.

        Args:
            result: CrowdDensityResult to evaluate.
            stream_config: Dict with location_name, lat, lon, threshold.

        Returns:
            CrowdAlert if fired; None if SAFE or within cooldown.
        """
        if result.severity != "CRITICAL":
            return None

        import time

        now = time.time()
        zone_key = result.zone_id
        if now - self._last_alert_time.get(zone_key, 0) < self._cooldown_seconds:
            return None

        self._last_alert_time[zone_key] = now

        person_count = result.person_count
        threshold = stream_config.get("threshold", 1)

        alert = CrowdAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc),
            stream_id=result.stream_id,
            zone_id=result.zone_id,
            zone_name=result.zone_name,
            zone_type=result.zone_type,
            location_name=stream_config.get("location_name", "Unknown"),
            lat=float(stream_config.get("lat", 0.0)),
            lon=float(stream_config.get("lon", 0.0)),
            person_count=person_count,
            threshold=threshold,
            severity="CRITICAL",
            violation_type=getattr(result, "violation_type", None),
            description=getattr(
                result,
                "description",
                f"CRITICAL alert in zone '{result.zone_name}'.",
            ),
            is_auto_popup=True,
        )

        import inspect

        for cb in self._callbacks:
            try:
                cb_ret = cb(alert)
                if inspect.isawaitable(cb_ret):
                    asyncio.create_task(cb_ret)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.error(
                    "Alert callback error: %s", exc, exc_info=True
                )

        return alert

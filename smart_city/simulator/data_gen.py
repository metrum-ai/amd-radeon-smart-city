# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Synthetic crowd-event data generation for the simulator.

Used by the real-time push loop in startup.py and the standalone
simulator runner to seed TimescaleDB with realistic crowd counts.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ZoneSpec:
    """Specification for a single surveillance zone used by the simulator.

    Attributes:
        stream_id: Numeric ID of the parent camera stream.
        zone_id: Unique string identifier for the zone.
        zone_name: Human-readable zone name.
        zone_type: Always "crowd_density".
        threshold_critical: Headcount that triggers a CRITICAL alert.
        location_name: Human-readable location label.
        lat: WGS-84 latitude.
        lon: WGS-84 longitude.
        peak_capacity: Maximum expected occupancy (for density scoring).
    """

    stream_id: int
    zone_id: str
    zone_name: str
    zone_type: str
    threshold_critical: int
    location_name: str
    lat: float
    lon: float
    peak_capacity: int = 250
    city: str = ""  # city label, e.g. "Austin Downtown, TX"


def _crowd_count(now: datetime, zone: ZoneSpec, rng: random.Random) -> int:
    """Generate a synthetic crowd count for a zone at a given time.

    Models diurnal and weekly patterns with Gaussian noise.

    Args:
        now: Current UTC timestamp.
        zone: Zone specification including capacity and type.
        rng: Seeded random generator for reproducibility.

    Returns:
        Non-negative integer crowd count.
    """
    # crowd_density: diurnal pattern
    hour = now.hour + now.minute / 60.0
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    # Peak hours: 8-10 am, 12-2 pm, 5-7 pm; weekend slightly shifted
    if weekday >= 5:
        base_peak = zone.peak_capacity * 0.85
        morning_peak = 0.5 * math.exp(-0.5 * ((hour - 11) / 2) ** 2)
        evening_peak = 0.8 * math.exp(-0.5 * ((hour - 15) / 2.5) ** 2)
    else:
        base_peak = zone.peak_capacity
        morning_peak = 0.7 * math.exp(-0.5 * ((hour - 9) / 1.5) ** 2)
        midday_peak = 0.5 * math.exp(-0.5 * ((hour - 13) / 1.5) ** 2)
        evening_peak = 0.9 * math.exp(-0.5 * ((hour - 18) / 2) ** 2)
        morning_peak = max(morning_peak, midday_peak)

    diurnal = max(morning_peak, evening_peak)
    mean_count = base_peak * diurnal
    noise = rng.gauss(0, max(5, mean_count * 0.15))
    count = int(max(0, mean_count + noise))
    return count


def _density_score(count: int, peak_capacity: int) -> float:
    """Calculate a 0–1 density score.

    Args:
        count: Observed crowd count.
        peak_capacity: Maximum expected occupancy for the zone.

    Returns:
        Float in [0.0, 1.0], clamped to that range.
    """
    if peak_capacity <= 0:
        return 0.0
    return round(min(1.0, count / peak_capacity), 4)


def _severity_from_count(count: int, threshold: int) -> str:
    """Map a crowd count to a severity string.

    Args:
        count: Observed crowd count.
        threshold: Threshold above which the severity is CRITICAL.

    Returns:
        'CRITICAL' if count >= threshold, otherwise 'SAFE'.
    """
    return "CRITICAL" if count >= threshold else "SAFE"

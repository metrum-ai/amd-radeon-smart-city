# Created by Metrum AI for AMD

"""Crowd density analyser: counts persons per zone via polygon ray-casting."""

import logging
from dataclasses import dataclass
from typing import Dict, List

from smart_city.combined_pipeline.ipc.messages import InferResult

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class index for "person"


def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm to test point-in-polygon.

    Args:
        x: X coordinate of the test point.
        y: Y coordinate of the test point.
        polygon: List of [x, y] vertices.

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


@dataclass
class CrowdDensityResult:
    """Result of crowd density analysis for a single zone.

    Attributes:
        stream_id: Source stream identifier.
        zone_id: Zone identifier string.
        zone_name: Human-readable zone name.
        zone_type: Always "crowd_density".
        person_count: Number of persons detected in the zone.
        density_score: Normalised density 0.0–1.0 (count / threshold_critical).
        severity: "SAFE" or "CRITICAL".
        timestamp: Unix epoch at analysis time.
        violation_type: Always "crowd_density".
    """

    stream_id: int
    zone_id: str
    zone_name: str
    zone_type: str
    person_count: int
    density_score: float
    severity: str
    timestamp: float
    violation_type: str = "crowd_density"


class CrowdDensityAnalyzer:
    """Analyses crowd density for crowd_density zone types.

    Args:
        zones: Dict mapping stream_id → list of ZoneConfig objects.
    """

    def __init__(self, zones: Dict[int, List]) -> None:
        """Initialise with zone configurations keyed by stream_id."""
        self._zones = zones

    def analyze(
        self,
        stream_id: int,
        infer_result: InferResult,
    ) -> List[CrowdDensityResult]:
        """Count persons per crowd_density zone.

        Args:
            stream_id: Stream to analyse.
            infer_result: Detection output from the scratch pipeline.

        Returns:
            List of CrowdDensityResult, one per crowd_density zone.
        """
        results: List[CrowdDensityResult] = []
        stream_zones = self._zones.get(stream_id, [])

        for zone in stream_zones:
            if zone.zone_type != "crowd_density":
                continue

            count = 0
            for det in infer_result.detections:
                if det.class_id != PERSON_CLASS_ID:
                    continue
                cx = (det.x1 + det.x2) / 2
                cy = (det.y1 + det.y2) / 2
                if _point_in_polygon(cx, cy, zone.polygon):
                    count += 1

            threshold = zone.threshold_critical
            density_score = min(count / max(threshold, 1), 1.0)
            severity = "CRITICAL" if count >= threshold else "SAFE"

            results.append(
                CrowdDensityResult(
                    stream_id=stream_id,
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    zone_type=zone.zone_type,
                    person_count=count,
                    density_score=density_score,
                    severity=severity,
                    timestamp=infer_result.ts_ns / 1e9,
                )
            )

        return results

# Created by Metrum AI for AMD

"""Stream-to-city allocation helpers.

When the pipeline runs with ``STREAM_COUNT=N`` active streams, entries are
selected from each city's natural pool in the metadata so each city gets
real coordinates and location names rather than relabelled Austin data.

Allocation formula (``K`` = number of unique cities):

    base = STREAM_COUNT // K
    per_city = [base, base, ..., base]
    per_city[0] += STREAM_COUNT - K * base    # remainder to primary city

Examples for ``K = 3`` cities (Austin / Dallas / Houston):

    STREAM_COUNT=0   -> no-op (all entries returned as-is)
    STREAM_COUNT=1   -> Austin 1, Dallas 0,  Houston 0
    STREAM_COUNT=2   -> Austin 2, Dallas 0,  Houston 0
    STREAM_COUNT=3   -> Austin 1, Dallas 1,  Houston 1
    STREAM_COUNT=20  -> Austin 8, Dallas 6,  Houston 6
    STREAM_COUNT=21  -> Austin 7, Dallas 7,  Houston 7
    STREAM_COUNT=25  -> Austin 9, Dallas 8,  Houston 8
    STREAM_COUNT=50  -> Austin 18, Dallas 16, Houston 16

The returned list contains only the ``STREAM_COUNT`` active entries (drawn
from each city's pool in the correct proportions).  Their original lat/lon
and location_name are preserved — no relabelling is performed.
When ``stream_count`` is 0 or omitted from the environment the full
metadata list is returned unchanged.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional


def _read_stream_count_env() -> int:
    """Read ``STREAM_COUNT`` (or legacy ``NUM_STREAMS``) from env, default 0."""
    raw = os.environ.get("STREAM_COUNT") or os.environ.get("NUM_STREAMS")
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def discover_cities_in_order(streams: Iterable[dict]) -> list[str]:
    """Return unique, non-empty city labels in order of first appearance."""
    seen: list[str] = []
    for s in streams:
        city = str(s.get("city") or "").strip()
        if city and city not in seen:
            seen.append(city)
    return seen


def compute_city_counts(stream_count: int, num_cities: int) -> list[int]:
    """Return per-city stream counts; the first city absorbs the remainder.

    Args:
        stream_count: Total number of active streams (non-negative).
        num_cities: Number of unique cities to allocate across.

    Returns:
        List of length ``num_cities`` whose entries sum to ``stream_count``
        (or to 0 when ``stream_count`` is 0 / negative).  When
        ``num_cities`` is 0 an empty list is returned.
    """
    if num_cities <= 0:
        return []
    if stream_count <= 0:
        return [0] * num_cities
    base = stream_count // num_cities
    counts = [base] * num_cities
    counts[0] += stream_count - base * num_cities
    return counts


def allocate_city_for_position(
    position_0based: int,
    stream_count: int,
    cities: list[str],
) -> Optional[str]:
    """Return the city for slot ``position_0based`` (0..stream_count-1).

    Returns ``None`` when no allocation should be applied — either the
    position is outside the active range, ``stream_count`` is 0, or no
    cities were discovered in the metadata.
    """
    if not cities or stream_count <= 0:
        return None
    if position_0based < 0 or position_0based >= stream_count:
        return None
    counts = compute_city_counts(stream_count, len(cities))
    cumulative = 0
    for city, n in zip(cities, counts):
        cumulative += n
        if position_0based < cumulative:
            return city
    return cities[-1]


def apply_city_allocation(
    streams: list[dict],
    stream_count: Optional[int] = None,
) -> list[dict]:
    """Return the active ``stream_count`` entries drawn from each city's pool.

    Instead of relabelling city on the first N entries (which would assign
    Austin coordinates to Dallas/Houston streams), this function selects the
    correct number of entries from each city's *natural* pool so every
    returned entry keeps its original lat, lon, location_name and city.

    When ``stream_count`` is omitted it is read from ``STREAM_COUNT`` /
    ``NUM_STREAMS`` env vars.  A value of 0 or a missing variable returns
    the full list unchanged.

    Args:
        streams: Stream metadata dicts loaded from streams_metadata.yaml.
        stream_count: Active stream count; defaults to the env var.

    Returns:
        List of shallow-copied dicts, length == min(stream_count, total).
        Each dict preserves its original city, lat, lon, and location_name.
    """
    if not streams:
        return []
    sc = int(stream_count) if stream_count is not None else _read_stream_count_env()
    if sc <= 0:
        return [dict(s) for s in streams]

    cities = discover_cities_in_order(streams)
    if not cities:
        return [dict(s) for s in streams]

    # Build per-city pools (preserving YAML order within each city).
    city_pools: dict[str, list[dict]] = {c: [] for c in cities}
    for s in streams:
        city = str(s.get("city") or "").strip()
        if city in city_pools:
            city_pools[city].append(s)

    counts = compute_city_counts(sc, len(cities))

    # Collect selected entries then renumber IDs to 1..N so callers that
    # look up metadata by sequential pipeline stream-ID (cam1..camN) get
    # the correct city data (lat/lon/location_name) for each slot.
    selected: list[dict] = []
    for city, n in zip(cities, counts):
        pool = city_pools[city]
        for s in pool[:n]:
            selected.append(dict(s))

    for new_id, entry in enumerate(selected, start=1):
        entry["id"] = new_id

    return selected

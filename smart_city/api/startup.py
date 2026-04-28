# Created by Metrum AI for AMD

"""Application startup: initialise all subsystems and attach to app.state."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import asyncpg
import httpx
import yaml
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# pymilvus is an optional dependency in some deployments; fall back to a
# placeholder so this module always imports.
try:  # pragma: no cover - import shim
    from pymilvus.exceptions import MilvusException  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - import shim
    class MilvusException(Exception):  # type: ignore[no-redef]
        """Fallback used when pymilvus is unavailable."""

# ---------------------------------------------------------------------------
# Default zone specs — loaded from streams_metadata.yaml at import time
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc  # noqa: E402


@_dc
class _ZoneSpec:
    stream_id: int
    zone_id: str
    zone_name: str
    zone_type: str
    threshold_critical: int
    location_name: str  # actual place name, e.g. "Congress Avenue Main Plaza"
    lat: float
    lon: float
    city: str = ""  # city label, e.g. "Austin Downtown, TX"
    peak_capacity: int = 250


def _build_default_zones() -> list:
    """Build zone specs from streams_metadata.yaml.

    Raises:
        RuntimeError: If streams_metadata.yaml cannot be loaded.
    """
    import yaml as _yaml  # noqa: PLC0415

    _meta_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "config", "streams_metadata.yaml"
    ))
    with open(_meta_path, encoding="utf-8") as _f:
        _cfg = _yaml.safe_load(_f)
    zones = []
    for s in _cfg.get("streams", []):
        sid = int(s["id"])
        zone_id = s.get("zone_id") or f"cam{sid}_main"
        loc = s.get("location_name", f"Camera {sid}")
        zones.append(_ZoneSpec(
            stream_id=sid,
            zone_id=zone_id,
            zone_name=loc,
            zone_type="crowd_density",
            threshold_critical=int(s.get("threshold_critical", 15)),
            location_name=loc,
            lat=float(s.get("lat", 0.0)),
            lon=float(s.get("lon", 0.0)),
            city=str(s.get("city", "")),
            peak_capacity=250,
        ))
    return zones


_DEFAULT_ZONES = _build_default_zones()

_VIOLATION_TYPE_MAP = {
    "crowd_density": "crowd_density",
}


class _DR:  # noqa: N801
    """Lightweight density-result placeholder for EventPersistence.submit_count."""

    timestamp: float
    stream_id: int
    zone_id: str
    zone_name: str
    person_count: int
    density_score: float
    severity: str


from smart_city.observability.metrics import (  # noqa: E402
    ACTIVE_STREAMS,
    ALERTS_FIRED,
    CROWD_DENSITY_GAUGE,
    STREAM_FPS,
    TOTAL_CROWD_COUNT,
)


def _load_streams_metadata(config_path: str) -> dict[int, dict]:
    """Load per-stream location/threshold metadata from streams_metadata.yaml.

    Returns a dict keyed by stream id (int).  Falls back to empty dict if the
    file is missing (e.g. during development without the full repo layout).
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("streams_metadata.yaml not found at %s — using empty metadata", path)
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {int(s["id"]): s for s in data.get("streams", [])}
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        logger.error(
            "Failed to load streams_metadata.yaml: %s", exc, exc_info=True
        )
        return {}


def _read_pipeline_stats(stats_file: str) -> dict[int, dict]:
    """Read the JSON stats file written by the pipeline's stats-dump thread.

    Returns a dict keyed by stream_id (int) with keys: count, fps, latency_ms, status.
    Returns empty dict if the file doesn't exist or can't be parsed.
    """
    try:
        with open(stats_file) as f:
            raw = json.load(f)
        # JSON keys are strings; convert to int
        return {int(k): v for k, v in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except (OSError, ValueError, TypeError) as exc:
        logger.debug(
            "pipeline_stats read error: %s", exc, exc_info=True
        )
        return {}


async def _realtime_push_loop(app: FastAPI) -> None:
    """Push live pipeline metrics every 2 s to density_cache, WebSocket hub,
    alert engine, and Prometheus gauges.

    Reads real per-stream stats from the pipeline's JSON stats file (written
    by the orchestrator's stats-dump thread).  Falls back gracefully to
    synthetic data when the pipeline is not yet running, so the dashboard
    always shows something useful during startup / demo mode.
    """
    import random
    from datetime import datetime, timezone

    from smart_city.simulator.data_gen import (
        _crowd_count,
        _density_score,
        ZoneSpec,
    )

    # Path to the JSON stats file written by the pipeline.
    # Matches the Docker volume mount point and the pipeline container's default.
    stats_file = os.environ.get("PIPELINE_STATS_FILE", "/pipeline_stats/pipeline_stats.json")

    # Load static per-stream location / threshold metadata.
    _meta_path = os.environ.get(
        "STREAMS_METADATA_FILE",
        str(Path(__file__).resolve().parents[1] / "config" / "streams_metadata.yaml"),
    )
    stream_meta_cfg: dict[int, dict] = _load_streams_metadata(_meta_path)
    _meta_reload_counter = 0  # reload metadata every 30 iterations (~60s)

    # Fallback simulator zones for when the pipeline is not running.
    fallback_zones = [
        ZoneSpec(
            stream_id=z.stream_id,
            zone_id=z.zone_id,
            zone_name=z.zone_name,
            zone_type="crowd_density",
            threshold_critical=z.threshold_critical,
            location_name=z.location_name,
            lat=z.lat,
            lon=z.lon,
            peak_capacity=z.peak_capacity,
        )
        for z in _DEFAULT_ZONES
    ]
    rng = random.Random()

    while True:
        try:
            now = datetime.now(tz=timezone.utc)
            ws_hub = getattr(app.state, "ws_hub", None)
            alert_engine = getattr(app.state, "alert_engine", None)
            event_persistence = getattr(app.state, "event_persistence", None)

            # Reload metadata every 30 iterations so threshold changes take effect
            # without restarting the API container.
            _meta_reload_counter += 1
            if _meta_reload_counter >= 30:
                stream_meta_cfg = _load_streams_metadata(_meta_path)
                _meta_reload_counter = 0

            # Read live stats from the pipeline stats file.
            pipeline_stats = _read_pipeline_stats(stats_file)
            using_live_data = bool(pipeline_stats)

            zones_data: dict = {}    # stream_id -> list of zone dicts
            stream_meta: dict = {}   # stream_id -> flat snapshot

            if using_live_data:
                # ----------------------------------------------------------------
                # LIVE MODE — use real pipeline counts, fps, and latency
                # ----------------------------------------------------------------
                for stream_id_int, stats in pipeline_stats.items():
                    # Pipeline uses 0-indexed stream IDs internally; metadata config
                    # uses 1-indexed IDs (cam1..cam50). Map 0→1, 1→2, …, 49→50.
                    sid = stream_id_int + 1
                    cfg = stream_meta_cfg.get(sid, {})
                    location_name = cfg.get("location_name", f"Camera {sid}")
                    lat = float(cfg.get("lat", 30.2672))
                    lon = float(cfg.get("lon", -97.7431))
                    city = cfg.get("city", "Unknown")
                    threshold = int(cfg.get("threshold_critical", 200))

                    count = int(stats.get("count", 0))
                    density_count_raw = stats.get("density_count")
                    density_count = (
                        float(density_count_raw)
                        if density_count_raw is not None
                        else None
                    )
                    det_count = int(stats.get("det_count", count))
                    fps = float(stats.get("fps", 0.0))
                    latency_ms = float(stats.get("latency_ms", 0.0))
                    status = str(stats.get("status", "LIVE"))
                    paired_published = bool(stats.get("paired_published", False))

                    density = min(1.0, det_count / max(threshold, 1))
                    severity = "CRITICAL" if det_count >= threshold else "SAFE"
                    # Use zone_id from metadata if present; fall back to
                    # cam{sid}_main so live data matches the seeded history.
                    zone_id = cfg.get("zone_id") or f"cam{sid}_main"
                    zone_type = cfg.get("zone_type", "crowd_density")
                    zone_name = location_name

                    # Accumulate zone data
                    if sid not in zones_data:
                        zones_data[sid] = []
                    zones_data[sid].append({
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "zone_type": zone_type,
                        "person_count": det_count,
                        "density_score": round(density, 3),
                        "severity": severity,
                        "violation_type": (
                            _VIOLATION_TYPE_MAP.get(zone_type, "crowd_density")
                            if severity == "CRITICAL"
                            else None
                        ),
                        "city": city,
                    })

                    stream_meta[sid] = {
                        "location_name": location_name,
                        "lat": lat,
                        "lon": lon,
                        "city": city,
                        "zone_id": zone_id,
                        "zone_name": zone_name,
                        "zone_type": zone_type,
                        "person_count": det_count,
                        "density_score": round(density, 3),
                        "severity": severity,
                        "threshold": threshold,
                        "det_count": det_count,
                        "density_count": density_count,
                        "paired_published": paired_published,
                        "violation_type": (
                            _VIOLATION_TYPE_MAP.get(zone_type, "crowd_density")
                            if severity == "CRITICAL"
                            else None
                        ),
                        "status": status,
                        "fps": fps,
                        "latency_ms": latency_ms,
                    }

                    # Persist to DB
                    if event_persistence is not None:
                        dr = _DR()
                        dr.timestamp = now.timestamp()
                        dr.stream_id = sid
                        dr.zone_id = zone_id
                        dr.zone_name = zone_name
                        dr.person_count = count
                        dr.density_score = density
                        dr.severity = severity
                        # Extract short city name (e.g. "Austin" from "Austin Downtown, TX")
                        short_city = city.split(",")[0].strip() if city else ""
                        event_persistence.submit_count(
                            dr,
                            city=short_city,
                            lat=lat,
                            lon=lon,
                            fps=fps,
                        )

                    # Prometheus crowd density
                    CROWD_DENSITY_GAUGE.labels(
                        stream_id=str(sid), zone_id=zone_id
                    ).set(count)
                    STREAM_FPS.labels(stream_id=str(sid)).set(fps)

                    # WebSocket broadcast
                    if ws_hub is not None:
                        await ws_hub.broadcast_count_update(
                            stream_id=sid,
                            zone_id=zone_id,
                            zone_name=zone_name,
                            location_name=location_name,
                            lat=lat,
                            lon=lon,
                            count=det_count,
                            severity=severity,
                        )

                    # Alert engine
                    if severity == "CRITICAL" and alert_engine is not None:
                        from smart_city.analytics.crowd.density_analyzer import (  # noqa: PLC0415
                            CrowdDensityResult,
                        )
                        result = CrowdDensityResult(
                            stream_id=sid,
                            zone_id=zone_id,
                            zone_name=zone_name,
                            zone_type="crowd_density",
                            person_count=count,
                            density_score=density,
                            severity="CRITICAL",
                            timestamp=now.timestamp(),
                            violation_type="crowd_density",
                        )
                        alert_engine.evaluate(
                            result,
                            {"location_name": location_name, "lat": lat, "lon": lon,
                             "threshold": threshold},
                        )

            else:
                # ----------------------------------------------------------------
                # FALLBACK / SIMULATOR MODE — synthetic data until pipeline starts
                # ----------------------------------------------------------------
                for zone in fallback_zones:
                    count = _crowd_count(now, zone, rng)
                    density = _density_score(count, zone.peak_capacity)
                    severity = "CRITICAL" if count >= zone.threshold_critical else "SAFE"
                    violation_type = (
                        _VIOLATION_TYPE_MAP[zone.zone_type]
                        if severity == "CRITICAL"
                        else None
                    )
                    sid = zone.stream_id

                    if sid not in zones_data:
                        zones_data[sid] = []
                    zones_data[sid].append({
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "zone_type": zone.zone_type,
                        "person_count": count,
                        "density_score": density,
                        "severity": severity,
                        "violation_type": violation_type,
                    })

                    prev = stream_meta.get(sid, {})
                    if prev.get("severity") != "CRITICAL":
                        stream_meta[sid] = {
                            "location_name": zone.location_name,
                            "lat": zone.lat,
                            "lon": zone.lon,
                            "zone_id": zone.zone_id,
                            "zone_name": zone.zone_name,
                            "zone_type": zone.zone_type,
                            "person_count": count,
                            "density_score": density,
                            "severity": severity,
                            "threshold": zone.threshold_critical,
                            "violation_type": violation_type,
                            "status": "LIVE",
                        }

                    if event_persistence is not None:
                        dr = _DR()
                        dr.timestamp = now.timestamp()
                        dr.stream_id = zone.stream_id
                        dr.zone_id = zone.zone_id
                        dr.zone_name = zone.zone_name
                        dr.person_count = count
                        dr.density_score = density
                        dr.severity = severity
                        short_city = zone.city.split(",")[0].strip() if zone.city else ""
                        event_persistence.submit_count(
                            dr,
                            city=short_city,
                            lat=zone.lat,
                            lon=zone.lon,
                        )

                    CROWD_DENSITY_GAUGE.labels(
                        stream_id=str(zone.stream_id), zone_id=zone.zone_id
                    ).set(count)

                    if ws_hub is not None:
                        await ws_hub.broadcast_count_update(
                            stream_id=zone.stream_id,
                            zone_id=zone.zone_id,
                            zone_name=zone.zone_name,
                            location_name=zone.location_name,
                            lat=zone.lat,
                            lon=zone.lon,
                            count=count,
                            severity=severity,
                        )

                    if severity == "CRITICAL" and alert_engine is not None:
                        from smart_city.analytics.crowd.density_analyzer import (  # noqa: PLC0415
                            CrowdDensityResult,
                        )
                        result = CrowdDensityResult(
                            stream_id=zone.stream_id,
                            zone_id=zone.zone_id,
                            zone_name=zone.zone_name,
                            zone_type=zone.zone_type,
                            person_count=count,
                            density_score=density,
                            severity="CRITICAL",
                            timestamp=now.timestamp(),
                            violation_type=violation_type or "crowd_density",
                        )
                        alert_engine.evaluate(
                            result,
                            {"location_name": zone.location_name, "lat": zone.lat,
                             "lon": zone.lon, "threshold": zone.threshold_critical},
                        )

                for sid in zones_data:
                    STREAM_FPS.labels(stream_id=str(sid)).set(0)
                    if sid in stream_meta:
                        stream_meta[sid]["fps"] = 0
                        stream_meta[sid]["latency_ms"] = 0

            # Prometheus totals
            total_count = sum(
                z["person_count"]
                for zone_list in zones_data.values()
                for z in zone_list
            )
            TOTAL_CROWD_COUNT.set(total_count)
            ACTIVE_STREAMS.set(len(zones_data))

            # Flush to density_cache
            for sid, meta in stream_meta.items():
                meta["zones"] = zones_data.get(sid, [])
                app.state.density_cache[sid] = meta

        except (
            asyncpg.PostgresError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
        ) as exc:
            logger.error(
                "Real-time push loop error: %s", exc, exc_info=True
            )

        await asyncio.sleep(2)


def _generate_density_heatmap(person_count: int, density_score: float,
                               width: int = 640, height: int = 360) -> str:
    """Generate a synthetic crowd density heatmap from aggregate count/density.

    The pipeline stats file only contains per-stream aggregate counts — bounding
    box positions are not written to pipeline_stats.json.  This function creates
    a representative heatmap by placing seeded Gaussian blobs scaled to the
    density score, suitable for the dashboard heatmap visualisation.

    Args:
        person_count: Current person count for the stream.
        density_score: Density score in [0, 1].
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        Base64-encoded JPEG string, or "" on error.
    """
    import base64

    import cv2
    import numpy as np

    try:
        rng = np.random.default_rng(person_count % 1000)
        canvas = np.zeros((height, width), dtype=np.float32)

        n_blobs = min(person_count, 120)
        if n_blobs > 0:
            n_clusters = max(1, min(3, person_count // 30))
            cluster_centres = rng.uniform(0.15, 0.85, size=(n_clusters, 2))
            for i in range(n_blobs):
                cx_f, cy_f = cluster_centres[i % n_clusters]
                cx = int(cx_f * width + rng.normal(0, width * 0.08))
                cy = int(cy_f * height + rng.normal(0, height * 0.08))
                cx = int(np.clip(cx, 0, width - 1))
                cy = int(np.clip(cy, 0, height - 1))
                canvas[cy, cx] += 1.0

        sigma = max(15, int(width * 0.04))
        smoothed = cv2.GaussianBlur(canvas, (0, 0), sigma)
        norm = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
        coloured = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".jpg", coloured, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except (cv2.error, ValueError, RuntimeError, OSError) as exc:
        logger.debug(
            "Heatmap generation error: %s", exc, exc_info=True
        )
        return ""


async def _heatmap_update_loop(app: FastAPI) -> None:
    """Regenerate per-stream density heatmaps every 10 s from density_cache.

    Runs heatmap generation in a thread executor to avoid blocking the event
    loop on numpy/cv2 operations.  Writes results to heatmap_cache and
    broadcasts to any active WebSocket subscribers on the heatmap:{stream_id}
    channel.
    """
    import asyncio as _asyncio

    while True:
        await _asyncio.sleep(10)
        try:
            density_cache: dict = getattr(app.state, "density_cache", {})
            heatmap_cache: dict = getattr(app.state, "heatmap_cache", {})
            ws_hub = getattr(app.state, "ws_hub", None)

            loop = _asyncio.get_event_loop()
            for stream_id, data in list(density_cache.items()):
                count = int(data.get("person_count", 0))
                density = float(data.get("density_score", 0.0))
                b64 = await loop.run_in_executor(
                    None, _generate_density_heatmap, count, density
                )
                heatmap_cache[stream_id] = b64
                if ws_hub is not None:
                    await ws_hub.broadcast_heatmap(stream_id, b64)
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            logger.error(
                "Heatmap update loop error: %s", exc, exc_info=True
            )


class StreamRegistry:
    """Proxy that handles dynamic stream registration via the REST API.

    The combined_pipeline runs as a separate container and reads its stream
    config at startup — it does not support runtime stream addition via IPC.
    This registry immediately registers new streams into density_cache so they
    are visible in API responses.  The pipeline will pick up new streams on its
    next restart (or via a config update).

    Args:
        app: The FastAPI application instance whose state holds density_cache.
    """

    def __init__(self, app: FastAPI) -> None:
        """Initialise with a reference to the FastAPI app."""
        self._app = app

    def add_stream(self, body: Any) -> None:
        """Register a new stream into density_cache immediately.

        Args:
            body: StreamCreate model with id, location_name, lat, lon, gpu_id.

        Raises:
            ValueError: If a stream with the same ID is already registered.
        """
        cache: dict = getattr(self._app.state, "density_cache", {})
        if body.id in cache:
            raise ValueError(f"Stream {body.id} is already registered.")
        cache[body.id] = {
            "location_name": body.location_name,
            "lat": float(getattr(body, "lat", 0.0)),
            "lon": float(getattr(body, "lon", 0.0)),
            "city": None,
            "zone_id": f"cam{body.id}_main",
            "zone_name": body.location_name,
            "zone_type": "crowd_density",
            "person_count": 0,
            "density_score": 0.0,
            "severity": "SAFE",
            "threshold": 200,
            "violation_type": None,
            "status": "STARTING",
            "fps": 0.0,
            "latency_ms": 0.0,
        }
        logger.info(
            "StreamRegistry: registered stream %d ('%s'). "
            "Pipeline will process it on next restart.",
            body.id, body.location_name,
        )


async def init_app_state(app: FastAPI, config: Any) -> None:
    """Initialise all platform subsystems and attach to ``app.state``.

    Called once during the FastAPI lifespan startup hook.

    Args:
        app: The FastAPI application instance.
        config: Loaded ``AppSettings`` pydantic model.
    """
    app.state.config = config

    # ------------------------------------------------------------------
    # 1. TimescaleDB connection pool
    # ------------------------------------------------------------------
    db_pool = None
    try:
        import asyncpg  # pylint: disable=import-outside-toplevel

        db_pool = await asyncpg.create_pool(
            dsn=config.storage.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        app.state.db_pool = db_pool
        # Expose a single-connection alias expected by routes/simulator.py
        app.state.db = db_pool
        logger.info("TimescaleDB pool established.")

        # Apply DDL schema idempotently on every startup
        from smart_city.storage.schemas import DDL  # pylint: disable=import-outside-toplevel

        async with db_pool.acquire() as _conn:
            for _stmt in DDL.split(";"):
                _s = _stmt.strip()
                if _s:
                    try:
                        await _conn.execute(_s)
                    except asyncpg.PostgresError as _ddl_exc:
                        logger.debug(
                            "DDL skipped: %s", _ddl_exc, exc_info=True
                        )
        logger.info("Schema initialisation complete.")
    except (
        asyncpg.PostgresError,
        OSError,
        ValueError,
        ImportError,
        RuntimeError,
    ) as exc:
        logger.error(
            "TimescaleDB pool failed: %s", exc, exc_info=True
        )
        app.state.db_pool = None
        app.state.db = None

    # ------------------------------------------------------------------
    # 1b. Simulator mode check + auto-seed (non-fatal)
    # ------------------------------------------------------------------
    retire_after = int(
        getattr(
            getattr(config, "simulator", None),
            "retire_after_days",
            7,
        )
    )
    app.state.sim_retire_after_days = retire_after
    if db_pool is not None:
        try:
            from smart_city.simulator.mode_manager import (  # pylint: disable=import-outside-toplevel
                ModeManager,
            )

            mgr = ModeManager(db_pool, retire_after_days=retire_after)
            status = await mgr.check_and_update()
            logger.info(
                "Simulator mode on startup: %s "
                "(%.1f / %d real-data days)",
                status.mode.value,
                status.real_days_covered,
                status.retire_after_days,
            )
        except (
            asyncpg.PostgresError,
            ImportError,
            ValueError,
            RuntimeError,
            OSError,
        ) as exc:
            logger.warning(
                "Simulator mode check failed (non-fatal): %s",
                exc,
                exc_info=True,
            )

        async def _auto_seed_task() -> None:
            """Run the 2-month historical seed once on startup in background.

            Waits for the DB pool to be healthy, then delegates to the same
            seed logic used by the manual simulator CLI so the dashboard has
            data immediately after deployment.
            """
            await asyncio.sleep(5)  # brief warm-up for the pool
            try:
                from smart_city.simulator.runner import (  # pylint: disable=import-outside-toplevel
                    _run_seed,
                )

                logger.info("Auto-seed: starting 2-month historical seed…")
                await _run_seed(db_pool, retire_after)
                logger.info("Auto-seed: complete.")
            except (
                asyncpg.PostgresError,
                ImportError,
                ValueError,
                RuntimeError,
                OSError,
            ) as exc:
                logger.warning(
                    "Auto-seed failed (non-fatal): %s",
                    exc,
                    exc_info=True,
                )

        asyncio.create_task(_auto_seed_task())
        logger.info("Auto-seed background task scheduled.")

    # ------------------------------------------------------------------
    # 2. WebSocket hub
    # ------------------------------------------------------------------
    from smart_city.api.websocket import (  # pylint: disable=import-outside-toplevel
        WebSocketHub,
    )

    ws_hub = WebSocketHub()
    app.state.ws_hub = ws_hub

    # ------------------------------------------------------------------
    # 3. LLM / RAG (Embedder + Milvus + vLLM)
    # ------------------------------------------------------------------
    # Ingestion is blocking: the application does not accept traffic
    # until all documents are embedded and stored in Milvus.  Milvus /
    # embedder service readiness is retried for up to 120 s so that
    # Docker-Compose dependency ordering is respected gracefully.
    from smart_city.llm.doc_ingestor import (  # pylint: disable=import-outside-toplevel
        ingest_documents,
    )
    from smart_city.llm.embedder import (  # pylint: disable=import-outside-toplevel
        Embedder,
    )
    from smart_city.llm.vector_store import (  # pylint: disable=import-outside-toplevel
        VectorStoreClient,
    )

    milvus_host = config.rag.milvus_host
    milvus_port = config.rag.milvus_port
    embed_base_url = getattr(config.rag, "embed_base_url", "") or None

    # --- Wait for Milvus ---
    _rag_ready_timeout = 120
    _rag_retry_interval = 5
    _rag_elapsed = 0
    vector_store: Optional[VectorStoreClient] = None
    while _rag_elapsed < _rag_ready_timeout:
        try:
            _vs = VectorStoreClient(host=milvus_host, port=milvus_port)
            _vs.connect()
            vector_store = _vs
            logger.info("Milvus connected after %ds.", _rag_elapsed)
            break
        except (
            MilvusException,
            ImportError,
            OSError,
            RuntimeError,
        ) as _milvus_exc:
            logger.warning(
                "Milvus not ready (%s); retrying in %ds… (%d/%d)",
                _milvus_exc, _rag_retry_interval,
                _rag_elapsed, _rag_ready_timeout,
            )
            await asyncio.sleep(_rag_retry_interval)
            _rag_elapsed += _rag_retry_interval

    if vector_store is None:
        raise RuntimeError(
            f"Milvus not reachable after {_rag_ready_timeout}s — aborting startup."
        )

    # --- Wait for embedder service ---
    embedder: Optional[Embedder] = None
    if embed_base_url:
        _emb_elapsed = 0
        while _emb_elapsed < _rag_ready_timeout:
            try:
                _r = await asyncio.to_thread(
                    httpx.get,
                    f"{embed_base_url.rstrip('/')}/health",
                    timeout=5.0,
                )
                _r.raise_for_status()
                logger.info(
                    "Embedder service ready after %ds.", _emb_elapsed
                )
                break
            except (httpx.HTTPError, OSError) as _emb_exc:
                logger.warning(
                    "Embedder service not ready (%s); retrying in %ds… (%d/%d)",
                    _emb_exc, _rag_retry_interval,
                    _emb_elapsed, _rag_ready_timeout,
                )
                await asyncio.sleep(_rag_retry_interval)
                _emb_elapsed += _rag_retry_interval
        else:
            raise RuntimeError(
                f"Embedder service not reachable after {_rag_ready_timeout}s"
                " — aborting startup."
            )

    embedder = Embedder(
        model_name=getattr(
            config.rag, "embedding_model", "BAAI/bge-small-en-v1.5"
        ),
        base_url=embed_base_url,
    )

    # --- Blocking document ingestion (no try/except — must complete) ---
    docs_path = getattr(config.rag, "docs_ingest_path", "/app/data/docs")
    # Only index PDF/MD — plain-text duplicates produce NaN embeddings and
    # inflate the collection with redundant chunks on every restart.
    docs_glob = getattr(config.rag, "docs_glob", "**/*.md,**/*.pdf")
    chunk_size = getattr(config.rag, "chunk_size", 500)
    chunk_overlap = getattr(config.rag, "chunk_overlap", 50)

    # Drop and recreate the collection so each deployment starts from a clean
    # slate — prevents duplicate chunks accumulating across restarts.
    vector_store.drop_and_recreate()

    logger.info("RAG document ingestion starting (blocking startup)…")
    _n_chunks = await ingest_documents(
        docs_ingest_path=docs_path,
        docs_glob=docs_glob,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("RAG document ingestion complete: %d chunks indexed.", _n_chunks)

    # --- Wire ReportGenerator (non-fatal: vLLM is optional) ---
    try:
        from smart_city.llm.report_generator import (  # pylint: disable=import-outside-toplevel
            ReportGenerator,
        )

        vllm_url = config.llm.base_url
        model_name = getattr(config.llm, "model", "qwen")
        api_key = getattr(config.llm, "api_key", "")
        max_tokens = getattr(config.llm, "max_tokens", 1024)

        app.state.report_generator = ReportGenerator(
            vllm_base_url=vllm_url,
            model_name=model_name,
            embedder=embedder,
            vector_store=vector_store,
            db_pool=db_pool,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        logger.info("LLM / RAG subsystem initialised.")
    except (
        ImportError,
        ValueError,
        AttributeError,
        RuntimeError,
        OSError,
    ) as exc:
        logger.error(
            "ReportGenerator init failed (non-fatal): %s", exc, exc_info=True
        )
        app.state.report_generator = None

    # ------------------------------------------------------------------
    # 5. EventPersistence (batch DB writer)
    # ------------------------------------------------------------------
    event_persistence = None
    if db_pool is not None:
        from smart_city.storage.event_persistence import (  # pylint: disable=import-outside-toplevel
            EventPersistence,
        )

        event_persistence = EventPersistence(db_pool)
        asyncio.create_task(event_persistence.run())
        logger.info("EventPersistence batch writer started.")
    app.state.event_persistence = event_persistence

    # ------------------------------------------------------------------
    # 6. ThresholdAlertEngine wired to WebSocket hub + DB writer
    # ------------------------------------------------------------------
    from smart_city.analytics.alerts.threshold_engine import (  # pylint: disable=import-outside-toplevel
        ThresholdAlertEngine,
    )

    alert_engine = ThresholdAlertEngine(cooldown_seconds=60.0)

    async def _on_alert(alert) -> None:
        ALERTS_FIRED.labels(
            stream_id=str(alert.stream_id),
            zone_id=alert.zone_id,
            severity=alert.severity,
            violation_type=alert.violation_type,
        ).inc()
        if ws_hub is not None:
            await ws_hub.broadcast_alert(alert)
        if event_persistence is not None:
            event_persistence.submit_alert(alert)

    alert_engine.register_callback(_on_alert)
    app.state.alert_engine = alert_engine
    logger.info("ThresholdAlertEngine initialised.")

    # ------------------------------------------------------------------
    # 6b. HistoricalPatternAnalyzer — runs on startup then every 6 h
    # ------------------------------------------------------------------
    if db_pool is not None:
        from smart_city.analytics.crowd.pattern_analyzer import (  # pylint: disable=import-outside-toplevel
            HistoricalPatternAnalyzer,
        )

        pattern_analyzer = HistoricalPatternAnalyzer(db_pool)
        app.state.pattern_analyzer = pattern_analyzer

        async def _pattern_analysis_loop() -> None:
            """Detect recurring patterns from historical data; run at startup
            (after 60 s warm-up) then every 6 hours."""
            await asyncio.sleep(60)
            while True:
                try:
                    patterns = await pattern_analyzer.analyze_weekly_patterns(
                        lookback_days=90
                    )
                    await pattern_analyzer.upsert_patterns(patterns)
                    logger.info(
                        "Pattern analysis complete: %d patterns detected.",
                        len(patterns),
                    )
                except (
                    asyncpg.PostgresError,
                    OSError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    logger.error(
                        "Pattern analysis error: %s", exc, exc_info=True
                    )
                await asyncio.sleep(6 * 3600)

        asyncio.create_task(_pattern_analysis_loop())
        logger.info("Pattern analysis background task started.")

    # ------------------------------------------------------------------
    # 7. In-memory caches + real-time push loop
    # ------------------------------------------------------------------
    app.state.density_cache = {}   # stream_id -> density data
    app.state.heatmap_cache = {}   # stream_id -> b64 heatmap
    app.state.report_jobs = {}     # report_id -> job state

    # Pipeline proxy — handles POST /streams without requiring the combined_pipeline
    # container to be running.  Streams added via the API are immediately visible in
    # density_cache; the pipeline picks them up on its next restart.
    app.state.pipeline = StreamRegistry(app)

    asyncio.create_task(_realtime_push_loop(app))
    logger.info("Real-time push loop started.")

    asyncio.create_task(_heatmap_update_loop(app))
    logger.info("Heatmap update loop started.")

    # ------------------------------------------------------------------
    # 8. WebSocket keepalive ping task
    # ------------------------------------------------------------------
    async def _ws_ping_loop() -> None:
        while True:
            try:
                await ws_hub.ping_all()
            except (RuntimeError, OSError, ValueError) as _e:
                logger.debug("WS ping error: %s", _e, exc_info=True)
            await asyncio.sleep(30)

    asyncio.create_task(_ws_ping_loop())
    logger.info("WebSocket ping task started.")

    logger.info("App state initialised successfully.")


async def shutdown_app_state(app: FastAPI) -> None:
    """Gracefully shut down all platform subsystems.

    Args:
        app: The FastAPI application instance.
    """
    # EventPersistence — flush remaining buffered events
    event_persistence = getattr(app.state, "event_persistence", None)
    if event_persistence:
        event_persistence.stop()
        try:
            await event_persistence.flush()
        except (asyncpg.PostgresError, OSError, RuntimeError) as exc:
            logger.warning(
                "EventPersistence flush failed during shutdown: %s",
                exc,
                exc_info=True,
            )

    # Milvus
    try:
        qa = getattr(app.state, "conversational_qa", None)
        if qa and hasattr(qa, "_vector_store"):
            qa._vector_store.disconnect()
    except (MilvusException, RuntimeError, OSError, AttributeError) as exc:
        logger.debug(
            "Milvus disconnect during shutdown failed: %s",
            exc,
            exc_info=True,
        )

    # DB pool
    db_pool = getattr(app.state, "db_pool", None)
    if db_pool:
        await db_pool.close()
        logger.info("TimescaleDB pool closed.")

    logger.info("App state shut down.")

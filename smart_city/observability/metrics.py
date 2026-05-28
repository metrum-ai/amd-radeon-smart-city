# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Prometheus metrics registry for the Smart City platform."""

import logging

logger = logging.getLogger(__name__)

# All metrics are module-level singletons so they are registered once
# and shared across the application process.

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
    )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    FRAMES_PROCESSED = Counter(
        "smartcity_frames_processed_total",
        "Total video frames processed by the detection pipeline.",
        ["stream_id"],
    )

    FRAME_PROCESSING_LATENCY = Histogram(
        "smartcity_frame_processing_latency_seconds",
        "End-to-end latency per frame from ingestion to analytics.",
        ["stream_id"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------
    DETECTIONS_TOTAL = Counter(
        "smartcity_detections_total",
        "Total objects detected by YOLO.",
        ["stream_id", "class_name"],
    )

    # ------------------------------------------------------------------
    # Crowd density
    # ------------------------------------------------------------------
    CROWD_DENSITY_GAUGE = Gauge(
        "smartcity_crowd_density_persons",
        "Current person count in a zone.",
        ["stream_id", "zone_id"],
    )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    ALERTS_FIRED = Counter(
        "smartcity_alerts_fired_total",
        "Total crowd alerts fired.",
        ["stream_id", "zone_id", "severity", "violation_type"],
    )

    ALERTS_ACKNOWLEDGED = Counter(
        "smartcity_alerts_acknowledged_total",
        "Total crowd alerts acknowledged by operators.",
        ["stream_id", "zone_id"],
    )

    # ------------------------------------------------------------------
    # HTTP / WebSocket
    # ------------------------------------------------------------------
    HTTP_REQUEST_DURATION = Histogram(
        "smartcity_http_request_duration_seconds",
        "FastAPI HTTP request duration in seconds.",
        ["method", "path", "status_code"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    WEBSOCKET_CONNECTIONS = Gauge(
        "smartcity_websocket_connections_active",
        "Number of active WebSocket connections.",
        ["channel"],
    )

    # ------------------------------------------------------------------
    # Stream / pipeline throughput
    # ------------------------------------------------------------------
    ACTIVE_STREAMS = Gauge(
        "smartcity_active_streams",
        "Number of active camera streams currently producing data.",
    )

    STREAM_FPS = Gauge(
        "smartcity_stream_fps",
        "Frames per second for a camera stream.",
        ["stream_id"],
    )

    TOTAL_CROWD_COUNT = Gauge(
        "smartcity_total_crowd_count",
        "Sum of person counts across all active zones.",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    DB_WRITE_LATENCY = Histogram(
        "smartcity_db_write_latency_seconds",
        "TimescaleDB write latency in seconds.",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    )

    MILVUS_INSERT_LATENCY = Histogram(
        "smartcity_milvus_insert_latency_seconds",
        "Milvus vector insert latency in seconds.",
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    )

    _METRICS_AVAILABLE = True
    logger.info("Prometheus metrics registered.")

except ImportError:
    logger.warning(
        "prometheus_client not installed; metrics disabled."
    )
    _METRICS_AVAILABLE = False

    # Define no-op stubs so import never fails
    class _Noop:  # pylint: disable=too-few-public-methods
        """No-op stub for Prometheus metric objects."""

        def labels(self, **_kwargs):
            """Return self for chaining."""
            return self

        def inc(self, _amount=1):
            """No-op increment."""

        def observe(self, _value):
            """No-op observe."""

        def set(self, _value):
            """No-op set."""

        def time(self):
            """No-op context manager."""
            import contextlib

            return contextlib.nullcontext()

    FRAMES_PROCESSED = _Noop()
    FRAME_PROCESSING_LATENCY = _Noop()
    DETECTIONS_TOTAL = _Noop()
    CROWD_DENSITY_GAUGE = _Noop()
    ALERTS_FIRED = _Noop()
    ALERTS_ACKNOWLEDGED = _Noop()
    HTTP_REQUEST_DURATION = _Noop()
    WEBSOCKET_CONNECTIONS = _Noop()
    ACTIVE_STREAMS = _Noop()
    STREAM_FPS = _Noop()
    TOTAL_CROWD_COUNT = _Noop()
    DB_WRITE_LATENCY = _Noop()
    MILVUS_INSERT_LATENCY = _Noop()

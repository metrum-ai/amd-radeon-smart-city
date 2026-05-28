# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Startup for the combined pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the pipeline configuration."""
    if path is None:
        path = os.environ.get("PIPELINE_CONFIG") or (
            Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml"
        )
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {p}")
    return _apply_env_overrides(data)


# ---------------------------------------------------------------------------
# Deployment-knob defaults
# ---------------------------------------------------------------------------
# These are values that legitimately vary per deployment (stream count, GPU
# routing, RTSP host) and are therefore env-driven, not YAML-driven. They used
# to be duplicated in pipeline.yaml; that caused drift between .env.example
# and pipeline.yaml. Now the resolution order is strictly:
#
#     env var  >  YAML (if present, kept as an escape hatch for standalone
#                       runs that ship a custom pipeline.yaml)
#              >  built-in default (this table)
#
# Perf-tuned constants (batch sizes, MIGraphX options, infer resolutions,
# model paths) STILL live in pipeline.yaml — those are benchmarked artifacts,
# not deployment knobs, and don't belong in .env.
# ---------------------------------------------------------------------------
_DEPLOYMENT_DEFAULTS: dict[tuple[str, str], Any] = {
    ("streams", "count"):            50,
    ("streams", "input_base_rtsp"):  "rtsp://mediamtx:8554/cam",
    ("density", "display_streams"):  50,
    ("density", "physical_gpu"):     1,
    ("density", "frame_interval"):   1,
    ("mediamtx", "host"):            "mediamtx",
}


def _resolve(
    cfg: dict[str, Any],
    section: str,
    key: str,
    env_var: str,
    cast: Any = str,
) -> None:
    """Populate cfg[section][key] using env > yaml > built-in default."""
    sect = cfg.setdefault(section, {})
    raw = os.environ.get(env_var)
    if raw is not None and raw != "":
        sect[key] = cast(raw)
        return
    if key in sect:
        return  # YAML already supplied a value
    sect[key] = _DEPLOYMENT_DEFAULTS[(section, key)]


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Environment variable overrides for Docker/container deployments.

    Deployment knobs (env-authoritative, with built-in defaults) are resolved
    first. Perf-tuned overrides (env as escape hatch only — no built-in
    default; YAML is the source of truth) come second.
    """
    # --- Deployment knobs ---------------------------------------------------
    stream_count = os.environ.get("STREAM_COUNT") or os.environ.get("NUM_STREAMS")
    if stream_count:
        cfg.setdefault("streams", {})["count"] = int(stream_count)
    elif "count" not in cfg.get("streams", {}):
        cfg.setdefault("streams", {})["count"] = _DEPLOYMENT_DEFAULTS[("streams", "count")]

    _resolve(cfg, "streams",  "input_base_rtsp", "INPUT_BASE_RTSP")
    _resolve(cfg, "mediamtx", "host",            "MEDIAMTX_HOST")
    _resolve(cfg, "density",  "display_streams", "DENSITY_DISPLAY_STREAMS", cast=int)
    _resolve(cfg, "density",  "physical_gpu",    "DENSITY_PHYSICAL_GPU",    cast=int)
    _resolve(cfg, "density",  "frame_interval",  "DENSITY_EVERY_N_FRAMES", cast=int)

    # --- Perf-tuned escape hatches (env overrides YAML; no built-in default
    #     because YAML is the canonical source for these) -------------------
    if v := os.environ.get("GPU_COUNT"):
        cfg.setdefault("streams", {})["gpu_count"] = int(v)
    if v := os.environ.get("YOLO_MODEL_PATH"):
        cfg.setdefault("yolo", {})["model_path"] = v
    if v := os.environ.get("OUTPUT_BASE_RTSP"):
        cfg.setdefault("_env", {})["output_base_rtsp"] = v
    if v := os.environ.get("DECODE_MODE"):
        cfg.setdefault("ingest", {})["decode_mode"] = v
    if v := os.environ.get("HW_DECODE_VAAPI_GPU"):
        cfg.setdefault("ingest", {})["hw_decode_vaapi_gpu"] = int(v)
    if v := os.environ.get("HW_DECODE_MAX_STREAMS"):
        cfg.setdefault("ingest", {})["hw_decode_max_streams"] = int(v)
    if v := os.environ.get("HW_DECODE_STAGGER_MS"):
        cfg.setdefault("ingest", {})["hw_decode_stagger_ms"] = int(v)
    if v := os.environ.get("ENCODE_MODE"):
        cfg.setdefault("output", {})["encode_mode"] = v
    if v := os.environ.get("YOLO_REPLICAS_PER_GPU"):
        cfg.setdefault("yolo", {})["replicas_per_gpu"] = int(v)
    if v := os.environ.get("YOLO_BATCH_SIZE"):
        cfg.setdefault("yolo", {})["batch_size"] = int(v)
    if v := os.environ.get("YOLO_BACKEND"):
        cfg.setdefault("yolo", {})["backend"] = v
    if v := os.environ.get("YOLO_START_GPU"):
        cfg.setdefault("yolo", {})["start_gpu_id"] = int(v)
    if v := os.environ.get("DENSITY_ENABLED"):
        cfg.setdefault("density", {})["enabled"] = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("SHMEM_SLOTS_PER_STREAM"):
        cfg.setdefault("shmem", {})["slots_per_stream"] = int(v)
    if v := os.environ.get("PAIRED_OUTPUT_FPS"):
        cfg.setdefault("output", {})["fps"] = int(v)
    return cfg


def output_rtsp_url(base: str, stream_index_one_based: int, suffix: str) -> str:
    """Generate the output RTSP URL."""
    return f"{base}{stream_index_one_based}{suffix}"

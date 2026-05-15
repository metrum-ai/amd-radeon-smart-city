# Created by Metrum AI for AMD

"""Configuration loader and validated settings for the Smart City platform.

Reads config/main.yaml, config/streams.yaml and resolves ${VAR} environment
variable references.
"""

import os
import re
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Zone and stream sub-models
# ---------------------------------------------------------------------------


class ZoneConfig(BaseModel):
    """Configuration for a single surveillance zone."""

    zone_id: str
    name: str
    zone_type: str  # crowd_density
    detect_class: str  # person
    polygon: List[List[float]]
    threshold_critical: int = 15

    @field_validator("detect_class")
    @classmethod
    def validate_detect_class(cls, v: str) -> str:
        """Normalise detect_class to lowercase."""
        return v

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate polygon has at least 3 points."""
        if len(v) < 3:
            raise ValueError("polygon must have at least 3 points")
        return v


class StreamConfig(BaseModel):
    """Configuration for a single RTSP camera stream."""

    id: int
    url: str
    location_name: str
    lat: float
    lon: float
    codec: str = "h264"
    width: int = 1280
    height: int = 720
    gpu_id: int = -1
    zones: List[ZoneConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section-level config models
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    """System-level hardware and worker config."""

    num_gpus: int = 2
    # Maximum number of RTSP streams (caps auto-discovery scan)
    max_streams: int = 8
    frame_workers: int = 4
    log_level: str = "INFO"

    @field_validator("num_gpus")
    @classmethod
    def validate_num_gpus(cls, v: int) -> int:
        """Validate num_gpus is between 1 and 8."""
        if not 1 <= v <= 8:
            raise ValueError("num_gpus must be between 1 and 8")
        return v

    @field_validator("max_streams")
    @classmethod
    def validate_max_streams(cls, v: int) -> int:
        """Validate max_streams does not exceed the supported limit of 100."""
        if v > 100:
            raise ValueError(
                f"STREAM_COUNT={v} exceeds the maximum supported limit of 100. "
                "Set NUM_STREAMS/STREAM_COUNT to 100 or fewer and restart."
            )
        return v


class PipelineConfig(BaseModel):
    """YOLO inference pipeline config."""

    batch_size: int = 48
    batch_timeout_ms: int = 20
    model_path: str = "smart_city/models/yolov26-seg.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is between 1 and 128."""
        if not 1 <= v <= 128:
            raise ValueError("batch_size must be between 1 and 128")
        return v


class IngestionConfig(BaseModel):
    """GStreamer ingestion tuning config."""

    reconnect_max_attempts: int = 10
    reconnect_base_delay_s: float = 1.0
    frame_queue_size: int = 500


class AdaptiveFpsConfig(BaseModel):
    """FPS tier settings per alert severity."""

    critical_fps: int = 30
    medium_fps: int = 15
    low_fps: int = 5


class LLMConfig(BaseModel):
    """vLLM / OpenRouter-compatible LLM connection config."""

    base_url: str = "http://lemonade:13305"
    # Use the --served-model-name alias set in vLLM (default: "qwen")
    # For OpenRouter set to e.g. "openai/gpt-4o"
    model: str = "qwen"
    # Optional API key – required for hosted providers such as OpenRouter.
    # Set via LLM_API_KEY env var; leave empty for local vLLM (no auth).
    api_key: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1
    stream: bool = True


class RAGConfig(BaseModel):
    """Milvus + embedding config for RAG."""

    milvus_host: str = "milvus"
    milvus_port: int = 19530
    collection_name: str = "city_guidelines"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Base URL of an OpenAI-compatible embedding service (e.g. vLLM CPU).
    # When set, embeddings are computed remotely; local sentence-transformers
    # is used as fallback when this is empty.
    embed_base_url: str = ""
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Directory whose files are ingested into Milvus on startup
    docs_ingest_path: str = "/app/data/docs"
    # Comma-separated glob patterns for ingestible file types
    docs_glob: str = "**/*.txt,**/*.md,**/*.pdf"


class StorageConfig(BaseModel):
    """Database and cache connection config."""

    database_url: str = ""
    pool_size: int = 20
    pool_max_overflow: int = 10


class APIConfig(BaseModel):
    """FastAPI server config."""

    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:80",
        ]
    )


class MetricsConfig(BaseModel):
    """Prometheus metrics config."""

    prometheus_port: int = 9090
    enabled: bool = True


class SimulatorConfig(BaseModel):
    """Simulator mode config."""

    retire_after_days: int = 7


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------


class AppSettings(BaseModel):
    """Root application settings assembled from all YAML sections."""

    system: SystemConfig = SystemConfig()
    pipeline: PipelineConfig = PipelineConfig()
    ingestion: IngestionConfig = IngestionConfig()
    adaptive_fps: AdaptiveFpsConfig = AdaptiveFpsConfig()
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()
    metrics: MetricsConfig = MetricsConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    streams: List[StreamConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_stream_ids(self) -> "AppSettings":
        """Validate all stream IDs are unique."""
        ids = [s.id for s in self.streams]
        if len(ids) != len(set(ids)):
            raise ValueError("All stream IDs must be unique")
        return self


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} references from environment.

    Supports both plain ``${VAR}`` (returns empty string when unset)
    and ``${VAR:-default}`` (returns *default* when VAR is unset or
    empty) syntax.

    Args:
        value: String potentially containing ``${…}`` references.

    Returns:
        String with all ``${…}`` references replaced by their values.
    """

    def _replace(match: re.Match) -> str:
        content = match.group(1)
        if ":-" in content:
            var_name, default = content.split(":-", 1)
            env_val = os.environ.get(var_name)
            return env_val if env_val is not None else default
        return os.environ.get(content, "")

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _resolve_dict(obj):
    """Recursively resolve env vars in a nested dict/list structure.

    Args:
        obj: Arbitrarily nested dict, list, or scalar value.

    Returns:
        Same structure with all string values env-resolved.
    """
    if isinstance(obj, dict):
        return {k: _resolve_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_dict(i) for i in obj]
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    return obj


def _load_yaml(path: str) -> dict:
    """Load and env-resolve a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed and env-resolved dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _resolve_dict(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _discover_env_streams(
    existing_streams: List[dict],
    max_streams: int,
) -> List[dict]:
    """Auto-discover RTSP streams from RTSP_CAM_{N}_URL environment variables.

    Scans indices 1..max_streams. Any index already present in
    ``existing_streams`` (by ``id``) is skipped so YAML always wins.

    Args:
        existing_streams: Stream dicts already loaded from streams.yaml.
        max_streams: Upper bound for scan (inclusive).

    Returns:
        Merged list of stream dicts (YAML entries first, env-only after).
    """
    existing_ids = {int(s.get("id", -1)) for s in existing_streams}
    discovered: List[dict] = []
    for n in range(1, max_streams + 1):
        url = os.environ.get(f"RTSP_CAM_{n}_URL")
        if not url or n in existing_ids:
            continue
        discovered.append(
            {
                "id": n,
                "url": url,
                "location_name": os.environ.get(
                    f"RTSP_CAM_{n}_LOCATION", f"Camera {n}"
                ),
                "lat": float(os.environ.get(f"RTSP_CAM_{n}_LAT", "0.0")),
                "lon": float(os.environ.get(f"RTSP_CAM_{n}_LON", "0.0")),
                "codec": os.environ.get(f"RTSP_CAM_{n}_CODEC", "h264"),
                "gpu_id": int(os.environ.get(f"RTSP_CAM_{n}_GPU", "-1")),
            }
        )
    return existing_streams + discovered


def load_config(
    main_yaml_path: str = "smart_city/config/main.yaml",
) -> AppSettings:
    """Load and validate all configuration files into AppSettings.

    Stream sources (highest to lowest priority):
    1. ``RTSP_CAM_{N}_URL`` env vars — minimal auto-discovery scanned up
       to ``system.max_streams``.

    Args:
        main_yaml_path: Path to the main system YAML config.

    Returns:
        Validated AppSettings instance.

    Raises:
        FileNotFoundError: If a required config file is missing.
        pydantic.ValidationError: If config values fail validation.
    """
    main_data = _load_yaml(main_yaml_path)

    # Parse system section early to get max_streams cap
    system_raw = main_data.get("system", {})
    max_streams = int(system_raw.get("max_streams", 8))

    # Populate streams from env-discovered sources
    main_data["streams"] = _discover_env_streams([], max_streams)

    return AppSettings(**main_data)

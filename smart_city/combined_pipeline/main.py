# Created by Metrum AI for AMD

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _export_dm_count_onnx(cfg: dict) -> None:
    """Export DM-Count VGG19 weights → ONNX before any GPU process starts.

    Runs on CPU only (no ROCm initialised). The ONNX file is then loaded
    by ORT + MIGraphXExecutionProvider in the density server process.
    """
    density_cfg = cfg.get("density", {})
    if not density_cfg.get("enabled", True):
        return

    weights = density_cfg.get("weights_path")
    if not weights or not Path(weights).exists():
        logger.warning("DM-Count weights not found at %s — density disabled", weights)
        return

    onnx_path = density_cfg.get("onnx_path", "/app/models/dm_count.onnx")
    batch_size = int(density_cfg.get("batch_size", 4))
    infer_w = int(density_cfg.get("infer_width", 320))
    infer_h = int(density_cfg.get("infer_height", 240))

    # Ensure CPU-only execution for the export step
    for key in ("ROCR_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES"):
        os.environ.pop(key, None)

    from combined_pipeline.inference.export_dm_count import export_dm_count_onnx
    ok = export_dm_count_onnx(weights, onnx_path, batch_size, infer_w, infer_h)
    if not ok:
        logger.error("DM-Count ONNX export failed — density will be disabled")
        cfg["density"]["enabled"] = False
    else:
        cfg["density"]["onnx_path"] = onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined YOLO + DM-Count pipeline with WebRTC output"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline YAML (default: combined_pipeline/config/pipeline.yaml)",
    )
    args = parser.parse_args()

    from combined_pipeline.control.startup import load_pipeline_config
    cfg = load_pipeline_config(args.config)

    _export_dm_count_onnx(cfg)

    from combined_pipeline.orchestrator.pipeline import run_pipeline
    run_pipeline(cfg=cfg)


if __name__ == "__main__":
    main()

# Created by Metrum AI for AMD

"""
Export DM-Count VGG19 model to ONNX for MIGraphX inference.

Run on CPU only — never initialises ROCm/CUDA. The exported ONNX file is
then loaded by ORT + MIGraphXExecutionProvider for GPU inference.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


class _DensityExport:
    """Thin wrapper so we can import this without torch at module level."""

    @staticmethod
    def export(
        weights_path: str,
        output_path: str,
        batch_size: int,
        infer_w: int,
        infer_h: int,
    ) -> None:
        import torch
        import torch.nn as nn

        from models.dm_count import VGG19DensityModel

        class _ExportWrapper(nn.Module):
            def __init__(self, base: VGG19DensityModel) -> None:
                super().__init__()
                self.base = base

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Return only the raw density map; count = density.sum()
                mu, _ = self.base(x)
                return mu

        logger.info("Loading DM-Count weights from %s (CPU only)", weights_path)
        model = VGG19DensityModel()
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()

        export_model = _ExportWrapper(model)
        dummy = torch.zeros(batch_size, 3, infer_h, infer_w, dtype=torch.float32)

        logger.info(
            "Exporting DM-Count ONNX (batch=%d input=%dx%d) → %s",
            batch_size, infer_w, infer_h, output_path,
        )
        # dynamo=False forces the legacy TorchScript-based exporter (no onnxscript needed).
        torch.onnx.export(
            export_model,
            dummy,
            output_path,
            dynamo=False,
            opset_version=17,
            input_names=["input"],
            output_names=["density_map"],
            dynamic_axes=None,  # Fixed batch → MIGraphX compiles one kernel, fastest
        )
        logger.info("DM-Count ONNX export complete → %s", output_path)


def export_dm_count_onnx(
    weights_path: str,
    output_path: str,
    batch_size: int = 4,
    infer_w: int = 320,
    infer_h: int = 240,
) -> bool:
    """Export DM-Count to ONNX. Returns True on success, False on failure."""
    out = Path(output_path)
    if out.exists():
        logger.info("DM-Count ONNX already exists at %s, skipping export", output_path)
        return True
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        _DensityExport.export(weights_path, output_path, batch_size, infer_w, infer_h)
        return True
    except Exception:
        logger.exception("DM-Count ONNX export failed")
        return False

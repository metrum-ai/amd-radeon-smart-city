# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
ONNX YOLO for the combined pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from multiprocessing.synchronize import Lock as LockType


def create_session(
    model_path: str,
    device_id: int,
    compile_lock: "LockType | None" = None,
) -> tuple[ort.InferenceSession, str, int, int, np.dtype]:
    """Build MIGraphX ORT session; optional process-wide lock for first compile."""
    providers = ort.get_available_providers()
    if "MIGraphXExecutionProvider" not in providers:
        raise RuntimeError(f"MIGraphXExecutionProvider unavailable: {providers}")

    path = str(Path(model_path).resolve())

    def _build() -> ort.InferenceSession:
        """Build the MIGraphX ORT session."""
        return ort.InferenceSession(
            path,
            providers=[("MIGraphXExecutionProvider", {"device_id": device_id})],
        )

    if compile_lock is not None:
        with compile_lock:
            session = _build()
    else:
        session = _build()

    input_meta = session.get_inputs()[0]
    sh = input_meta.shape

    def _dim(x: object, default: int) -> int:
        return int(x) if isinstance(x, int) else default

    height = _dim(sh[2], 288)
    width = _dim(sh[3], 384)
    dtype = np.float16 if "float16" in input_meta.type else np.float32
    return session, input_meta.name, width, height, dtype


def warmup_session(
    session: ort.InferenceSession,
    input_name: str,
    dtype: np.dtype,
    infer_w: int,
    infer_h: int,
    batch_sizes: list[int],
) -> None:
    """Warmup the MIGraphX ORT session."""
    for bs in batch_sizes:
        x = np.zeros((bs, 3, infer_h, infer_w), dtype=dtype)
        session.run(None, {input_name: x})


def frame_to_infer_plane(
    rgb_hwc: np.ndarray, dst_chw: np.ndarray, infer_w: int, infer_h: int
) -> None:
    """Resize if needed, write NCHW float into dst_chw (preallocated)."""
    import cv2

    h, w = rgb_hwc.shape[:2]
    if (w, h) != (infer_w, infer_h):
        resized = cv2.resize(rgb_hwc, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = rgb_hwc
    t = resized.astype(np.float32) / 255.0
    plane = np.transpose(t, (2, 0, 1))
    np.copyto(dst_chw, plane.astype(dst_chw.dtype, copy=False))

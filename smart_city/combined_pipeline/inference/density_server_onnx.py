# Created by Metrum AI for AMD

"""
ONNX-based DM-Count density server — runs on GPU via MIGraphXExecutionProvider.

Replaces the PyTorch density_server.py. No PyTorch at runtime; uses ORT only,
exactly like the YOLO server. This works on gfx1201 (RDNA4) where PyTorch ROCm hangs.

Input queue:  (stream_id: int, frame_rgb: np.ndarray HWC uint8, seq: int)
Output queues: per stream_id → (seq: int, density_map: np.ndarray 2D float32, count: float)
               maxsize=2, latest-wins (stale results are drained before each put).
"""
from __future__ import annotations

import logging
import queue
import sys
import time
from multiprocessing import Event, Queue
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (float32)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _roi_crop(
    frame: np.ndarray,
    infer_w: int,
    infer_h: int,
) -> np.ndarray:
    """Centre-crop to the inference aspect ratio then resize.

    Maximises person pixels per model cell by removing irrelevant background
    (sky, road, buildings) before downsampling rather than padding it in.
    Experiment on crowd_14 showed 8× higher count mean and 37/48 vs 5/48
    nonzero frames compared to letterbox at the same 160×120 resolution.

    Args:
        frame: HWC uint8 RGB input.
        infer_w: Target width.
        infer_h: Target height.

    Returns:
        HWC uint8 array of shape (infer_h, infer_w, 3).
    """
    h, w = frame.shape[:2]
    target_ar = infer_w / infer_h
    src_ar = w / h
    if src_ar > target_ar:
        # Source is wider than target: crop sides
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        cropped = frame[:, x0 : x0 + new_w]
    else:
        # Source is taller than target: crop top/bottom
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        cropped = frame[y0 : y0 + new_h, :]
    return cv2.resize(cropped, (infer_w, infer_h), interpolation=cv2.INTER_AREA)


def _preprocess_batch(
    frames: list[np.ndarray],
    infer_w: int,
    infer_h: int,
    batch_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    """ROI-crop, ImageNet-normalise, pad to fixed batch → NCHW float."""
    batch = np.zeros((batch_size, 3, infer_h, infer_w), dtype=np.float32)
    for i, frame in enumerate(frames):
        if i >= batch_size:
            break
        cropped = _roi_crop(frame, infer_w, infer_h)
        chw = np.ascontiguousarray(
            cropped.transpose(2, 0, 1), dtype=np.float32
        ) / 255.0
        batch[i] = chw
    # Normalise the active slots
    n = min(len(frames), batch_size)
    batch[:n] = (batch[:n] - _MEAN) / _STD
    return batch.astype(dtype)


def run_density_server_onnx(
    gpu_id: int,
    model_path: str,
    infer_w: int,
    infer_h: int,
    batch_size: int,
    batch_timeout_ms: float,
    in_queue: Queue,
    result_queues: dict[int, Queue],
    stop_event: Event,
    compile_lock,
    startup_delay_s: float = 90.0,  # wait for YOLO+workers to stabilise before compiling VGG19
    migraphx_options: dict | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info(
        "DensityServer-ONNX starting: gpu=%d display_streams=%d res=%dx%d delay=%.0fs",
        gpu_id, len(result_queues), infer_w, infer_h, startup_delay_s,
    )

    # MIGraphX JIT compilation heavily stresses the shared KFD kernel module.
    # Compiling VGG19 while YOLO is simultaneously starting up causes SIGABRT.
    # Wait until YOLO inference is in steady state before triggering GPU compilation.
    if startup_delay_s > 0:
        logger.info(
            "DensityServer-ONNX: waiting %.0fs for YOLO steady state before MIGraphX compile…",
            startup_delay_s,
        )
        deadline = time.monotonic() + startup_delay_s
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(1.0)
        if stop_event.is_set():
            logger.info("DensityServer-ONNX: stop requested during delay, exiting")
            return

    providers = ort.get_available_providers()
    if "MIGraphXExecutionProvider" not in providers:
        raise RuntimeError(f"MIGraphXExecutionProvider unavailable: {providers}")

    provider_options: dict = {"device_id": gpu_id}
    for k, v in (migraphx_options or {}).items():
        provider_options[k] = str(v)

    def _build() -> ort.InferenceSession:
        return ort.InferenceSession(
            model_path,
            providers=[("MIGraphXExecutionProvider", provider_options)],
        )

    with compile_lock:
        session = _build()

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    dtype_str = input_meta.type
    dtype = np.float16 if "float16" in dtype_str else np.float32

    # Warmup
    dummy = np.zeros((batch_size, 3, infer_h, infer_w), dtype=dtype)
    for _ in range(3):
        session.run(None, {input_name: dummy})
    logger.info("DensityServer-ONNX ready (gpu=%d batch=%d)", gpu_id, batch_size)

    timeout_s = max(batch_timeout_ms / 1000.0, 0.001)
    frame_count = 0
    batch_count = 0
    last_log = time.monotonic()

    while not stop_event.is_set():
        batch = _collect_batch(in_queue, batch_size, timeout_s, stop_event)
        if not batch:
            continue

        stream_ids: list[int] = []
        frames: list[np.ndarray] = []
        seqs: list[int] = []

        for sid, frame, seq in batch:
            if sid not in result_queues:
                continue
            stream_ids.append(sid)
            frames.append(frame)
            seqs.append(seq)

        if not frames:
            continue

        try:
            inp = _preprocess_batch(frames, infer_w, infer_h, batch_size, dtype)
            outs = session.run(None, {input_name: inp})
            density_batch = outs[0]  # (batch_size, 1, H', W')

            for i, sid in enumerate(stream_ids):
                out_q = result_queues.get(sid)
                if out_q is None:
                    continue
                density_map = density_batch[i, 0].astype(np.float32)
                count = float(np.maximum(density_map, 0).sum())
                # Latest-wins: discard stale result
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    out_q.put_nowait((seqs[i], density_map, count))
                except queue.Full:
                    pass

            frame_count += len(stream_ids)
            batch_count += 1
        except Exception:
            logger.exception("DensityServer-ONNX batch error")

        now = time.monotonic()
        if now - last_log >= 10.0:
            elapsed = now - last_log
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            avg = frame_count / batch_count if batch_count > 0 else 0.0
            logger.info(
                "DensityServer-ONNX: %.1f FPS (%.1f frames/batch avg, gpu=%d)",
                fps, avg, gpu_id,
            )
            frame_count = 0
            batch_count = 0
            last_log = now

    logger.info("DensityServer-ONNX stopped (gpu=%d)", gpu_id)


def _collect_batch(
    in_queue: Queue,
    max_batch: int,
    timeout_s: float,
    stop_event: Event,
) -> list[tuple[int, np.ndarray, int]]:
    batch: list[tuple[int, np.ndarray, int]] = []
    deadline = time.monotonic() + timeout_s
    while len(batch) < max_batch:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and batch:
            break
        wait = max(0.001, min(remaining, 0.02)) if batch else min(max(0.001, remaining), 0.05)
        try:
            item = in_queue.get(timeout=wait)
            batch.append(item)
        except queue.Empty:
            if batch:
                break
            if stop_event.is_set():
                break
    return batch

# Created by Metrum AI for AMD

from __future__ import annotations

import logging
import queue
import sys
import time
import zlib
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Lock
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from combined_pipeline.ipc.frame_store import attach_frame_store
from combined_pipeline.ipc.messages import Detection, InferJob, InferResult, result_to_payload

logger = logging.getLogger(__name__)

_KEEP_CLASSES = [0, 2]  # COCO: person, car


# ---------------------------------------------------------------------------
# MIGraphX hook (used by both backends for ONNX session creation)
# ---------------------------------------------------------------------------

def _install_migraphx_hook(gpu_id: int, migraphx_options: dict | None = None) -> None:
    """Intercept ONNX session creation to force MIGraphX provider.

    migraphx_options: extra provider options from config, e.g.:
        {"migraphx_fp16_enable": "1", "migraphx_exhaustive_tune": "1"}
    Supported keys (all string "1"/"0"):
        migraphx_fp16_enable, migraphx_bf16_enable,
        migraphx_int8_enable, migraphx_fp8_enable,
        migraphx_exhaustive_tune,
        migraphx_int8_calibration_table_name (path string),
        migraphx_int8_use_native_calibration_table
    """
    try:
        import onnxruntime as ort
        if "MIGraphXExecutionProvider" not in ort.get_available_providers():
            logger.warning("MIGraphXExecutionProvider not available")
            return
        _original_init = ort.InferenceSession.__init__

        ep_opts: dict = {"device_id": gpu_id}
        if migraphx_options:
            ep_opts.update({k: str(v) for k, v in migraphx_options.items()})

        active = [k for k, v in ep_opts.items() if k != "device_id" and str(v) not in ("0", "")]
        logger.info("MIGraphX hook installed for GPU %d — options: %s", gpu_id,
                    active if active else "fp32 (default)")

        def _migraphx_init(self_session, *args, **kwargs):
            providers = [("MIGraphXExecutionProvider", ep_opts), "CPUExecutionProvider"]
            args_list = list(args)
            if len(args_list) > 1:
                args_list[1] = providers
            kwargs["providers"] = providers
            if not kwargs.get("sess_options"):
                kwargs["sess_options"] = ort.SessionOptions()
            _original_init(self_session, *args_list, **kwargs)

        ort.InferenceSession.__init__ = _migraphx_init
    except ImportError:
        logger.warning("onnxruntime not available")


# ---------------------------------------------------------------------------
# Mask encoding helper (Ultralytics backend only)
# ---------------------------------------------------------------------------

def _encode_mask(
    mask_float: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    full_w: int, full_h: int,
) -> bytes | None:
    if mask_float.shape[1] != full_w or mask_float.shape[0] != full_h:
        mask_float = cv2.resize(mask_float, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(full_w, int(x2) + 1), min(full_h, int(y2) + 1)
    if bx2 <= bx1 or by2 <= by1:
        return None
    binary = (mask_float[by1:by2, bx1:bx2] > 0.5).astype(np.uint8)
    if not binary.any():
        return None
    return zlib.compress(binary.tobytes(), level=1)


# ---------------------------------------------------------------------------
# Ultralytics backend
# ---------------------------------------------------------------------------

def _run_ultralytics(
    gpu_id: int,
    model_path: str,
    infer_w: int,
    infer_h: int,
    full_w: int,
    full_h: int,
    store,
    in_queue: Queue,
    result_queues: list[Queue],
    stop_event: Event,
    ready_event: Event,
    compile_lock: Lock,
    batch_size: int,
    batch_timeout_ms: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    warmup_batch_sizes: list[int],
    migraphx_options: dict | None = None,
) -> None:
    _install_migraphx_hook(gpu_id, migraphx_options)
    from ultralytics import YOLO  # noqa: PLC0415

    with compile_lock:
        model = YOLO(model_path)
        if not model_path.endswith(".onnx"):
            model.to(f"cuda:{gpu_id}")
        logger.info("YOLOServer gpu=%d warming up (ultralytics, imgsz=%d)...", gpu_id, infer_w)
        dummy = np.zeros((infer_h, infer_w, 3), dtype=np.uint8)
        for bs in warmup_batch_sizes:
            model.predict([dummy] * bs, imgsz=infer_w, verbose=False,
                          conf=conf_threshold, iou=iou_threshold, classes=_KEEP_CLASSES)
        logger.info("YOLOServer gpu=%d warmup complete", gpu_id)

    ready_event.set()
    timeout_s = max(batch_timeout_ms / 1000.0, 0.001)
    pending: list[InferJob] = []
    deadline: float | None = None
    _fps_frames = 0
    _fps_batches = 0
    _fps_t0 = time.monotonic()

    def flush_batch(jobs: list[InferJob]) -> None:
        nonlocal _fps_frames, _fps_batches, _fps_t0
        frames = [store.full_view(j.full_slot).copy() for j in jobs]
        results = model.predict(frames, imgsz=infer_w, conf=conf_threshold,
                                iou=iou_threshold, classes=_KEEP_CLASSES,
                                verbose=False, max_det=max_detections)
        for job, result in zip(jobs, results):
            dets: list[Detection] = []
            if result.boxes is not None and len(result.boxes):
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                masks_data = result.masks.data.cpu().numpy() if result.masks is not None else None
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    mb: bytes | None = None
                    if masks_data is not None:
                        mb = _encode_mask(masks_data[i], x1, y1, x2, y2, full_w, full_h)
                    dets.append(Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                                          class_id=int(cls_ids[i]), score=float(confs[i]), mask_bytes=mb))
            r = InferResult(stream_id=job.stream_id, seq=job.seq, full_slot=job.full_slot,
                            infer_slot=job.infer_slot, width=job.width, height=job.height,
                            detections=dets, count=len(dets), ts_ns=job.ts_ns)
            result_queues[job.stream_id].put(result_to_payload(r))

        _fps_frames += len(jobs)
        _fps_batches += 1
        elapsed = time.monotonic() - _fps_t0
        if elapsed >= 10.0:
            logger.info("YOLOServer gpu=%d: %.1f FPS (%.1f frames/batch avg) [ultralytics]",
                        gpu_id, _fps_frames / elapsed, _fps_frames / max(_fps_batches, 1))
            _fps_frames = 0; _fps_batches = 0; _fps_t0 = time.monotonic()

    _run_loop(in_queue, stop_event, pending, deadline, timeout_s, batch_size, flush_batch)


# ---------------------------------------------------------------------------
# Raw ONNX backend — bypasses all Ultralytics overhead
# ---------------------------------------------------------------------------

def _run_raw_onnx(
    gpu_id: int,
    model_path: str,
    infer_w: int,
    infer_h: int,
    full_w: int,
    full_h: int,
    store,
    in_queue: Queue,
    result_queues: list[Queue],
    stop_event: Event,
    ready_event: Event,
    compile_lock: Lock,
    batch_size: int,
    batch_timeout_ms: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    warmup_batch_sizes: list[int],
    infer_dtype: np.dtype,
    migraphx_options: dict | None = None,
) -> None:
    _install_migraphx_hook(gpu_id, migraphx_options)
    from combined_pipeline.inference.onnx_yolo import create_session, warmup_session, frame_to_infer_plane  # noqa: PLC0415
    from combined_pipeline.inference.postprocess import (  # noqa: PLC0415
        decode_yolo_detect_batch,
        decode_detect_topk,
        decode_seg_topk,
    )

    session, input_name, _mw, _mh, dtype = create_session(model_path, device_id=gpu_id, compile_lock=compile_lock)
    logger.info("YOLOServer gpu=%d warming up (raw_onnx, imgsz=%d)...", gpu_id, infer_w)
    warmup_session(session, input_name, dtype, infer_w, infer_h, warmup_batch_sizes)
    logger.info("YOLOServer gpu=%d warmup complete", gpu_id)

    # Detect output format once at warmup time, not per-batch.
    # Topk models output (batch, K, 6) for detect or (batch, K, 38) for seg —
    # min feature dim ≤ 38.  Legacy raw models have ≥ 84 features.
    _probe = session.run(None, {input_name: np.zeros((1, 3, infer_h, infer_w), dtype=dtype)})
    _n_features = min(_probe[0].shape[1], _probe[0].shape[2])
    _is_topk = _n_features <= 38
    _is_seg_topk = _is_topk and len(_probe) > 1   # seg model has output1 (prototypes)
    logger.info(
        "YOLOServer gpu=%d output format: %s (n_features=%d)",
        gpu_id,
        "topk-seg" if _is_seg_topk else ("topk-detect" if _is_topk else "legacy-raw"),
        _n_features,
    )

    ready_event.set()

    timeout_s = max(batch_timeout_ms / 1000.0, 0.001)
    pending: list[InferJob] = []
    deadline: float | None = None
    _fps_frames = 0
    _fps_batches = 0
    _fps_t0 = time.monotonic()

    # Pre-allocate NCHW batch buffer
    batch_buf = np.zeros((batch_size, 3, infer_h, infer_w), dtype=dtype)

    _decode_kwargs = dict(
        infer_width=infer_w, infer_height=infer_h,
        full_width=full_w, full_height=full_h,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    def flush_batch(jobs: list[InferJob]) -> None:
        nonlocal _fps_frames, _fps_batches, _fps_t0
        active = len(jobs)
        for i, job in enumerate(jobs):
            frame_to_infer_plane(store.full_view(job.full_slot), batch_buf[i], infer_w=infer_w, infer_h=infer_h)

        # Always submit the full pre-allocated batch_buf (zero-padded) so MIGraphX
        # sees a constant input shape and never re-compiles for partial batches.
        outs = session.run(None, {input_name: batch_buf})
        output0 = outs[0][:active]

        if _is_seg_topk:
            output1 = outs[1][:active]
            dets_batch = decode_seg_topk(output0, output1, **_decode_kwargs)
        elif _is_topk:
            dets_batch = decode_detect_topk(output0, **_decode_kwargs)
        else:
            dets_batch = decode_yolo_detect_batch(output0, **_decode_kwargs)

        for job, dets in zip(jobs, dets_batch):
            r = InferResult(stream_id=job.stream_id, seq=job.seq, full_slot=job.full_slot,
                            infer_slot=job.infer_slot, width=job.width, height=job.height,
                            detections=dets, count=len(dets), ts_ns=job.ts_ns)
            result_queues[job.stream_id].put(result_to_payload(r))

        _fps_frames += active
        _fps_batches += 1
        elapsed = time.monotonic() - _fps_t0
        if elapsed >= 10.0:
            logger.info("YOLOServer gpu=%d: %.1f FPS (%.1f frames/batch avg) [raw_onnx]",
                        gpu_id, _fps_frames / elapsed, _fps_frames / max(_fps_batches, 1))
            _fps_frames = 0; _fps_batches = 0; _fps_t0 = time.monotonic()

    _run_loop(in_queue, stop_event, pending, deadline, timeout_s, batch_size, flush_batch)


# ---------------------------------------------------------------------------
# Shared event loop
# ---------------------------------------------------------------------------

def _run_loop(
    in_queue: Queue,
    stop_event: Event,
    pending: list[InferJob],
    deadline: float | None,
    timeout_s: float,
    batch_size: int,
    flush_batch,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                job = in_queue.get(timeout=0.02)
                if job is not None:
                    pending.append(job)
            except queue.Empty:
                pass

            now = time.monotonic()
            if pending and deadline is None:
                deadline = now + timeout_s

            if pending and (len(pending) >= batch_size or (deadline is not None and now >= deadline)):
                flush_batch(pending)
                pending.clear()
                deadline = None

        if pending:
            flush_batch(pending)
    finally:
        pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_yolo_server(
    gpu_id: int,
    model_path: str,
    infer_w: int,
    infer_h: int,
    infer_dtype_str: str,
    num_slots: int,
    full_h: int,
    full_w: int,
    full_shm_name: str,
    infer_shm_name: str,
    in_queue: Queue,
    result_queues: list[Queue],
    stop_event: Event,
    ready_event: Event,
    compile_lock: Lock,
    batch_size: int,
    batch_timeout_ms: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    warmup_batch_sizes: list[int],
    backend: str = "ultralytics",
    migraphx_options: dict | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger.info("YOLOServer gpu=%d starting (backend=%s)", gpu_id, backend)

    infer_dtype = np.dtype(infer_dtype_str)
    store = attach_frame_store(
        full_shm_name=full_shm_name, infer_shm_name=infer_shm_name,
        num_slots=num_slots, full_height=full_h, full_width=full_w,
        infer_height=infer_h, infer_width=infer_w, infer_dtype=infer_dtype,
    )

    common = dict(
        gpu_id=gpu_id, model_path=model_path, infer_w=infer_w, infer_h=infer_h,
        full_w=full_w, full_h=full_h, store=store, in_queue=in_queue,
        result_queues=result_queues, stop_event=stop_event, ready_event=ready_event,
        compile_lock=compile_lock, batch_size=batch_size, batch_timeout_ms=batch_timeout_ms,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        max_detections=max_detections, warmup_batch_sizes=warmup_batch_sizes,
        migraphx_options=migraphx_options,
    )

    try:
        if backend in ("raw_onnx", "onnx"):
            _run_raw_onnx(**common, infer_dtype=infer_dtype)
        else:
            _run_ultralytics(**common)
    finally:
        store.close()
        logger.info("YOLOServer gpu=%d stopped", gpu_id)

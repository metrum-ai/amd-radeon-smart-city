# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
YOLO server for the combined pipeline.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Lock
from pathlib import Path

import numpy as np
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from combined_pipeline.ipc.frame_store import attach_frame_store
from combined_pipeline.ipc.messages import Detection, InferJob, InferResult, result_to_payload
from combined_pipeline.inference.onnx_yolo import (
    create_session,
    frame_to_infer_plane,
    warmup_session,
)
from combined_pipeline.inference.postprocess import (
    decode_detect_topk,
    decode_seg_topk,
    decode_yolo_detect_batch,
)

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
# Raw ONNX backend
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
    """Run the YOLO server."""
    _install_migraphx_hook(gpu_id, migraphx_options)

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

    # The stream worker has already populated FrameStore.infer_view for every
    # submitted slot — a (3, H, W) preprocessed NCHW float32 plane in shared
    # memory. The previous implementation re-ran cv2.resize + astype + transpose
    # here, which dominated CPU time on the inference loop (~30-50% of the loop
    # at batch=32). We now memcpy the pre-built infer planes directly into the
    # contiguous batch buffer. infer_dtype must match dst dtype — the pipeline
    # orchestrator wires both to "float32", so this is a pure memcpy.
    _infer_match = (store.infer_dtype == dtype)
    if not _infer_match:
        logger.warning(
            "YOLOServer gpu=%d: infer_dtype=%s != model dtype=%s — "
            "falling back to per-frame resize (slower CPU path).",
            gpu_id, store.infer_dtype, dtype,
        )
    _decode_kwargs = dict(
        infer_width=infer_w, infer_height=infer_h,
        full_width=full_w, full_height=full_h,
        conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    # Optional micro-profiling: set YOLO_PROFILE_BATCH=1 to log per-batch
    # phase timings (preprocess / session.run / decode / put). Off by default
    # because the time.monotonic() calls add ~1us per phase.
    _profile = os.environ.get("YOLO_PROFILE_BATCH", "0") == "1"
    _prof_pre = 0.0
    _prof_run = 0.0
    _prof_dec = 0.0
    _prof_put = 0.0
    # Parallelise per-batch shared-memory→batch-buf memcpy across threads.
    # The np.copyto path releases the GIL inside the C-level memcpy, so a
    # thread pool truly overlaps the per-slot copies and scales near-linearly
    # in memory bandwidth. With FP32 batches at infer 384×384, single-threaded
    # measured ~53 ms/batch in the YOLO server profile; threaded scales it.
    # YOLO_PREPROCESS_THREADS controls the worker count; 0 keeps the previous
    # serial path for environments where threading regresses (e.g. tiny
    # batches or NUMA-bound hosts). Default 8 covers batch=32 nicely.
    _pre_threads = max(0, int(os.environ.get("YOLO_PREPROCESS_THREADS", "8")))
    _pre_executor: object | None = None
    if _pre_threads > 0:
        _pre_executor = ThreadPoolExecutor(
            max_workers=_pre_threads, thread_name_prefix=f"yolo-pre-{gpu_id}",
        )
        logger.info(
            "YOLOServer gpu=%d: parallel preprocess with %d threads", gpu_id, _pre_threads,
        )

    def _copy_chunk(chunk: list[tuple[int, int]]) -> None:
        """Copy a chunk."""
        # chunk: list of (batch_buf_index, infer_slot)
        for bi, slot in chunk:
            np.copyto(batch_buf[bi], store.infer_view(slot))

    def flush_batch(jobs: list[InferJob]) -> None:
        """Flush a batch."""
        nonlocal _fps_frames, _fps_batches, _fps_t0
        nonlocal _prof_pre, _prof_run, _prof_dec, _prof_put
        active = len(jobs)
        t0 = time.monotonic() if _profile else 0.0
        if _infer_match:
            if _pre_executor is not None and active >= _pre_threads * 2:
                # Static round-robin partition keeps each thread's chunk small
                # and predictable; over-partitioning isn't worth the extra
                # submit/join overhead at our batch sizes.
                pairs = [(i, jobs[i].infer_slot) for i in range(active)]
                step = (active + _pre_threads - 1) // _pre_threads
                chunks = [pairs[i:i + step] for i in range(0, active, step)]
                futs = [_pre_executor.submit(_copy_chunk, c) for c in chunks]  # type: ignore[union-attr]
                for f in futs:
                    f.result()
            else:
                for i, job in enumerate(jobs):
                    np.copyto(batch_buf[i], store.infer_view(job.infer_slot))
        else:
            for i, job in enumerate(jobs):
                frame_to_infer_plane(store.full_view(job.full_slot), batch_buf[i], infer_w=infer_w, infer_h=infer_h)
        t1 = time.monotonic() if _profile else 0.0

        # Always submit the full pre-allocated batch_buf (zero-padded) so MIGraphX
        # sees a constant input shape and never re-compiles for partial batches.
        outs = session.run(None, {input_name: batch_buf})
        output0 = outs[0][:active]
        t2 = time.monotonic() if _profile else 0.0

        if _is_seg_topk:
            output1 = outs[1][:active]
            dets_batch = decode_seg_topk(output0, output1, **_decode_kwargs)
        elif _is_topk:
            dets_batch = decode_detect_topk(output0, **_decode_kwargs)
        else:
            dets_batch = decode_yolo_detect_batch(output0, **_decode_kwargs)
        t3 = time.monotonic() if _profile else 0.0

        for job, dets in zip(jobs, dets_batch):
            r = InferResult(stream_id=job.stream_id, seq=job.seq, full_slot=job.full_slot,
                            infer_slot=job.infer_slot, width=job.width, height=job.height,
                            detections=dets, count=len(dets), ts_ns=job.ts_ns)
            result_queues[job.stream_id].put(result_to_payload(r))
        t4 = time.monotonic() if _profile else 0.0

        if _profile:
            _prof_pre += (t1 - t0) * 1000.0
            _prof_run += (t2 - t1) * 1000.0
            _prof_dec += (t3 - t2) * 1000.0
            _prof_put += (t4 - t3) * 1000.0

        _fps_frames += active
        _fps_batches += 1
        elapsed = time.monotonic() - _fps_t0
        if elapsed >= 10.0:
            avg_n = max(_fps_batches, 1)
            if _profile:
                logger.info(
                    "YOLOServer gpu=%d: %.1f FPS (%.1f frames/batch avg) [raw_onnx] "
                    "pre=%.2fms run=%.2fms dec=%.2fms put=%.2fms",
                    gpu_id, _fps_frames / elapsed, _fps_frames / avg_n,
                    _prof_pre / avg_n, _prof_run / avg_n,
                    _prof_dec / avg_n, _prof_put / avg_n,
                )
                _prof_pre = _prof_run = _prof_dec = _prof_put = 0.0
            else:
                logger.info("YOLOServer gpu=%d: %.1f FPS (%.1f frames/batch avg) [raw_onnx]",
                            gpu_id, _fps_frames / elapsed, _fps_frames / avg_n)
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
    """Run the loop."""
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
    backend: str = "raw_onnx",
    migraphx_options: dict | None = None,
) -> None:
    """Run the YOLO server."""
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
            raise ValueError(f"Unknown YOLO backend: {backend!r}. Only 'raw_onnx'/'onnx' are supported.")
    finally:
        store.close()
        logger.info("YOLOServer gpu=%d stopped", gpu_id)

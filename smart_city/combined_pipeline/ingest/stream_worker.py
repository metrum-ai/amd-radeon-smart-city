# Created by Metrum AI for AMD

from __future__ import annotations

import faulthandler
import logging
import queue
import sys
import time
from collections import deque
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from combined_pipeline.ingest.rtsp_source import RtspSource, pick_decoder
from combined_pipeline.inference.density_server_onnx import _roi_crop as density_roi_crop
from combined_pipeline.inference.onnx_yolo import frame_to_infer_plane
from combined_pipeline.ipc.frame_store import attach_frame_store, global_slot
from combined_pipeline.ipc.messages import InferJob, InferResult, payload_to_result
from combined_pipeline.output.rtsp_publisher import RtspPublisher
from combined_pipeline.overlay.renderer import (
    blend_heatmap_over_frame,
    compute_density_heatmap_rgb,
    draw_detections,
)

logger = logging.getLogger(__name__)


def should_submit_density_frame(seq: int, cadence: int) -> bool:
    """Return whether this frame should be sent to density inference."""
    return seq % max(1, cadence) == 0


def should_publish_output_frame(
    now: float,
    last_publish_at: float | None,
    fps: int,
) -> bool:
    """Return whether enough time has elapsed to publish another output frame."""
    if fps <= 0 or last_publish_at is None:
        return True
    return now - last_publish_at >= 1.0 / fps


def run_stream_worker(
    stream_id: int,
    input_rtsp: str,
    # Output RTSP endpoints
    yolo_rtsp: str,       # YOLO-only streams (non-density)
    density_rtsp: str,    # Legacy - kept for backward compat, not used
    paired_rtsp: str,     # Combined YOLO+Density side-by-side (density streams)
    gpu_id: int,
    slots_per_stream: int,
    num_global_slots: int,
    full_shm_name: str,
    infer_shm_name: str,
    full_w: int,
    full_h: int,
    infer_w: int,
    infer_h: int,
    infer_dtype_str: str,
    job_queue: Queue,
    result_queue: Queue,
    slot_free_queue: Queue,
    stop_event: Event,
    output_fps: int,
    encode_mode: str,
    decode_mode: str,
    # Density path - both None for YOLO-only streams
    density_in_queue: Optional[Queue],
    density_out_queue: Optional[Queue],
    density_frame_interval: int = 1,
    density_infer_w: int = 0,
    density_infer_h: int = 0,
    use_hw_encode: bool = False,
    vaapi_device: Optional[str] = None,
    hw_decode_stagger_ms: int = 0,
    # Shared stats dict (multiprocessing Manager dict) updated with live metrics
    shared_stats: Optional[Any] = None,
) -> None:
    faulthandler.enable()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    infer_dtype = np.dtype(infer_dtype_str)
    store = attach_frame_store(
        full_shm_name=full_shm_name,
        infer_shm_name=infer_shm_name,
        num_slots=num_global_slots,
        full_height=full_h,
        full_width=full_w,
        infer_height=infer_h,
        infer_width=infer_w,
        infer_dtype=infer_dtype,
    )

    # Stagger hardware decode startup to avoid VA-API session surge.
    # stream_id * stagger_ms ensures each worker opens its VA-API context
    # ~stagger_ms apart, preventing the driver from receiving 50 simultaneous
    # context-create calls which caused previous GPU hangs.
    if hw_decode_stagger_ms > 0 and decode_mode == "hardware":
        stagger_s = (stream_id * hw_decode_stagger_ms) / 1000.0
        if stagger_s > 0:
            logger.info("Stream %d: staggering VA-API init by %.1fs", stream_id, stagger_s)
            time.sleep(stagger_s)

    decoder = pick_decoder(decode_mode=decode_mode, gpu_id=gpu_id, vaapi_device=vaapi_device)
    logger.info("Stream %d: using decoder %s", stream_id, decoder)
    source = RtspSource(input_rtsp, decoder, width=full_w, height=full_h)
    source.start()

    is_density_stream = density_in_queue is not None and density_out_queue is not None

    # For density streams: single paired publisher (2*full_w x full_h)
    # For non-density streams: yolo-only publisher (full_w x full_h)
    yolo_publisher: RtspPublisher | None = None
    paired_publisher: RtspPublisher | None = None

    seq = 0
    expected_seq = 0
    pending: dict[int, InferResult] = {}

    # in_flight replaces slot_free_queue for intra-process slot tracking.
    # multiprocessing.Queue.put() is asynchronous (background feeder thread),
    # so get_nowait() called immediately after put() in the same process can
    # fail because the item is still in the buffer, not the pipe.  Tracking
    # with a plain integer has no such race.
    in_flight = 0

    # Latency tracker — only log on stream 0 to avoid log spam
    _lat_samples: deque[float] = deque(maxlen=600)
    _lat_log_interval = 300  # log percentiles every N frames
    _lat_frame_count = 0

    # FPS / metadata tracker — update shared_stats every 60 rendered frames
    _fps_window: deque[float] = deque(maxlen=120)  # frame render timestamps
    _latest_count: int = 0
    _latest_latency_ms: float = 0.0
    _stats_update_interval = 60  # frames between shared_stats updates
    _stats_frame_count = 0
    _last_output_publish_at: float | None = None

    # Seq-keyed buffer: map seq → (density_map, raw_count).
    # Bounded to 16 entries — older unmatched results are evicted to avoid
    # unbounded growth when density runs slower than YOLO.
    _DENSITY_BUF_MAX = 16
    _density_buf: dict[int, tuple[np.ndarray, float]] = {}

    # Smoothed count via EMA (display/stats only; raw sum unchanged in server).
    # Alpha 0.3 weights new frame at 30%, stabilises in ~3 frames.
    _COUNT_EMA_ALPHA = 0.3
    latest_density_map: np.ndarray | None = None
    latest_density_count: float = 0.0    # EMA-smoothed, for display
    _density_raw_count: float = 0.0      # raw DM-Count sum, kept for debug

    # Cache of the last computed heatmap RGB panel and the density_map array it
    # was built from. Density typically updates ~3-5× slower than YOLO at 50
    # streams, so caching the upscaled+colormapped panel saves the expensive
    # per-frame normalise/gamma/blur/resize/applyColorMap work and keeps just
    # the cv2.addWeighted blend on the per-frame hot path. Storing the ndarray
    # reference (not just id()) keeps it alive — avoids id() reuse after GC
    # accidentally returning a stale cached panel.
    _cached_heatmap_rgb: np.ndarray | None = None
    _cached_heatmap_source: np.ndarray | None = None

    def _poll_density() -> None:
        """Drain the density output queue into the seq-keyed buffer."""
        if not is_density_stream:
            return
        try:
            while True:
                d_seq, d_map, d_count = density_out_queue.get_nowait()  # type: ignore[union-attr]
                _density_buf[d_seq] = (d_map, d_count)
                # Evict oldest entries when buffer is full
                if len(_density_buf) > _DENSITY_BUF_MAX:
                    oldest = min(_density_buf)
                    del _density_buf[oldest]
        except queue.Empty:
            pass

    def _get_publisher(rtsp_url: str, width: int = full_w) -> RtspPublisher:
        return RtspPublisher(
            rtsp_url, width, full_h, output_fps, encode_mode, use_hw=use_hw_encode
        )

    def flush_pending() -> None:
        nonlocal expected_seq, yolo_publisher, paired_publisher
        nonlocal _lat_frame_count, in_flight
        nonlocal _latest_count, _latest_latency_ms, _stats_frame_count
        nonlocal latest_density_map, latest_density_count, _density_raw_count
        nonlocal _cached_heatmap_rgb, _cached_heatmap_source
        nonlocal _last_output_publish_at

        while expected_seq in pending:
            r = pending.pop(expected_seq)
            expected_seq += 1
            in_flight -= 1  # slot is now free (pure integer, no pipe/IPC overhead)

            # Measure end-to-end latency: frame captured → result ready for render
            lat_ms = 0.0
            if r.ts_ns > 0:
                lat_ms = (time.time_ns() - r.ts_ns) / 1_000_000.0
                if stream_id == 0:
                    _lat_samples.append(lat_ms)
                    _lat_frame_count += 1
                    if _lat_frame_count % _lat_log_interval == 0 and len(_lat_samples) >= 10:
                        sorted_s = sorted(_lat_samples)
                        n = len(sorted_s)
                        p50 = sorted_s[int(n * 0.50)]
                        p95 = sorted_s[int(n * 0.95)]
                        p99 = sorted_s[int(n * 0.99)]
                        p_min = sorted_s[0]
                        p_max = sorted_s[-1]
                        logger.info(
                            "LATENCY stream=0 frames=%d  "
                            "min=%.1fms  p50=%.1fms  p95=%.1fms  p99=%.1fms  max=%.1fms",
                            _lat_frame_count, p_min, p50, p95, p99, p_max,
                        )

            # Resolve density for this exact YOLO seq.
            # Prefer exact match; fall back to the closest older entry so the
            # heatmap is never from a *future* frame.
            if is_density_stream and _density_buf:
                matched_seq = r.seq if r.seq in _density_buf else max(
                    (s for s in _density_buf if s <= r.seq), default=None
                )
                if matched_seq is not None:
                    d_map, d_raw = _density_buf.pop(matched_seq)
                    # Apply EMA smoothing to displayed count only
                    if latest_density_map is None:
                        latest_density_count = d_raw  # cold start
                    else:
                        latest_density_count = (
                            _COUNT_EMA_ALPHA * d_raw
                            + (1.0 - _COUNT_EMA_ALPHA) * latest_density_count
                        )
                    _density_raw_count = d_raw
                    latest_density_map = d_map

            # Track FPS and latest crowd count for metadata emission
            _fps_window.append(time.monotonic())
            # Prefer DM-Count estimate when available for density-enabled streams.
            # Fall back to YOLO detection count until the first density frame arrives.
            if is_density_stream and latest_density_map is not None:
                _latest_count = max(0, int(round(latest_density_count)))
            else:
                _latest_count = int(r.count)
            if lat_ms > 0:
                _latest_latency_ms = lat_ms

            # Periodically push stats to shared dict (if provided)
            if shared_stats is not None:
                _stats_frame_count += 1
                if _stats_frame_count % _stats_update_interval == 0:
                    window = list(_fps_window)
                    if len(window) >= 2:
                        elapsed = window[-1] - window[0]
                        fps = round((len(window) - 1) / elapsed, 1) if elapsed > 0 else 0.0
                    else:
                        fps = 0.0
                    shared_stats[stream_id] = {
                        "count": _latest_count,
                        "det_count": int(r.count),
                        "density_count": (
                            float(latest_density_count)
                            if is_density_stream and latest_density_map is not None
                            else None
                        ),
                        "paired_published": bool(paired_publisher is not None),
                        "fps": fps,
                        "latency_ms": round(_latest_latency_ms, 1),
                        "status": "LIVE",
                    }

            now = time.monotonic()
            if not should_publish_output_frame(
                now, _last_output_publish_at, output_fps,
            ):
                continue
            _last_output_publish_at = now

            raw_frame = store.full_view(r.full_slot)

            # --- YOLO output frame ---
            yolo_frame = raw_frame.copy()
            det_label = f"det:{r.count}"
            draw_detections(yolo_frame, r.detections, count_label=det_label)

            if is_density_stream and latest_density_map is not None:
                # Density streams with density data: publish paired (YOLO | Density)
                # Recompute the heatmap RGB panel only when latest_density_map
                # actually changed (`is` identity check; we keep a reference to
                # the source array so its id() can't be reused by GC). Otherwise
                # the cached panel is still valid and the per-frame hot path is
                # just frame.copy() + addWeighted in blend_heatmap_over_frame.
                if (
                    _cached_heatmap_rgb is None
                    or _cached_heatmap_source is not latest_density_map
                    or _cached_heatmap_rgb.shape[:2] != raw_frame.shape[:2]
                ):
                    _cached_heatmap_rgb = compute_density_heatmap_rgb(
                        latest_density_map,
                        raw_frame.shape[0],
                        raw_frame.shape[1],
                    )
                    _cached_heatmap_source = latest_density_map
                density_frame = blend_heatmap_over_frame(
                    raw_frame, _cached_heatmap_rgb, alpha=0.55,
                )
                paired_frame = np.hstack([yolo_frame, density_frame])

                # Close yolo publisher if we were using it during warmup
                if yolo_publisher is not None:
                    try:
                        yolo_publisher.stop()
                    except Exception:
                        pass
                    yolo_publisher = None

                if paired_publisher is None:
                    paired_publisher = _get_publisher(paired_rtsp, width=full_w * 2)
                try:
                    paired_publisher.write(paired_frame)
                except RuntimeError:
                    logger.warning(
                        "stream_id=%d: paired publisher died, restarting", stream_id
                    )
                    try:
                        paired_publisher.stop()
                    except Exception:
                        pass
                    time.sleep(0.5 + stream_id * 0.02)
                    paired_publisher = _get_publisher(paired_rtsp, width=full_w * 2)
                    paired_publisher.write(paired_frame)
            else:
                # YOLO-only: non-density streams OR density streams during warmup
                if yolo_publisher is None:
                    yolo_publisher = _get_publisher(yolo_rtsp)
                try:
                    yolo_publisher.write(yolo_frame)
                except RuntimeError:
                    logger.warning(
                        "stream_id=%d: YOLO publisher died, restarting", stream_id
                    )
                    try:
                        yolo_publisher.stop()
                    except Exception:
                        pass
                    time.sleep(0.5 + stream_id * 0.02)
                    yolo_publisher = _get_publisher(yolo_rtsp)
                    yolo_publisher.write(yolo_frame)

    def _drain_results() -> None:
        """Non-blocking drain of result_queue; flush any in-order pending results.

        Drain ALL queued payloads into `pending` first, then poll density once
        and call flush_pending once. The previous version called flush_pending
        after every single get_nowait, which interleaved the heavy publish work
        (cv2 ops + np.hstack + ffmpeg pipe write) between drain steps and
        forced the worker to publish-one, drain-one, publish-one, ... — a
        latency win at the cost of throughput. Batching the drain lets the
        worker pull the entire backlog in microseconds, then do one publish
        pass for all in-order frames.
        """
        drained = False
        try:
            while True:
                pay = result_queue.get_nowait()
                r = payload_to_result(pay)
                pending[r.seq] = r
                drained = True
        except queue.Empty:
            pass
        if drained:
            _poll_density()
            flush_pending()

    def _wait_for_slot() -> None:
        """Block on the result queue until in_flight drops below slots_per_stream.

        Replaces slot_free_queue.get(timeout=X): pure blocking on result_queue
        means the OS wakes this process the instant a result arrives, with no
        pipe-feeder-thread race and no 50ms dead-time overhead.
        """
        while in_flight >= slots_per_stream:
            try:
                pay = result_queue.get(timeout=0.05)
                r = payload_to_result(pay)
                pending[r.seq] = r
            except queue.Empty:
                continue
            # Drain any additional results that arrived during the blocking get,
            # then poll density and publish in one shot (see _drain_results).
            try:
                while True:
                    pay = result_queue.get_nowait()
                    r = payload_to_result(pay)
                    pending[r.seq] = r
            except queue.Empty:
                pass
            _poll_density()
            flush_pending()

    try:
        while not stop_event.is_set():
            if source.error:
                raise RuntimeError(source.error)

            _drain_results()

            if in_flight >= slots_per_stream:
                _wait_for_slot()
                if stop_event.is_set():
                    break

            try:
                frame = source.read(timeout_seconds=1.0)
            except TimeoutError:
                continue

            # Round-robin slot assignment: safe because in_flight < slots_per_stream
            # guarantees each local_slot is idle before reuse.
            local_slot = seq % slots_per_stream
            gs = global_slot(stream_id, slots_per_stream, local_slot)
            np.copyto(store.full_view(gs), frame)
            frame_to_infer_plane(frame, store.infer_view(gs), infer_w=infer_w, infer_h=infer_h)
            in_flight += 1

            job = InferJob(
                stream_id=stream_id,
                seq=seq,
                full_slot=gs,
                infer_slot=gs,
                width=full_w,
                height=full_h,
                ts_ns=time.time_ns(),
            )
            job_queue.put(job)

            if is_density_stream and should_submit_density_frame(
                seq, density_frame_interval
            ):
                # Pre-crop+resize to the density inference size in the worker so the
                # queue payload drops from 640x640x3 (~1.2 MB) to 160x120x3 (~58 kB)
                # — that's a 21× reduction in per-frame pickle/copy cost across
                # 50 streams, and it eliminates the per-frame full-frame.copy().
                # When density_infer_{w,h} are 0 (legacy callers), fall back to
                # shipping the full frame so the server can still crop+resize.
                if density_infer_w > 0 and density_infer_h > 0:
                    tile = density_roi_crop(frame, density_infer_w, density_infer_h)
                    payload = np.ascontiguousarray(tile)
                else:
                    payload = frame.copy()
                try:
                    density_in_queue.put_nowait((stream_id, payload, seq))  # type: ignore[union-attr]
                except queue.Full:
                    pass

            seq += 1

    finally:
        source.stop()
        if yolo_publisher is not None:
            yolo_publisher.stop()
        if paired_publisher is not None:
            paired_publisher.stop()
        store.close()
        if shared_stats is not None:
            try:
                shared_stats[stream_id] = {
                    "count": _latest_count,
                    "paired_published": False,
                    "fps": 0.0,
                    "latency_ms": round(_latest_latency_ms, 1),
                    "status": "OFFLINE",
                }
            except Exception:
                pass
        logger.info("StreamWorker stream_id=%d stopped", stream_id)

# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import threading
import time
from multiprocessing import Event, Lock, Process, Queue
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from combined_pipeline.control.readiness import wait_tcp
from combined_pipeline.control.startup import load_pipeline_config, output_rtsp_url
from combined_pipeline.ipc.frame_store import FrameStore, make_slot_free_queues
from combined_pipeline.orchestrator.gpu_router import gpu_for_stream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spawned process entry points
# ---------------------------------------------------------------------------

def _run_yolo_server(**kwargs: object) -> None:
    """YOLO server — each process sees only its own physical GPU via ROCR isolation.

    ROCR_VISIBLE_DEVICES=N exposes only physical GPU N to this process.
    Inside the process, device_id=0 always refers to the first (and only) visible GPU.
    The gpu_id kwarg is remapped to 0 before passing to run_yolo_server.
    """
    physical_gpu = int(kwargs.get("gpu_id", 0))
    os.environ["ROCR_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ["HSA_VISIBLE_DEVICES"] = str(physical_gpu)
    hsa_gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_gfx
    # Within this process, the one visible GPU is always device_id=0.
    kwargs = dict(kwargs)
    kwargs["gpu_id"] = 0
    from combined_pipeline.inference.yolo_server import run_yolo_server
    run_yolo_server(**kwargs)  # type: ignore[arg-type]


def _run_density_server_onnx(**kwargs: object) -> None:
    """Density server — confined to a single physical GPU via ROCR isolation."""
    phys_gpu = str(kwargs.pop("physical_gpu", 1))
    os.environ["ROCR_VISIBLE_DEVICES"] = phys_gpu
    os.environ["HSA_VISIBLE_DEVICES"] = phys_gpu
    # Explicitly pin HSA_OVERRIDE_GFX_VERSION so MIGraphX sees the correct
    # target architecture (gfx1201 / RDNA4) regardless of spawn-time env state.
    hsa_gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_gfx
    # Disable MIGraphX MLIR backend for density inference.
    # The MLIR path generates mlir_convert_convolution_broadcast_add_relu kernels
    # that produce HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION on gfx1201 when FP16 is
    # enabled.  The non-MLIR MIGraphX path is gfx1201-compatible and still
    # honours migraphx_fp16_enable without crashing. Keep this scoped to density:
    # YOLO's raw_onnx path also uses MIGraphX and should keep the known-good
    # default MLIR behavior.
    os.environ.setdefault("MIGRAPHX_DISABLE_MLIR", "1")
    from combined_pipeline.inference.density_server_onnx import run_density_server_onnx
    run_density_server_onnx(**kwargs)  # type: ignore[arg-type]


def _run_stream_worker(**kwargs: object) -> None:
    gpu_id = int(kwargs.get("gpu_id", 0))
    decode_mode = str(kwargs.get("decode_mode", "software"))
    vaapi_gpu_id = int(kwargs.pop("vaapi_gpu_id", 1))  # default: VA-API on GPU 1

    if not os.environ.get("XDG_RUNTIME_DIR"):
        xdg_dir = tempfile.mkdtemp(prefix="xdg-runtime-")
        os.chmod(xdg_dir, 0o700)
        os.environ["XDG_RUNTIME_DIR"] = xdg_dir

    # For hardware decode, use the dedicated VA-API GPU (not the YOLO GPU).
    # This keeps VA-API's VCN video decode engine off the YOLO compute GPU,
    # preventing the KFD hang we saw when both shared GPU 0.
    if decode_mode == "hardware":
        os.environ["GST_VAAPI_DRM_DEVICE"] = f"/dev/dri/renderD{128 + vaapi_gpu_id}"
        os.environ["LIBVA_DRIVER_NAME"] = "radeonsi"
        os.environ.pop("LIBVA_DRIVERS_PATH", None)
    else:
        os.environ["GST_VAAPI_DRM_DEVICE"] = f"/dev/dri/renderD{128 + gpu_id}"

    for key in (
        "ROCR_VISIBLE_DEVICES",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "VK_ICD_FILENAMES",
    ):
        os.environ.pop(key, None)

    from combined_pipeline.ingest.stream_worker import run_stream_worker
    run_stream_worker(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_vaapi_once() -> bool:
    """Probe hardware encode ONCE from the orchestrator process (before any worker spawns).
    Returns True if h264_vaapi is functional."""
    import subprocess
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:r=1",
                "-frames:v", "1",
                "-vaapi_device", "/dev/dri/renderD128",
                "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi",
                "-f", "null", "-",
            ],
            capture_output=True,
            timeout=10,
        )
        ok = r.returncode == 0
        logger.info("VAAPI encode probe: %s", "available" if ok else "unavailable")
        return ok
    except Exception as exc:
        logger.warning("VAAPI encode probe failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if cfg is None:
        cfg = load_pipeline_config(cfg_path)

    streams_cfg = cfg["streams"]
    ingest_cfg = cfg.get("ingest", {})
    yolo_cfg = cfg["yolo"]
    density_cfg = cfg.get("density", {})
    output_cfg = cfg["output"]
    shmem_cfg = cfg.get("shmem", {})
    mtx_cfg = cfg["mediamtx"]

    mp.set_start_method("spawn", force=True)

    stream_count = int(streams_cfg["count"])
    gpu_count = max(1, int(streams_cfg.get("gpu_count", 2)))
    input_base = str(streams_cfg["input_base_rtsp"])
    yolo_suffix = str(streams_cfg.get("yolo_output_suffix", "_yolo"))
    density_suffix = str(streams_cfg.get("density_output_suffix", "_density"))
    paired_suffix = str(streams_cfg.get("paired_output_suffix", "_paired"))

    full_w = int(output_cfg["full_width"])
    full_h = int(output_cfg["full_height"])
    infer_w = int(yolo_cfg["infer_width"])
    infer_h = int(yolo_cfg["infer_height"])
    slots_per_stream = int(shmem_cfg.get("slots_per_stream", 4))
    num_slots = stream_count * slots_per_stream

    model_path = str(yolo_cfg["model_path"])
    batch_size = int(yolo_cfg["batch_size"])
    batch_timeout_ms = float(yolo_cfg["batch_timeout_ms"])
    conf_th = float(yolo_cfg.get("conf_threshold", 0.25))
    iou_th = float(yolo_cfg.get("iou_threshold", 0.45))
    max_det = int(yolo_cfg.get("max_detections", 100))
    warmup_bs = [int(x) for x in yolo_cfg.get("warmup_batch_sizes", [1, 4, 8])]
    yolo_backend = str(yolo_cfg.get("backend", "raw_onnx"))
    migraphx_options = yolo_cfg.get("migraphx_options", {}) or {}
    replicas_per_gpu = max(1, int(yolo_cfg.get("replicas_per_gpu", 1)))
    yolo_start_gpu = int(yolo_cfg.get("start_gpu_id", 0))  # physical GPU offset for YOLO servers

    decode_mode = str(ingest_cfg.get("decode_mode", "software"))
    # Hardware decode routes to GPU 1 (renderD129) to keep VA-API off the YOLO GPU.
    # GPU 1 is idle during stream startup (DensityServer has 110s delay) so its
    # VCN video decode block is available without competing with MIGraphX.
    hw_decode_vaapi_gpu = int(ingest_cfg.get("hw_decode_vaapi_gpu", 1))
    vaapi_device_path = f"/dev/dri/renderD{128 + hw_decode_vaapi_gpu}"
    # Limit hardware decode to first N streams to avoid KFD overload at scale.
    hw_decode_max = int(ingest_cfg.get("hw_decode_max_streams", stream_count))
    # Stagger VA-API context creation across workers to avoid surge-induced driver hang.
    hw_decode_stagger_ms = int(ingest_cfg.get("hw_decode_stagger_ms", 200))
    encode_mode = str(output_cfg.get("encode_mode", "hardware"))
    output_fps = int(output_cfg.get("fps", 30))

    density_enabled = bool(density_cfg.get("enabled", True))
    display_streams = min(int(density_cfg.get("display_streams", 15)), stream_count)
    density_onnx_path = str(density_cfg.get("onnx_path", "/app/models/dm_count.onnx"))
    density_gpu_id = int(density_cfg.get("gpu_id", 0))        # 0 = first visible GPU inside isolated process
    density_physical_gpu = int(density_cfg.get("physical_gpu", 1))  # physical GPU exposed via ROCR
    density_batch = int(density_cfg.get("batch_size", 4))
    density_timeout_ms = float(density_cfg.get("batch_timeout_ms", 50))
    density_infer_w = int(density_cfg.get("infer_width", 320))
    density_infer_h = int(density_cfg.get("infer_height", 240))
    density_frame_interval = max(1, int(density_cfg.get("frame_interval", 1)))
    density_migraphx_options = density_cfg.get("migraphx_options", {}) or {}

    host = str(mtx_cfg["host"])
    rtsp_port = int(mtx_cfg["rtsp_port"])
    infer_dtype = "float32"
    out_base = cfg.get("_env", {}).get("output_base_rtsp") or f"rtsp://{host}:{rtsp_port}/cam"

    # Path where live pipeline stats are written for the FastAPI backend to read.
    # Configurable via env var; default matches the API container's default so
    # local runs without Docker still find the file at the same path.
    pipeline_stats_path = os.environ.get(
        "PIPELINE_STATS_FILE", "/pipeline_stats/pipeline_stats.json"
    )

    # Check ONNX density model exists when density is enabled
    if density_enabled and not Path(density_onnx_path).exists():
        logger.warning(
            "DM-Count ONNX not found at %s — density disabled. "
            "Run main.py to export it first.", density_onnx_path
        )
        density_enabled = False

    # Pre-probe VAAPI encode ONCE before spawning any workers.
    # This prevents 50 concurrent probe subprocesses from crashing the amdgpu driver.
    use_hw_encode = False
    if encode_mode == "hardware":
        use_hw_encode = _probe_vaapi_once()
        if not use_hw_encode:
            logger.info("Falling back to libx264 software encode for all streams")

    # Each replica independently batches from the shared job_queue, so
    # per-replica batch size = total batch_size / replicas_per_gpu.
    replica_batch_size = max(1, batch_size // replicas_per_gpu)

    logger.info(
        "Pipeline: streams=%d yolo_gpus=%d replicas/gpu=%d batch=%d(per-replica=%d) "
        "density=%s(%d streams, phys-gpu%d, every=%d) decode=%s encode=%s",
        stream_count, gpu_count, replicas_per_gpu, batch_size, replica_batch_size,
        "onnx-gpu" if density_enabled else "off",
        display_streams if density_enabled else 0,
        density_physical_gpu,
        density_frame_interval,
        decode_mode,
        "hw(vaapi)" if use_hw_encode else "sw(libx264)",
    )

    if not wait_tcp(host, rtsp_port, timeout_s=90.0):
        raise RuntimeError(f"MediaMTX RTSP not reachable at {host}:{rtsp_port}")

    # --- Shared memory ---
    import numpy as np
    store = FrameStore(
        num_slots=num_slots,
        full_height=full_h,
        full_width=full_w,
        infer_height=infer_h,
        infer_width=infer_w,
        infer_dtype=np.dtype(infer_dtype),
    )

    # Shared stats dict (Manager dict) — stream workers write live count/fps/latency.
    # A background thread dumps it to pipeline_stats_path every 2 s for cross-process
    # and cross-container consumption by the FastAPI backend.
    _manager = mp.Manager()
    shared_stats = _manager.dict()

    # One compile lock serialises MIGraphX compilation across YOLO + Density servers.
    # With replicas, this ensures only one replica compiles at a time — critical since
    # MIGraphX compilation is GPU-memory-intensive and concurrent compiles can OOM.
    compile_lock = Lock()
    stop_event = Event()

    # One ready_event per replica (gpu_count × replicas_per_gpu)
    total_yolo_replicas = gpu_count * replicas_per_gpu
    ready_events: list[Event] = [Event() for _ in range(total_yolo_replicas)]

    job_qs: list[Queue] = [Queue() for _ in range(gpu_count)]
    result_qs: list[Queue] = [Queue() for _ in range(stream_count)]
    slot_qs = make_slot_free_queues(stream_count, slots_per_stream)

    # --- Density queues ---
    density_in_q: Optional[Queue] = None
    density_out_qs: dict[int, Queue] = {}
    if density_enabled and display_streams > 0:
        density_in_q = Queue(maxsize=display_streams * 2)
        for sid in range(display_streams):
            density_out_qs[sid] = Queue(maxsize=2)

    # --- YOLO servers (N replicas per GPU) ---
    # All replicas on the same GPU share a single job_queue (competitive consumers).
    # Each replica independently collects batches of replica_batch_size and runs
    # inference. This mirrors the Ray Serve pattern: while one replica is doing
    # CPU preprocessing/postprocessing, another is running GPU inference.
    replica_warmup_bs = [max(1, bs // replicas_per_gpu) for bs in warmup_bs]
    yolo_base_args = dict(
        model_path=model_path,
        infer_w=infer_w,
        infer_h=infer_h,
        infer_dtype_str=infer_dtype,
        num_slots=num_slots,
        full_h=full_h,
        full_w=full_w,
        full_shm_name=store.full_shm_name,
        infer_shm_name=store.infer_shm_name,
        stop_event=stop_event,
        compile_lock=compile_lock,
        batch_size=replica_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        conf_threshold=conf_th,
        iou_threshold=iou_th,
        max_detections=max_det,
        warmup_batch_sizes=replica_warmup_bs,
        backend=yolo_backend,
        migraphx_options=migraphx_options,
    )

    yolo_procs: list[Process] = []
    replica_idx = 0
    for gid in range(gpu_count):
        phys_gid = gid + yolo_start_gpu  # apply physical GPU offset
        for rid in range(replicas_per_gpu):
            p = Process(
                target=_run_yolo_server,
                kwargs={
                    **yolo_base_args,
                    "gpu_id": phys_gid,
                    "in_queue": job_qs[gid],
                    "result_queues": result_qs,
                    "ready_event": ready_events[replica_idx],
                },
                name=f"yolo-gpu{phys_gid}-r{rid}",
                daemon=True,
            )
            p.start()
            yolo_procs.append(p)
            logger.info(
                "YOLOServer gpu=%d replica=%d started (pid=%d, batch=%d)",
                phys_gid, rid, p.pid, replica_batch_size,
            )
            replica_idx += 1

    for idx, ready in enumerate(ready_events):
        while not ready.is_set():
            proc = yolo_procs[idx]
            if not proc.is_alive():
                stop_event.set()
                raise RuntimeError(
                    f"YOLO replica {idx} ({proc.name}) exited before ready "
                    f"(exitcode={proc.exitcode})"
                )
            time.sleep(0.2)
        logger.info("YOLO replica %d (%s) ready", idx, yolo_procs[idx].name)

    # --- Background stats dump thread ---
    # Writes shared_stats to pipeline_stats_path every 2 s so the FastAPI backend
    # (running in another process/container) can read live pipeline metrics.
    def _stats_dump_loop() -> None:
        while not stop_event.is_set():
            try:
                snapshot = dict(shared_stats)
                tmp_path = pipeline_stats_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(snapshot, f)
                os.replace(tmp_path, pipeline_stats_path)
            except Exception as exc:
                logger.debug("Stats dump error: %s", exc)
            time.sleep(2.0)

    stats_thread = threading.Thread(target=_stats_dump_loop, daemon=True, name="stats-dump")
    stats_thread.start()
    logger.info("Stats dump thread started → %s", pipeline_stats_path)

    # --- Density server (ONNX + MIGraphX on GPU) ---
    density_proc: Optional[Process] = None
    if density_enabled and display_streams > 0 and density_in_q is not None:
        # Delay = (stream_count streams × 1 s stagger) + 60 s steady-state buffer.
        # VGG19 MIGraphX compilation stresses the shared KFD module; delaying until
        # all 50 stream workers are up and YOLO is in steady state prevents SIGABRT.
        density_startup_delay = stream_count * 1.0 + 60.0
        density_proc = Process(
            target=_run_density_server_onnx,
            kwargs=dict(
                gpu_id=density_gpu_id,
                physical_gpu=density_physical_gpu,
                model_path=density_onnx_path,
                infer_w=density_infer_w,
                infer_h=density_infer_h,
                batch_size=density_batch,
                batch_timeout_ms=density_timeout_ms,
                in_queue=density_in_q,
                result_queues=density_out_qs,
                stop_event=stop_event,
                compile_lock=compile_lock,
                startup_delay_s=density_startup_delay,
                migraphx_options=density_migraphx_options,
            ),
            name="density-onnx",
            daemon=True,
        )
        density_proc.start()
        logger.info(
            "DensityServer-ONNX started (pid=%d gpu=%d display_streams=%d)",
            density_proc.pid, density_gpu_id, display_streams,
        )

    # --- Stream workers (staggered 1 s apart to avoid VA-API init spikes) ---
    workers: list[Process] = []
    for sid in range(stream_count):
        gid = gpu_for_stream(sid, gpu_count)
        cam_i = sid + 1
        in_url = f"{input_base}{cam_i}"
        yolo_url = output_rtsp_url(out_base, cam_i, yolo_suffix)
        density_url = output_rtsp_url(out_base, cam_i, density_suffix)
        paired_url = output_rtsp_url(out_base, cam_i, paired_suffix)

        # Per-stream decode mode: only first hw_decode_max streams get hardware decode.
        stream_decode_mode = decode_mode if (decode_mode == "hardware" and sid < hw_decode_max) else "software"

        p = Process(
            target=_run_stream_worker,
            kwargs=dict(
                stream_id=sid,
                input_rtsp=in_url,
                yolo_rtsp=yolo_url,
                density_rtsp=density_url,
                paired_rtsp=paired_url,
                gpu_id=gid,
                slots_per_stream=slots_per_stream,
                num_global_slots=num_slots,
                full_shm_name=store.full_shm_name,
                infer_shm_name=store.infer_shm_name,
                full_w=full_w,
                full_h=full_h,
                infer_w=infer_w,
                infer_h=infer_h,
                infer_dtype_str=infer_dtype,
                job_queue=job_qs[gid],
                result_queue=result_qs[sid],
                slot_free_queue=slot_qs[sid],
                stop_event=stop_event,
                output_fps=output_fps,
                encode_mode=encode_mode,
                decode_mode=stream_decode_mode,
                vaapi_gpu_id=hw_decode_vaapi_gpu,
                vaapi_device=vaapi_device_path if stream_decode_mode == "hardware" else None,
                hw_decode_stagger_ms=hw_decode_stagger_ms if stream_decode_mode == "hardware" else 0,
                density_in_queue=density_in_q if sid < display_streams else None,
                density_out_queue=density_out_qs.get(sid),
                density_frame_interval=density_frame_interval,
                density_infer_w=density_infer_w if density_enabled else 0,
                density_infer_h=density_infer_h if density_enabled else 0,
                use_hw_encode=use_hw_encode,
                shared_stats=shared_stats,
            ),
            name=f"stream-{sid}",
            daemon=True,
        )
        p.start()
        workers.append(p)
        logger.info(
            "StreamWorker-%d started (pid=%d gpu=%d density=%s decode=%s)",
            sid, p.pid, gid, "yes" if sid < display_streams else "no", stream_decode_mode,
        )
        time.sleep(1.0)

    # --- Signal handling ---
    def _shutdown(*_args: object) -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    all_procs = yolo_procs + ([density_proc] if density_proc else []) + workers

    try:
        while not stop_event.is_set():
            for p in all_procs:
                if not p.is_alive() and p.exitcode not in (0, None):
                    logger.error(
                        "child %s exited unexpectedly (exitcode=%d)", p.name, p.exitcode
                    )
                    stop_event.set()
                    break
            time.sleep(0.5)
    finally:
        stop_event.set()
        for p in all_procs:
            p.join(timeout=8)
            if p.is_alive():
                p.terminate()
        store.close()
        store.unlink()
        try:
            _manager.shutdown()
        except Exception:
            pass
        logger.info("Pipeline stopped")

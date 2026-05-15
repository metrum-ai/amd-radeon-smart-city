# Created by Metrum AI for AMD

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache so probing happens once per process, not once per stream
_vaapi_encode_available: Optional[bool] = None
_vaapi_lock = threading.Lock()

# Default render node for VA-API encode. Override with VAAPI_ENCODE_DEVICE env
# var when routing encode to a non-default GPU (e.g. when YOLO occupies GPU 0
# and you want encode to land on GPU 1's VCN engine).
DEFAULT_VAAPI_ENCODE_DEVICE = "/dev/dri/renderD128"


def _vaapi_device_path() -> str:
    return os.environ.get("VAAPI_ENCODE_DEVICE", DEFAULT_VAAPI_ENCODE_DEVICE)


def _probe_vaapi_encode(device: Optional[str] = None) -> bool:
    """Run a single-frame test encode to verify h264_vaapi is functional.

    Uses 256x256 (above the 96x32 minimum supported by AMD VCN H.264) so the
    probe succeeds when the codec is actually usable. The previous 64x64 probe
    failed even on a working device because 64x64 is below VCN's min width.
    """
    global _vaapi_encode_available
    with _vaapi_lock:
        if _vaapi_encode_available is not None:
            return _vaapi_encode_available
        dev = device or _vaapi_device_path()
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "nullsrc=s=256x256:r=1",
                    "-frames:v", "1",
                    "-vaapi_device", dev,
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi",
                    "-f", "null", "-",
                ],
                capture_output=True,
                timeout=8,
            )
            _vaapi_encode_available = result.returncode == 0
        except Exception:
            _vaapi_encode_available = False
        logger.info(
            "VAAPI encode probe (%s): %s",
            dev,
            "ok" if _vaapi_encode_available else "unavailable",
        )
        return _vaapi_encode_available  # type: ignore[return-value]


def _build_ffmpeg_cmd(
    rtsp_url: str,
    width: int,
    height: int,
    fps: int,
    use_hw: bool,
) -> list[str]:
    key_interval = max(1, fps)
    common = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-use_wallclock_as_timestamps", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
    ]
    # Target ~1.5 Mbps per stream (paired 1280×640 ≈ 1.5M, single 640×640 ≈ 1M).
    # Without bitrate control, ultrafast+zerolatency produces 5-8 Mbps which
    # overwhelms WebRTC sessions when multiple streams are viewed concurrently
    # (MediaMTX logs "reader is too slow, discarding N frames").
    pixels = width * height
    target_kbps = max(800, min(2000, pixels // 250))
    max_kbps = int(target_kbps * 1.5)
    bufsize_kbps = target_kbps * 2

    if use_hw:
        enc = [
            "-vaapi_device", _vaapi_device_path(),
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-profile:v", "constrained_baseline",
            "-b:v", f"{target_kbps}k",
            "-maxrate", f"{max_kbps}k",
            "-bufsize", f"{bufsize_kbps}k",
            "-bf", "0",
            "-g", str(key_interval),
            "-force_key_frames", "expr:gte(t,n_forced*1)",
        ]
    else:
        enc = [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-b:v", f"{target_kbps}k",
            "-maxrate", f"{max_kbps}k",
            "-bufsize", f"{bufsize_kbps}k",
            "-bf", "0",
            "-x264-params",
            (
                "repeat-headers=1:"
                f"keyint={key_interval}:"
                f"min-keyint={key_interval}:"
                "scenecut=0"
            ),
            "-g", str(key_interval),
            "-force_key_frames", "expr:gte(t,n_forced*1)",
        ]
    return common + enc + [
        "-flush_packets", "1",
        "-f", "rtsp", "-rtsp_transport", "tcp", rtsp_url,
    ]


class RtspPublisher:
    """Per-stream FFmpeg RTSP publisher.

    Accepts a pre-computed use_hw flag (probed once in the orchestrator)
    to avoid 50 concurrent ffmpeg probe subprocesses crashing the amdgpu driver.
    """

    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        fps: int,
        encode_mode: str,
        use_hw: bool | None = None,  # pre-computed by orchestrator; None = probe locally
    ) -> None:
        self.rtsp_url = rtsp_url
        if use_hw is not None:
            self._use_hw = use_hw
        else:
            self._use_hw = encode_mode == "hardware" and _probe_vaapi_encode()
        if encode_mode == "hardware" and not self._use_hw:
            logger.warning(
                "VAAPI encode unavailable, falling back to libx264 for %s", rtsp_url
            )

        cmd = _build_ffmpeg_cmd(rtsp_url, width, height, fps, self._use_hw)
        self._proc: Optional[subprocess.Popen] = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._write_error: Exception | None = None
        self._frames: "queue.Queue[bytes | None]" = queue.Queue(maxsize=1)
        self._writer = threading.Thread(target=self._drain_frames, daemon=True)
        self._writer.start()
        self._stderr_reader = threading.Thread(
            target=self._log_stderr, args=(self._proc,), daemon=True
        )
        self._stderr_reader.start()

    def _log_stderr(self, proc: subprocess.Popen) -> None:
        """Forward ffmpeg stderr lines to the Python logger at WARNING level."""
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stripped = line.rstrip()
            if stripped:
                logger.warning(
                    "ffmpeg [%s]: %s",
                    self.rtsp_url,
                    stripped.decode("utf-8", errors="replace"),
                )

    def _drain_frames(self) -> None:
        while True:
            payload = self._frames.get()
            if payload is None:
                return
            if self._proc is None or self._proc.stdin is None:
                return
            try:
                self._proc.stdin.write(payload)
            except Exception as exc:
                self._write_error = exc
                return

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("publisher not started")
        if self._write_error is not None:
            raise RuntimeError("publisher write failed") from self._write_error
        payload = np.ascontiguousarray(frame).tobytes()
        # Drop the oldest frame if FFmpeg can't keep up
        try:
            self._frames.put_nowait(payload)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(payload)

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                self._frames.put_nowait(None)
            self._writer.join(timeout=2)
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

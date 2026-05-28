# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
RTSP source for the combined pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterator
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

logger = logging.getLogger(__name__)

_gst_child_ready = False
SOFTWARE_DECODER = "avdec_h264 direct-rendering=false"


@contextlib.contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Silence native VA-API/libdrm stderr noise during known-safe probes."""
    sys.stderr.flush()
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def init_gst_in_child() -> None:
    """Call once per spawned process before any GStreamer use."""
    global _gst_child_ready
    if not _gst_child_ready:
        Gst.init(None)
        _gst_child_ready = True


def _gst_element_exists(name: str) -> bool:
    """Check if a GStreamer element type is registered WITHOUT spawning a subprocess.

    Spawning gst-inspect-1.0 for VA-API elements probes the VA display, which
    causes GPU hangs on AMD when MIGraphX is running on the same machine.
    Instead, query the GStreamer registry in-process after Gst.init().
    """
    init_gst_in_child()
    registry = Gst.Registry.get()
    feature = registry.find_feature(name, Gst.ElementFactory)
    return feature is not None


def _set_vaapi_device(device: str) -> None:
    """Point VA-API at a specific DRM render node (e.g. /dev/dri/renderD129).

    Must be called before GStreamer initialises the VA-API plugin.
    Sets both LIBVA_DRM_DEVICE (new drivers) and DISPLAY/DRI_PRIME as fallback.
    """
    os.environ["LIBVA_DRM_DEVICE"] = device
    os.environ["GST_VAAPI_DRM_DEVICE"] = device
    # Some older libva builds read LIBVA_DEVICE_DRIVER_NAME instead; harmless to set.
    os.environ.setdefault("LIBVA_DRIVER_NAME", os.environ.get("LIBVA_DRIVER_NAME", "radeonsi"))
    os.environ.pop("LIBVA_DRIVERS_PATH", None)


def is_invalid_decoded_frame(frame: np.ndarray) -> bool:
    """Return whether a decoded RGB frame is unusable for YOLO inference."""
    if frame.size == 0:
        return True
    return int(frame.max()) <= 2 and float(frame.mean()) < 1.0


def pick_decoder(*, decode_mode: str, gpu_id: int, vaapi_device: str | None = None) -> str:
    """Return the best available H.264 decoder element for this process.

    Preference order when decode_mode == 'hardware':
      1. vaapih264dec with GST_VAAPI_DRM_DEVICE (validated on this stack)
      2. vah264dec (new VA plugin, guarded by first-frame validation)
      3. avdec_h264 (software fallback)

    vaapi_device: DRM render node path (e.g. '/dev/dri/renderD129').
                  Exported through VA-API environment variables before plugin init.
    """
    if decode_mode != "hardware":
        return SOFTWARE_DECODER

    if vaapi_device:
        _set_vaapi_device(vaapi_device)

    with _suppress_native_stderr():
        if _gst_element_exists("vaapih264dec"):
            return "vaapih264dec low-latency=true"
        if _gst_element_exists("vah264dec"):
            return "vah264dec"
    logger.warning("No VA-API H264 decoder available, falling back to software")
    return SOFTWARE_DECODER


def build_rtsp_pipeline(
    rtsp_url: str, decoder: str, width: int | None, height: int | None
) -> str:
    scale_caps = ""
    scale_chain = ""
    if width and height:
        scale_chain = "videoscale ! "
        scale_caps = f",width={width},height={height}"
    decode_chain = f"{decoder} ! "
    return (
        f"rtspsrc location={rtsp_url} latency=100 protocols=tcp ! "
        "rtph264depay ! h264parse ! "
        f"{decode_chain}videoconvert ! {scale_chain}"
        f"video/x-raw,format=RGB{scale_caps} ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    )


class RtspSource:
    def __init__(
        self,
        rtsp_url: str,
        decoder: str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.decoder = decoder
        self.width = width
        self.height = height
        self.error: str | None = None
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink = None
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending_frame: np.ndarray | None = None

    def _start_software_fallback(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self.decoder = SOFTWARE_DECODER
        pipeline_str = build_rtsp_pipeline(
            self.rtsp_url, self.decoder, self.width, self.height
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsink = self._pipeline.get_by_name("sink")
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"RTSP source failed (sw fallback): {self.rtsp_url}")

    def start(self) -> None:
        init_gst_in_child()
        pipeline_str = build_rtsp_pipeline(self.rtsp_url, self.decoder, self.width, self.height)
        logger.info("Starting pipeline: %s", pipeline_str[:120])
        try:
            with self._native_stderr_context():
                self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as exc:
            if self.decoder != SOFTWARE_DECODER:
                logger.warning(
                    "Failed to create pipeline with %s (%s), retrying with software decode",
                    self.decoder, exc,
                )
                self.decoder = SOFTWARE_DECODER
                pipeline_str = build_rtsp_pipeline(
                    self.rtsp_url, self.decoder, self.width, self.height
                )
                self._pipeline = Gst.parse_launch(pipeline_str)
            else:
                raise
        self._appsink = self._pipeline.get_by_name("sink")
        with self._native_stderr_context():
            ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            if self.decoder != SOFTWARE_DECODER:
                logger.warning(
                    "Hardware pipeline failed to start, falling back to software decode"
                )
                self._start_software_fallback()
            else:
                raise RuntimeError(f"RTSP source failed: {self.rtsp_url}")
        if self.decoder != SOFTWARE_DECODER:
            try:
                frame = self._pull_frame(timeout_seconds=15.0)
            except TimeoutError:
                logger.warning(
                    "Hardware decoder %s did not produce a validation frame; "
                    "falling back to software decode for %s",
                    self.decoder, self.rtsp_url,
                )
                self._start_software_fallback()
                frame = None
            if frame is not None and is_invalid_decoded_frame(frame):
                logger.warning(
                    "Hardware decoder %s produced an invalid first frame; "
                    "falling back to software decode for %s",
                    self.decoder, self.rtsp_url,
                )
                self._start_software_fallback()
            elif frame is not None:
                self._pending_frame = frame
        self._watcher = threading.Thread(target=self._watch_bus, daemon=True)
        self._watcher.start()

    def _native_stderr_context(self) -> contextlib.AbstractContextManager[None]:
        if self.decoder == SOFTWARE_DECODER:
            return contextlib.nullcontext()
        return _suppress_native_stderr()

    def _watch_bus(self) -> None:
        assert self._pipeline is not None
        bus = self._pipeline.get_bus()
        while not self._stop.is_set():
            msg = bus.timed_pop_filtered(
                200 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if msg is None:
                continue
            if msg.type == Gst.MessageType.EOS:
                continue
            err, dbg = msg.parse_error()
            self.error = f"{err.message} ({dbg or 'no debug'})"
            self._stop.set()

    def _pull_frame(self, timeout_seconds: float = 5.0) -> np.ndarray:
        if self._appsink is None:
            raise RuntimeError("appsink not ready")
        timeout_ns = int(timeout_seconds * Gst.SECOND)
        if hasattr(self._appsink, "try_pull_sample"):
            sample = self._appsink.try_pull_sample(timeout_ns)
        else:
            sample = self._appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            raise TimeoutError(f"timeout {self.rtsp_url}")
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        buf = sample.get_buffer()
        ok, mapped = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("buffer map failed")
        try:
            return np.ndarray(shape=(h, w, 3), dtype=np.uint8, buffer=mapped.data).copy()
        finally:
            buf.unmap(mapped)

    def read(self, timeout_seconds: float = 5.0) -> np.ndarray:
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame
        frame = self._pull_frame(timeout_seconds)
        if self.decoder != SOFTWARE_DECODER and is_invalid_decoded_frame(frame):
            raise RuntimeError(f"hardware decoder produced invalid frame: {self.rtsp_url}")
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._watcher is not None:
            self._watcher.join(timeout=2)

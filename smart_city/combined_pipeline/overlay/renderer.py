# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Renderer for the combined pipeline.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from combined_pipeline.inference.postprocess import COCO_NAMES
from combined_pipeline.ipc.messages import Detection

# High-contrast RGB colours for the two surveillance classes.
# Person: bright cyan-green (stands out on most backgrounds)
# Car: vivid orange-red (easily distinguishable from person)
_CLASS_COLOR_RGB: dict[int, tuple[int, int, int]] = {
    0: (  0, 230, 118),   # person — vivid spring green
    2: (255,  80,   0),   # car    — vivid orange-red
}

def _class_color_rgb(class_id: int) -> tuple[int, int, int]:
    if class_id in _CLASS_COLOR_RGB:
        return _CLASS_COLOR_RGB[class_id]
    hue = int((class_id * 137) % 180)
    hsv = np.array([[[hue, 240, 220]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _decode_mask(
    d: Detection, frame_h: int, frame_w: int
) -> tuple[int, int, int, int, np.ndarray] | None:
    """Decompress and reshape the per-pixel binary mask from IPC bytes."""
    if not d.mask_bytes:
        return None
    x1 = max(0, int(d.x1))
    y1 = max(0, int(d.y1))
    x2 = min(frame_w, int(d.x2) + 1)
    y2 = min(frame_h, int(d.y2) + 1)
    bh, bw = y2 - y1, x2 - x1
    if bh <= 0 or bw <= 0:
        return None
    try:
        raw = np.frombuffer(zlib.decompress(d.mask_bytes), dtype=np.uint8).reshape(bh, bw)
    except (zlib.error, ValueError):
        return None
    return x1, y1, x2, y2, raw


def draw_detections(
    frame_rgb: np.ndarray,
    detections: list[Detection],
    *,
    count_label: str | None = None,
) -> None:
    """In-place draw segmentation masks on uint8 HWC RGB frame.

    Uses per-pixel blending on the decoded binary mask. No polygon round-trip
    means no loss of boundary detail on irregular shapes.
    """
    if not detections:
        if count_label:
            cv2.putText(
                frame_rgb, count_label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return

    fh, fw = frame_rgb.shape[:2]

    # Decode all masks once; skip detections with no mask data.
    decoded = [_decode_mask(d, fh, fw) for d in detections]

    # Pass 1: fill all masks on an overlay then blend at 65% opacity.
    overlay = frame_rgb.copy()
    for d, info in zip(detections, decoded):
        if info is None:
            continue
        x1, y1, x2, y2, raw = info
        color = _class_color_rgb(d.class_id)
        overlay[y1:y2, x1:x2][raw > 0] = color

    cv2.addWeighted(overlay, 0.65, frame_rgb, 0.35, 0, dst=frame_rgb)

    # Pass 2: crisp outlines + class labels at full opacity.
    for d, info in zip(detections, decoded):
        color = _class_color_rgb(d.class_id)
        label = COCO_NAMES.get(d.class_id, str(d.class_id))

        if info is not None:
            # Segmentation mask path: draw contour outline
            x1, y1, x2, y2, raw = info
            contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                offset = np.array([[[x1, y1]]])
                shifted = [c + offset for c in contours]
                cv2.drawContours(frame_rgb, shifted, -1, color, 2, cv2.LINE_AA)
            lx, ly = x1, max(y1 - 4, 14)
        else:
            # Detection-only path: draw a filled semi-transparent bbox + solid border
            bx1, by1 = max(0, int(d.x1)), max(0, int(d.y1))
            bx2, by2 = min(frame_rgb.shape[1], int(d.x2)), min(frame_rgb.shape[0], int(d.y2))
            if bx2 > bx1 and by2 > by1:
                roi = frame_rgb[by1:by2, bx1:bx2]
                bbox_overlay = roi.copy()
                bbox_overlay[:] = color
                cv2.addWeighted(bbox_overlay, 0.30, roi, 0.70, 0, dst=roi)
                cv2.rectangle(frame_rgb, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
            lx, ly = bx1, max(by1 - 4, 14)

        cv2.putText(frame_rgb, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_rgb, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if count_label:
        cv2.putText(
            frame_rgb, count_label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )


def compute_density_heatmap_rgb(
    density_map: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    """Build the HWC uint8 RGB heatmap for a density map at the requested size.

    Split out from make_density_frame so the expensive normalise→gamma→blur→
    resize→colormap pipeline can be cached per-stream and reused across the
    several YOLO frames that share the same density_map (DM-Count typically
    runs 3-5× slower than YOLO at 50 streams).
    """
    dm = np.maximum(density_map, 0).astype(np.float32)

    # Percentile normalisation: scale by the 99th percentile of non-zero values
    # so the colormap always spans the actual signal range regardless of how
    # small the absolute DM output is.  This prevents flat-blue panels on sparse
    # scenes caused by domain-shift undercounting.
    dm_positive = dm[dm > 1e-9]
    if dm_positive.size > 0:
        ceiling = float(np.percentile(dm_positive, 99))
        if ceiling > 1e-9:
            dm = (dm / ceiling).clip(0.0, 1.0)
        else:
            dm = np.zeros_like(dm)
    else:
        dm = np.zeros_like(dm)

    # Gamma stretch: push mid-range values upward so hotspots appear more
    # vibrant without clipping peaks (gamma < 1 brightens the midrange).
    dm = np.power(dm, 0.6)

    # Light Gaussian blur on the native (small) map before upscaling so cell
    # boundaries blend smoothly instead of producing hard pixel-grid blocks.
    dm = cv2.GaussianBlur(dm, (0, 0), sigmaX=1.2)

    # Upsample with INTER_CUBIC for smoother spatial field reconstruction.
    dm_resized = cv2.resize(dm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    dm_resized = dm_resized.clip(0.0, 1.0)

    # COLORMAP_JET: blue→cyan→green→yellow→red for the thermal surveillance look.
    dm_uint8 = (dm_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_JET)
    return np.ascontiguousarray(heatmap_bgr[:, :, ::-1])


def blend_heatmap_over_frame(
    frame_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a precomputed heatmap_rgb panel over a copy of frame_rgb."""
    out = frame_rgb.copy()
    cv2.addWeighted(out, 1.0 - alpha, heatmap_rgb, alpha, 0, dst=out)
    return out


def make_density_frame(
    frame_rgb: np.ndarray,
    density_map: np.ndarray,
    count: float,
    alpha: float = 0.55,
    norm_max: float | None = None,
) -> np.ndarray:
    """Return a NEW frame with the DM-Count heatmap blended over a copy of frame_rgb.

    Uses a stronger alpha (0.55) so the heatmap is clearly visible as a separate view.
    Thin wrapper around the split helpers; kept for backward compatibility with
    callers that don't want to manage a per-stream heatmap cache themselves.

    Args:
        frame_rgb: HWC uint8 RGB source frame.
        density_map: 2-D float32 density output from DM-Count.
        count: Smoothed scalar crowd count for the label.
        alpha: Heatmap blend weight.
        norm_max: Unused; kept for call-site compatibility.

    Returns:
        New HWC uint8 RGB frame with heatmap overlay and crowd label.
    """
    h, w = frame_rgb.shape[:2]
    heatmap_rgb = compute_density_heatmap_rgb(density_map, h, w)
    return blend_heatmap_over_frame(frame_rgb, heatmap_rgb, alpha)


def blend_density_heatmap(
    frame_rgb: np.ndarray,
    density_map: np.ndarray,
    count: float,
    alpha: float = 0.30,
) -> None:
    """In-place blend DM-Count heatmap (legacy combined-view helper)."""
    blended = make_density_frame(frame_rgb, density_map, count, alpha=alpha)
    np.copyto(frame_rgb, blended)

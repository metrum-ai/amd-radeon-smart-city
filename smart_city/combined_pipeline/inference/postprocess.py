# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Postprocess for the combined pipeline.
"""

from __future__ import annotations

import logging
import os
import zlib

import cv2
import numpy as np

from combined_pipeline.ipc.messages import Detection

logger = logging.getLogger(__name__)

# Diagnostic only: logs candidate count/score distribution before NMS.
_DEBUG_CANDIDATE_COUNT = os.environ.get("YOLO_DEBUG_CANDIDATES", "0") == "1"

# COCO classes kept for surveillance: person + cars only.
# Bicycles, motorcycles, buses, trucks are excluded — they produce the most
# false positives in crowd scenes and are not the primary objects of interest.
_KEEP_CLASSES: frozenset[int] = frozenset({0, 2})

COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    x = np.clip(x.astype(np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Vectorized N×N IoU matrix. boxes: (N, 4) xyxy float32.

    Replaces the per-pair _iou_xyxy Python loop: one numpy broadcast op
    computes all pairwise IoUs at once, eliminating O(N²) Python calls.
    """
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS; boxes (N,4) xyxy, scores (N,).

    Same result as the old N x N IoU matrix, but IoU is evaluated only against
    survivors: N is pre-cap, so crowded frames hit hundreds. Metrum AI.
    """
    if boxes.size == 0:
        return []
    b = boxes.astype(np.float32, copy=False)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(b[i, 0], b[rest, 0])
        y1 = np.maximum(b[i, 1], b[rest, 1])
        x2 = np.minimum(b[i, 2], b[rest, 2])
        y2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = area[i] + area[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo_detect_batch(
    output0: np.ndarray,
    *,
    infer_width: int,
    infer_height: int,
    full_width: int,
    full_height: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[list[Detection]]:
    """Decode raw ONNX output from a detect-only YOLO model (no mask branch).

    output0: (batch, anchors, 4+nc) or (batch, 4+nc, anchors).
    Skips all mask processing — pure bbox + class decode.
    This is significantly faster than the seg variant.
    """
    if output0.ndim != 3:
        raise ValueError(f"output0 must be 3D, got {output0.shape}")

    b, d1, d2 = output0.shape
    # Reduce over the native layout: transposing first makes the class block a
    # strided view that astype must gather-copy. Engineering by Metrum AI.
    nf_first = d1 < d2
    nf = d1 if nf_first else d2
    nc = nf - 4  # detect head: 4 bbox coords + nc classes
    if nc < 1:
        raise ValueError(f"Invalid feature count {nf} for YOLO-detect head")

    sx = full_width / float(infer_width)
    sy = full_height / float(infer_height)
    red_axis = 0 if nf_first else 1

    batch_out: list[list[Detection]] = []
    for bi in range(b):
        p = output0[bi]
        cls_block = p[4:, :] if nf_first else p[:, 4:]
        # This export's ONNX graph already applies sigmoid to class scores
        # internally — applying it again here double-sigmoids them, collapsing them into a narrow band near 0.5.
        scores_all = cls_block.max(axis=red_axis).astype(np.float32, copy=False)

        if _DEBUG_CANDIDATE_COUNT:
            classes_all = cls_block.argmax(axis=red_axis)
            raw_kept_mask = np.isin(classes_all, list(_KEEP_CLASSES))
            raw_scores = scores_all[raw_kept_mask]
            logger.warning(
                "DIAG score distribution (person/car anchors, %d total): "
                ">=0.25: %d  >=0.40: %d  >=0.50: %d  >=0.60: %d  >=0.70: %d  >=0.80: %d",
                len(raw_scores),
                int((raw_scores >= 0.25).sum()), int((raw_scores >= 0.40).sum()),
                int((raw_scores >= 0.50).sum()), int((raw_scores >= 0.60).sum()),
                int((raw_scores >= 0.70).sum()), int((raw_scores >= 0.80).sum()),
            )

        # Threshold first, then argmax only survivors: argmax over the full grid
        # cost ~40 ms/batch and >99% is discarded next line. Metrum AI.
        cand = np.flatnonzero(scores_all >= conf_threshold)
        if cand.size == 0:
            batch_out.append([])
            continue

        sub = cls_block[:, cand] if nf_first else cls_block[cand, :]
        classes = sub.argmax(axis=red_axis)
        keep_cls = np.isin(classes, list(_KEEP_CLASSES))
        if not keep_cls.any():
            batch_out.append([])
            continue

        cand = cand[keep_cls]
        classes = classes[keep_cls]
        scores = scores_all[cand]
        # Gather boxes for surviving anchors only, rather than copying all of them.
        boxes = (p[:4, :].T[cand] if nf_first else p[cand, :4]).astype(np.float32)

        if _DEBUG_CANDIDATE_COUNT:
            logger.warning("DIAG decode_yolo_detect_batch: %d candidates pre-NMS", len(boxes))

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2.0) * sx
        y1 = (cy - h / 2.0) * sy
        x2 = (cx + w / 2.0) * sx
        y2 = (cy + h / 2.0) * sy
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, full_width - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, full_height - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, full_width - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, full_height - 1)

        keep = nms_xyxy(xyxy, scores, iou_threshold)
        keep = keep[:max_detections]

        dets: list[Detection] = []
        for idx in keep:
            dets.append(Detection(
                x1=float(xyxy[idx, 0]),
                y1=float(xyxy[idx, 1]),
                x2=float(xyxy[idx, 2]),
                y2=float(xyxy[idx, 3]),
                class_id=int(classes[idx]),
                score=float(scores[idx]),
                mask_bytes=None,
            ))
        batch_out.append(dets)
    return batch_out


# ---------------------------------------------------------------------------
# Decoders for GPU-pre-processed YOLO output (TopK + box decode already done)
#
# Every YOLO ONNX model in this project bakes TopK candidate selection,
# cxcywh→xyxy decode, sigmoid on class scores, and class-id extraction into
# the ONNX graph itself (last ~15 nodes).  What arrives on CPU per batch is
# already a compact tensor — only confidence filtering and NMS remain on CPU.
#
# Detect format: (batch, K, 6)  → [x1, y1, x2, y2, conf, class_id]
# Seg    format: (batch, K, 38) → [x1, y1, x2, y2, conf, class_id, *32_coeffs]
#
# Boxes are in infer_width × infer_height pixel coordinates.
# The decoders scale them to full_width × full_height before returning.
# ---------------------------------------------------------------------------

def decode_detect_topk(
    output0: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    infer_width: int,
    infer_height: int,
    full_width: int,
    full_height: int,
) -> list[list[Detection]]:
    """Decode (batch, K, 6) YOLO detect output [x1,y1,x2,y2,conf,class_id].

    Applies confidence filter, surveillance-class filter, and vectorized NMS.
    No sigmoid or cxcywh→xyxy needed — the model's GPU graph already did both.
    """
    b = output0.shape[0]
    sx = full_width / float(infer_width)
    sy = full_height / float(infer_height)

    batch_out: list[list[Detection]] = []
    for bi in range(b):
        rows = output0[bi].astype(np.float32)   # (K, 6)
        confs = rows[:, 4]
        cls_ids = rows[:, 5].astype(np.int32)

        keep_mask = (confs >= conf_threshold) & np.isin(cls_ids, list(_KEEP_CLASSES))
        rows = rows[keep_mask]
        confs = confs[keep_mask]
        cls_ids = cls_ids[keep_mask]

        if len(rows) == 0:
            batch_out.append([])
            continue

        xyxy = rows[:, :4].copy()
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]] * sx, 0.0, full_width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]] * sy, 0.0, full_height - 1)

        keep = nms_xyxy(xyxy, confs, iou_threshold)
        keep = keep[:max_detections]

        batch_out.append([
            Detection(
                x1=float(xyxy[i, 0]), y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]), y2=float(xyxy[i, 3]),
                class_id=int(cls_ids[i]), score=float(confs[i]),
                mask_bytes=None,
            )
            for i in keep
        ])
    return batch_out


def decode_seg_topk(
    output0: np.ndarray,
    output1: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    infer_width: int,
    infer_height: int,
    full_width: int,
    full_height: int,
) -> list[list[Detection]]:
    """Decode (batch, K, 38) YOLO seg output [x1,y1,x2,y2,conf,class_id,*32_coeffs].

    output1: prototype masks (batch, 32, mh, mw).  The mask coefficient ×
    prototype matmul is batched across all surviving candidates in one numpy
    call before per-detection compression, avoiding a Python loop per detection.
    """
    b = output0.shape[0]
    sx = full_width / float(infer_width)
    sy = full_height / float(infer_height)

    batch_out: list[list[Detection]] = []
    for bi in range(b):
        rows = output0[bi].astype(np.float32)   # (K, 38)
        confs = rows[:, 4]
        cls_ids = rows[:, 5].astype(np.int32)

        keep_mask = (confs >= conf_threshold) & np.isin(cls_ids, list(_KEEP_CLASSES))
        rows = rows[keep_mask]
        confs = confs[keep_mask]
        cls_ids = cls_ids[keep_mask]

        if len(rows) == 0:
            batch_out.append([])
            continue

        xyxy = rows[:, :4].copy()
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]] * sx, 0.0, full_width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]] * sy, 0.0, full_height - 1)
        mask_coeffs = rows[:, 6:]   # (n_cands, 32)

        keep = nms_xyxy(xyxy, confs, iou_threshold)
        keep = keep[:max_detections]

        proto = output1[bi]  # (32, mh, mw)
        mh, mw = proto.shape[1], proto.shape[2]
        kept_coeffs = mask_coeffs[keep]  # (n_kept, 32)

        # Batch matmul: (n_kept, 32) @ (32, mh*mw) → (n_kept, mh*mw)
        # Avoids a Python loop over kept detections for the matmul step.
        flat_masks = _sigmoid(kept_coeffs @ proto.reshape(32, -1))  # (n_kept, mh*mw)

        dets: list[Detection] = []
        for rank, i in enumerate(keep):
            x1f, y1f = float(xyxy[i, 0]), float(xyxy[i, 1])
            x2f, y2f = float(xyxy[i, 2]), float(xyxy[i, 3])

            # Crop proto-space region before upsampling to prevent mask bleed.
            px = mw / float(full_width)
            py = mh / float(full_height)
            x1p = max(0, int(x1f * px))
            y1p = max(0, int(y1f * py))
            x2p = min(mw, int(x2f * px) + 1)
            y2p = min(mh, int(y2f * py) + 1)
            mask_map = flat_masks[rank].reshape(mh, mw)
            cropped = np.zeros((mh, mw), dtype=np.float32)
            cropped[y1p:y2p, x1p:x2p] = mask_map[y1p:y2p, x1p:x2p]

            full_float = cv2.resize(cropped, (full_width, full_height), interpolation=cv2.INTER_LINEAR)

            bx1 = max(0, int(x1f))
            by1 = max(0, int(y1f))
            bx2 = min(full_width, int(x2f) + 1)
            by2 = min(full_height, int(y2f) + 1)
            mask_bytes: bytes | None = None
            if bx2 > bx1 and by2 > by1:
                binary = (full_float[by1:by2, bx1:bx2] > 0.45).astype(np.uint8)
                if binary.any():
                    mask_bytes = zlib.compress(binary.tobytes(), level=1)

            dets.append(Detection(
                x1=x1f, y1=y1f, x2=x2f, y2=y2f,
                class_id=int(cls_ids[i]), score=float(confs[i]),
                mask_bytes=mask_bytes,
            ))
        batch_out.append(dets)
    return batch_out

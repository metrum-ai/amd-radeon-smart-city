# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Postprocess for the combined pipeline.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from combined_pipeline.ipc.messages import Detection

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

    IoU matrix is computed once via _iou_matrix (vectorized). The outer loop
    iterates at most N times (N ≤ max_detections) with O(1) numpy ops each —
    no Python-level per-pair IoU calls.
    """
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1].tolist()
    iou = _iou_matrix(boxes.astype(np.float32))
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > iou_threshold
    return keep


def num_classes_from_features(n_features: int) -> int:
    """Get the number of classes from features."""
    nc = n_features - 4 - 32
    if nc < 1:
        raise ValueError(f"Invalid feature count {n_features} for YOLO-seg head")
    return nc


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
    if d1 < d2:
        pred = np.transpose(output0, (0, 2, 1))
    else:
        pred = output0

    nf = pred.shape[2]
    nc = nf - 4  # detect head: 4 bbox coords + nc classes
    if nc < 1:
        raise ValueError(f"Invalid feature count {nf} for YOLO-detect head")

    sx = full_width / float(infer_width)
    sy = full_height / float(infer_height)

    batch_out: list[list[Detection]] = []
    for bi in range(b):
        p = pred[bi]
        boxes = p[:, :4].astype(np.float32)
        cls_logits = p[:, 4:].astype(np.float32)
        scores = _sigmoid(cls_logits).max(axis=1)
        classes = cls_logits.argmax(axis=1)

        conf_mask = scores >= conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        classes = classes[conf_mask]

        keep_cls = np.isin(classes, list(_KEEP_CLASSES))
        boxes = boxes[keep_cls]
        scores = scores[keep_cls]
        classes = classes[keep_cls]

        if boxes.size == 0:
            batch_out.append([])
            continue

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


def _encode_mask(
    mask_coeff: np.ndarray,
    protos: np.ndarray,
    box_xyxy_full: np.ndarray,
    full_width: int,
    full_height: int,
) -> bytes | None:
    """Return zlib-compressed uint8 binary mask cropped to the detection bbox.

    The renderer decodes, reshapes to (y2-y1, x2-x1), and blends per-pixel —
    identical to how Ultralytics renders masks internally. No polygon conversion
    means no loss of boundary detail.
    """
    mh, mw = protos.shape[1], protos.shape[2]
    flat = mask_coeff @ protos.reshape(32, -1)
    mask_map = _sigmoid(flat).reshape(mh, mw)

    # Zero outside bbox in proto space before upsampling to prevent bleed.
    sx = mw / float(full_width)
    sy = mh / float(full_height)
    x1p = max(0, int(box_xyxy_full[0] * sx))
    y1p = max(0, int(box_xyxy_full[1] * sy))
    x2p = min(mw, int(box_xyxy_full[2] * sx) + 1)
    y2p = min(mh, int(box_xyxy_full[3] * sy) + 1)
    cropped = np.zeros((mh, mw), dtype=np.float32)
    cropped[y1p:y2p, x1p:x2p] = mask_map[y1p:y2p, x1p:x2p]

    # Bilinear upsample to full resolution for smooth sub-pixel boundaries.
    full_float = cv2.resize(cropped, (full_width, full_height), interpolation=cv2.INTER_LINEAR)

    # Crop to the detection bbox before compressing to keep payload tiny.
    x1 = max(0, int(box_xyxy_full[0]))
    y1 = max(0, int(box_xyxy_full[1]))
    x2 = min(full_width, int(box_xyxy_full[2]) + 1)
    y2 = min(full_height, int(box_xyxy_full[3]) + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    binary = (full_float[y1:y2, x1:x2] > 0.45).astype(np.uint8)
    if not binary.any():
        return None

    # level=1 is fast; sparse binary arrays compress very well (~5-20× ratio).
    return zlib.compress(binary.tobytes(), level=1)


def decode_yolo_seg_batch(
    output0: np.ndarray,
    *,
    infer_width: int,
    infer_height: int,
    full_width: int,
    full_height: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    output1: np.ndarray | None = None,
) -> list[list[Detection]]:
    """
    output0: (batch, anchors, features) or (batch, features, anchors).
    output1: prototype masks (batch, 32, mh, mw) — optional, enables segmentation polygons.
    """
    if output0.ndim != 3:
        raise ValueError(f"output0 must be 3D, got {output0.shape}")

    b, d1, d2 = output0.shape
    if d1 < d2:
        pred = np.transpose(output0, (0, 2, 1))
    else:
        pred = output0

    nf = pred.shape[2]
    nc = num_classes_from_features(nf)

    sx = full_width / float(infer_width)
    sy = full_height / float(infer_height)

    batch_out: list[list[Detection]] = []
    for bi in range(b):
        p = pred[bi]
        boxes = p[:, :4].astype(np.float32)
        cls_logits = p[:, 4 : 4 + nc].astype(np.float32)
        scores = _sigmoid(cls_logits).max(axis=1)
        classes = cls_logits.argmax(axis=1)

        conf_mask = scores >= conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        classes = classes[conf_mask]
        mask_coeffs = p[conf_mask, 4 + nc :].astype(np.float32)

        keep_cls = np.isin(classes, list(_KEEP_CLASSES))
        boxes = boxes[keep_cls]
        scores = scores[keep_cls]
        classes = classes[keep_cls]
        mask_coeffs = mask_coeffs[keep_cls]

        if boxes.size == 0:
            batch_out.append([])
            continue

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

        protos = output1[bi] if output1 is not None else None

        dets: list[Detection] = []
        for idx in keep:
            mask_bytes: bytes | None = None
            if protos is not None:
                mask_bytes = _encode_mask(
                    mask_coeffs[idx], protos, xyxy[idx], full_width, full_height
                )
            dets.append(
                Detection(
                    x1=float(xyxy[idx, 0]),
                    y1=float(xyxy[idx, 1]),
                    x2=float(xyxy[idx, 2]),
                    y2=float(xyxy[idx, 3]),
                    class_id=int(classes[idx]),
                    score=float(scores[idx]),
                    mask_bytes=mask_bytes,
                )
            )
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

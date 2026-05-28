# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
IPC messages for the combined pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InferJob:
    """Infer job."""
    stream_id: int
    seq: int
    full_slot: int
    infer_slot: int
    width: int
    height: int
    ts_ns: int


@dataclass(slots=True)
class Detection:
    """Detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float
    mask_bytes: bytes | None = None


@dataclass(slots=True)
class InferResult:
    """Infer result."""
    stream_id: int
    seq: int
    full_slot: int
    infer_slot: int
    width: int
    height: int
    detections: list[Detection]
    count: int
    ts_ns: int = 0  # echoed from InferJob for end-to-end latency measurement


def result_to_payload(r: InferResult) -> dict[str, Any]:
    """Convert an InferResult to a payload."""
    return {
        "stream_id": r.stream_id,
        "seq": r.seq,
        "full_slot": r.full_slot,
        "infer_slot": r.infer_slot,
        "width": r.width,
        "height": r.height,
        "count": r.count,
        "ts_ns": r.ts_ns,
        "detections": [
            {
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
                "class_id": d.class_id,
                "score": d.score,
                "mask_bytes": d.mask_bytes,
            }
            for d in r.detections
        ],
    }


def payload_to_result(p: dict[str, Any]) -> InferResult:
    """Convert a payload to an InferResult."""
    dets = [
        Detection(
            x1=float(x["x1"]),
            y1=float(x["y1"]),
            x2=float(x["x2"]),
            y2=float(x["y2"]),
            class_id=int(x["class_id"]),
            score=float(x["score"]),
            mask_bytes=x.get("mask_bytes"),
        )
        for x in p["detections"]
    ]
    return InferResult(
        stream_id=int(p["stream_id"]),
        seq=int(p["seq"]),
        full_slot=int(p["full_slot"]),
        infer_slot=int(p["infer_slot"]),
        width=int(p["width"]),
        height=int(p["height"]),
        detections=dets,
        count=int(p["count"]),
        ts_ns=int(p.get("ts_ns", 0)),
    )

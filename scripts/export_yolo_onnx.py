#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Export an Ultralytics YOLO model to FP16 ONNX for MIGraphX inference.

This is the conversion step that turns the upstream Ultralytics ``.pt`` weights
into the ``.onnx`` file the pipeline loads via ORT + MIGraphXExecutionProvider.
We do not redistribute YOLO weights. The Ultralytics package downloads the
``.pt`` file from its CDN on first use, then this script exports to the exact
shape, precision, and opset the rest of the system expects.

Defaults match the production pipeline:

  * model:  yolo26s.pt
  * output: smart_city/models/yolo26s-384-dynamic.onnx
  * imgsz:  384x288  (matches inference resolution; saves a cv2.resize)
  * half:   True     (FP16 doubles RDNA4 throughput)
  * dynamic: True    (dynamic batch axis lets MIGraphX batch up to 64 frames)
  * opset:  13       (required for MIGraphX 1.23.2; opset 17 breaks reshape)
  * simplify: False  (MIGraphX dislikes some simplifier passes)

Usage::

    python scripts/export_yolo_onnx.py
    python scripts/export_yolo_onnx.py --model yolov26n.pt \\
        --output smart_city/models/yolo26n-384-dynamic.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def export_fp16(
    model_path: str,
    output_path: str,
    width: int,
    height: int,
    dynamic: bool,
    opset: int,
) -> None:
    """Export the YOLO model to FP16 ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: ultralytics is not installed.\n"
            "  Install with:  pip install ultralytics\n"
            f"  Underlying ImportError: {exc}\n"
        )
        sys.exit(2)

    print(f"Loading {model_path} (will auto-download from Ultralytics if missing) ...")
    model = YOLO(model_path)

    batch_desc = "dynamic batch" if dynamic else "static batch_size=1"
    print(f"Exporting FP16 ONNX at {width}x{height} ({batch_desc}, opset {opset}) ...")
    model.export(
        format="onnx",
        imgsz=[height, width],
        half=True,
        dynamic=dynamic,
        simplify=False,
        opset=opset,
    )

    auto_path = os.path.splitext(model_path)[0] + ".onnx"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if auto_path != str(out):
        os.rename(auto_path, out)
        print(f"Moved {auto_path} -> {out}")
    else:
        print(f"Saved to {out}")

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done. {out} ({size_mb:.1f} MB)")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export YOLO to FP16 ONNX at inference resolution",
    )
    parser.add_argument("--model", default="yolo26s.pt", help="Source .pt model")
    parser.add_argument(
        "--output",
        default="smart_city/models/yolo26s-384-dynamic.onnx",
        help="Output .onnx path (default: smart_city/models/yolo26s-384-dynamic.onnx)",
    )
    parser.add_argument("--width", type=int, default=384, help="Inference width")
    parser.add_argument("--height", type=int, default=288, help="Inference height")
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        default=True,
        help="Export with static batch_size=1 (debugging only)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13, required for MIGraphX 1.23.2)",
    )
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        print(f"{out} already exists — skipping export.")
        return

    export_fp16(args.model, args.output, args.width, args.height, args.dynamic, args.opset)


if __name__ == "__main__":
    main()

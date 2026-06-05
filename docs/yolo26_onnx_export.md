# YOLOv26 ONNX Model Setup

This repository does not install or vendor `ultralytics`. Before running setup, you must provide the YOLOv26 ONNX model locally at:

```text
smart_city/models/yolo26s-384-dynamic.onnx
```

## 1. Review The License

Before downloading or converting a YOLOv26 model with `ultralytics`, confirm that your use case is licensed appropriately.

The `ultralytics` license page states that community YOLO code and trained models are covered by AGPL-3.0 by default, and that commercial or proprietary use without open-sourcing the full project requires an enterprise license.

Source: https://www.ultralytics.com/license

## 2. Create A Temporary Export Environment

Run these commands outside the product dependency setup. `/tmp` is used here so the export tooling does not become part of the application environment:

```bash
python3 -m venv /tmp/yolo26-export
/tmp/yolo26-export/bin/python -m pip install --upgrade pip
/tmp/yolo26-export/bin/python -m pip install ultralytics onnx onnxruntime
```

## 3. Download The YOLOv26 Model

Download the YOLOv26 source model into a temporary local folder. This keeps the `.pt` file outside the repository:

```bash
mkdir -p /tmp/yolo26-models
curl -L \
  -o /tmp/yolo26-models/yolo26s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

If your organization provides a licensed internal copy, use that file instead and keep it outside the repository, for example:

```text
/tmp/yolo26-models/yolo26s.pt
```

## 4. Create The Export Helper

Save the following script outside the repository, for example at
`/tmp/export_yolo26_onnx.py`:

```python
#!/usr/bin/env python3
"""Export your YOLOv26 .pt model to the ONNX file expected by the app."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def export_model(
    model_path: str,
    output_path: str,
    width: int,
    height: int,
    opset: int,
) -> None:
    """Export YOLOv26 weights to ONNX."""
    model = YOLO(model_path)
    model.export(
        format="onnx",
        imgsz=[height, width],
        half=True,
        dynamic=True,
        simplify=False,
        opset=opset,
    )

    generated_path = os.path.splitext(model_path)[0] + ".onnx"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if generated_path != str(output):
        os.replace(generated_path, output)


def main() -> None:
    """Parse arguments and export the model."""
    parser = argparse.ArgumentParser(description="Export YOLOv26 to ONNX.")
    parser.add_argument("--model", required=True, help="Path to your .pt file")
    parser.add_argument(
        "--output",
        default="smart_city/models/yolo26s-384-dynamic.onnx",
        help="Where to write the ONNX file",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    export_model(args.model, args.output, args.width, args.height, args.opset)


if __name__ == "__main__":
    main()
```

## 5. Export Into The App Model Path

From the repository root, run:

```bash
/tmp/yolo26-export/bin/python /tmp/export_yolo26_onnx.py \
  --model /tmp/yolo26-models/yolo26s.pt \
  --output smart_city/models/yolo26s-384-dynamic.onnx
```

Use a different `--model` path if your licensed `.pt` file is stored elsewhere.

## 6. Check The Generated ONNX

Confirm the generated ONNX exists at the path expected by setup and runtime:

```bash
ls -lh smart_city/models/yolo26s-384-dynamic.onnx
```

Optionally verify that ONNX Runtime can load it:

```bash
/tmp/yolo26-export/bin/python - <<'PY'
import onnxruntime as ort

session = ort.InferenceSession(
    "smart_city/models/yolo26s-384-dynamic.onnx",
    providers=["CPUExecutionProvider"],
)
print("inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])
PY
```

Then continue with the normal setup flow in [README.md](../README.md#deploying-the-solution). `setup.sh` fails fast if this ONNX file is missing.

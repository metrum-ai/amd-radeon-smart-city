# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

LABEL maintainer="smartcity-public-safety"
LABEL description="YOLO + DM-Count analytics pipeline (AMD ROCm)"

WORKDIR /app

# GStreamer + VA-API + Python GI bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-vaapi \
    libva2 \
    libva-drm2 \
    mesa-va-drivers \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libgstreamer1.0-0 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --home-dir /app \
        --shell /usr/sbin/nologin --no-create-home --no-log-init appuser \
    && mkdir -p /pipeline_stats \
    && chown -R 10001:10001 /app /pipeline_stats

# Link PyGObject into the venv site-packages (ROCm image uses a venv)
RUN VENV_SP=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    ln -sf /usr/lib/python3/dist-packages/gi "$VENV_SP/gi" && \
    python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print(Gst.version_string())"

# Python dependencies — onnxruntime-rocm (MIGraphX) + torchvision
RUN pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm 2>/dev/null || true && \
    pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "pyyaml>=6,<7" \
        "opencv-python-headless>=4.9,<5" \
        "torchvision" \
        "onnx>=1.16,<2" && \
    pip install --no-cache-dir \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# Pipeline source — copied from smart_city/combined_pipeline/ into /app/combined_pipeline/
# so that "from combined_pipeline.*" imports resolve with PYTHONPATH=/app
COPY --chown=10001:10001 smart_city/combined_pipeline/ /app/combined_pipeline/

# Models directory: weights (gitignored) + Python architecture code
COPY --chown=10001:10001 smart_city/models/ /app/models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GST_VAAPI_ALL_DRIVERS=1 \
    HSA_OVERRIDE_GFX_VERSION=12.0.1 \
    LIBVA_DRIVER_NAME=radeonsi \
    YOLO_AUTOINSTALL=False

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD test -f /pipeline_stats/pipeline_stats.json || exit 1

CMD ["python", "-m", "combined_pipeline.main"]

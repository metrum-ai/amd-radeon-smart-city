# Created by Metrum AI for AMD

# ============================================================
# Smart City Public Safety Platform
# Base: AMD ROCm PyTorch – ROCm 7.1.1, Python 3.11, PyTorch 2.10.0
# Torch and torchvision are pre-installed in the base image;
# only application-level dependencies are installed here.
# ============================================================
FROM rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0

LABEL maintainer="smart-city-team"
LABEL description="Smart City Public Safety Platform"
LABEL version="1.0.0"

# GStreamer, OpenGL runtime, Python GObject bindings,
# and WeasyPrint PDF rendering dependencies (libpango + fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
        libglib2.0-0 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-good1.0-0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-rtsp \
        libgl1-mesa-glx \
        libgomp1 \
        curl \
        python3-gi \
        python3-dev \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers

# Copy requirements first to maximise layer cache reuse
COPY smart_city/requirements.txt ./

# torch/torchvision are already present in the base image;
# pip will skip them if versions satisfy constraints.
RUN pip install --upgrade pip==24.0 \
    && pip install --no-cache-dir -r requirements.txt

# Non-root user for security; pre-create HF cache dir so the named volume
# mount inherits the correct ownership.
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && mkdir -p /app/.cache/huggingface \
    && mkdir -p /app/data/reports

# Copy application source
COPY --chown=appuser:appuser . .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "smart_city.main:app", "--host", "0.0.0.0", "--port", "8000"]

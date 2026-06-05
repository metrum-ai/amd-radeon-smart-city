<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->
# Release Notes

## v1.3.1

### Updates

* **Minor Documentation Fix**: The documentation directory (`docs/`) is now available in the repository; it was previously missing from earlier release.

## v1.3

### Features

* **GPU‑Accelerated Video Intelligence**: 50 concurrent RTSP streams processed in real time with `YOLOv26` (person/vehicle detection) + `DM‑Count` (crowd density estimation), served by `MIGraphX` on 2× AMD Radeon AI PRO R9700S GPUs (ROCm 7.2).

* **GAIA Agentic Intelligence**: AMD GAIA framework orchestrates an **Investigator Agent** (TimescaleDB queries, alert/trend pattern mining) and an **SOP Advisor Agent** (Milvus RAG over policy documents) under an **Orchestrator Agent**, generating context‑aware incident analyses. Produces policy‑grounded summaries, severity timelines, and recommended actions using local LLM reasoning.

* **Real‑Time Alerting System**: Automatic alerts triggered instantly on threshold breach with **zero operator delay**. Live‑stream auto‑popups with crowd‑density heatmaps appear on the dashboard the moment a zone crosses CRITICAL — no operator intervention required.

* **On‑Prem ROCm‑Optimized LLM Serving**: `unsloth/Qwen3-30B-A3B-GGUF` (UD‑Q4_K_XL, ~17.7 GB) served by **Lemonade Server** on a dedicated R9700S GPU for local incident report generation. Retrieves policy + alert history through the GAIA agents for grounded, auditable outputs — no cloud dependency.

* **City‑Scale Awareness Metrics**: Unified dashboard with live map view, auto‑highlighted critical zones, real‑time analytics for streams, latency, crowd metrics, and system KPIs. WebRTC delivery for sub‑second video playback in any modern browser.

* **Enhanced Multi-GPU Deployment and Profiling Support:** Simplified deployment workflows for both 2-GPU and 4-GPU configurations, along with integrated support for profiling workflows.

* **Ubuntu 24.04 Support**: Updated setup and runtime compatibility for Ubuntu 24.04 environments.

* **VA‑API Hardware Decoding Support**: Added hardware‑accelerated H.264 decode via AMD's VCN engine for RTSP output streams.

* **GPU Allocation Intelligence**: Automatically selects the optimal GPU profile and allocates workloads to the least-utilized or available GPUs.

* **Compliance and Security Hardening**: Resolved high‑severity vulnerability findings. Added SPDX copyright headers to all source files.

### Updates

* **Ultralytics Dependency Removal**: Removed `ultralytics` from the application setup and runtime dependency path. YOLOv26 ONNX models must now be provided locally before setup, keeping model export and licensing review outside the default deployment flow.

---

### Components

#### Python Libraries

| Component | Version | License |
|-----------|---------|---------|
| FastAPI | 0.120.4 | [MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Starlette | 0.49.1 | [BSD-3-Clause](https://github.com/encode/starlette/blob/master/LICENSE.md) |
| Uvicorn | 0.29.0 | [BSD-3-Clause](https://github.com/encode/uvicorn/blob/master/LICENSE.md) |
| Pydantic | 2.9.2 | [MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE) |
| pydantic‑settings | 2.2.1 | [MIT](https://github.com/pydantic/pydantic-settings/blob/main/LICENSE) |
| PyYAML | 6.0.1 | [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| python‑multipart | 0.0.27 | [Apache-2.0](https://github.com/Kludex/python-multipart/blob/master/LICENSE.txt) |
| asyncpg | 0.29.0 | [Apache-2.0](https://github.com/MagicStack/asyncpg/blob/master/LICENSE) |
| httpx | 0.27.0 | [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| setuptools | 78.1.1 | [MIT](https://github.com/pypa/setuptools/blob/main/LICENSE) |
| marshmallow | 3.26.2 | [MIT](https://github.com/marshmallow-code/marshmallow/blob/dev/LICENSE) |
| sentence‑transformers | 3.1.0 | [Apache-2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE) |
| pymilvus | 2.4.3 | [Apache-2.0](https://github.com/milvus-io/pymilvus/blob/master/LICENSE) |
| amd‑gaia | 0.19.0 | [MIT](https://github.com/amd/gaia/blob/main/LICENSE.md) |
| OpenCV (headless) | 4.9.0.80 | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| numpy | 1.26.4 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| prometheus‑client | 0.20.0 | [Apache-2.0](https://github.com/prometheus/client_python/blob/master/LICENSE) |
| pypdf | 6.10.2 | [BSD-3-Clause](https://github.com/py-pdf/pypdf/blob/main/LICENSE) |
| Markdown | 3.8.1 | [BSD-3-Clause](https://github.com/Python-Markdown/markdown/blob/master/LICENSE.md) |
| WeasyPrint | 68.0 | [BSD-3-Clause](https://github.com/Kozea/WeasyPrint/blob/main/LICENSE) |
| PyTorch | 2.10.0 / 2.9.1 | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| onnxruntime‑migraphx | 1.23.2 | [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| onnx | ≥1.16 | [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE) |

#### System Libraries

| Component | Version | License |
|-----------|---------|---------|
| GStreamer core + plugins | 1.24+ | [LGPL-2.1-or-later](https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/licensing.html) |
| PyGObject / python3‑gi | system | [LGPL-2.1-or-later](https://gitlab.gnome.org/GNOME/pygobject/-/blob/master/COPYING) |
| FFmpeg (system apt package) | system | [LGPL-2.1-or-later](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) |
| libva / VA‑API (libva2, libva‑drm2) | system | [MIT](https://github.com/intel/libva/blob/master/COPYING) |
| Mesa GPU drivers (mesa‑va‑drivers, mesa‑vulkan‑drivers) | system | [MIT](https://docs.mesa3d.org/license.html) |

#### Frontend

| Component | Version | License |
|-----------|---------|---------|
| React | 19.0.0 | [MIT](https://github.com/facebook/react/blob/main/LICENSE) |
| react‑dom | 19.0.0 | [MIT](https://github.com/facebook/react/blob/main/LICENSE) |
| react‑redux | 9.2.0 | [MIT](https://github.com/reduxjs/react-redux/blob/master/LICENSE.md) |
| reduxjs/toolkit | 2.11.2 | [MIT](https://github.com/reduxjs/redux-toolkit/blob/master/LICENSE) |
| mui/material | 7.3.8 | [MIT](https://github.com/mui/material-ui/blob/master/LICENSE) |
| mui/icons‑material | 7.3.8 | [MIT](https://github.com/mui/material-ui/blob/master/LICENSE) |
| emotion/react | 11.14.0 | [MIT](https://github.com/emotion-js/emotion/blob/main/LICENSE) |
| emotion/styled | 11.14.1 | [MIT](https://github.com/emotion-js/emotion/blob/main/LICENSE) |
| echarts | 6.0.0 | [Apache-2.0](https://github.com/apache/echarts/blob/master/LICENSE) |
| leaflet | 1.9.4 | [BSD-2-Clause](https://github.com/Leaflet/Leaflet/blob/main/LICENSE) |
| dompurify | 3.4.0 | [Apache-2.0 OR MPL-2.0](https://github.com/cure53/DOMPurify/blob/main/LICENSE) |
| marked | 17.0.5 | [MIT](https://github.com/markedjs/marked/blob/master/LICENSE) |
| motion | 12.34.3 | [MIT](https://github.com/motiondivision/motion/blob/master/LICENSE.md) |
| fontsource‑variable/geist‑mono | 5.2.7 | [OFL-1.1](https://github.com/vercel/geist-font/blob/main/LICENSE.txt) |
| TypeScript | 5.4.5 | [Apache-2.0](https://github.com/microsoft/TypeScript/blob/main/LICENSE.txt) |
| Vite | 5.2.14 | [MIT](https://github.com/vitejs/vite/blob/main/LICENSE) |
| pnpm (build tool) | 10.30.2 | [MIT](https://github.com/pnpm/pnpm/blob/main/LICENSE) |

#### Base Docker Images

| Image | License |
|-------|---------|
| `rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0` | [MIT](https://github.com/ROCm/ROCm/blob/develop/docs/about/LICENSE.md) AND [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` | [MIT](https://github.com/ROCm/ROCm/blob/develop/docs/about/LICENSE.md) AND [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| `jrottenberg/ffmpeg:6-ubuntu` | [Apache-2.0](https://github.com/jrottenberg/ffmpeg/blob/master/LICENSE) AND [LGPL-2.1-or-later](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) |
| `node:22-alpine` | [MIT](https://github.com/nodejs/docker-node/blob/main/LICENSE) |
| `nginx:1.30.1-alpine` | [BSD-2-Clause](https://github.com/nginx/docker-nginx/blob/master/LICENSE) |

#### Infrastructure Services

| Component | Version | License |
|-----------|---------|---------|
| TimescaleDB | 2.14.2 / PG 16 | [Apache-2.0 AND LicenseRef-Timescale](https://github.com/timescale/timescaledb/blob/main/LICENSE) |
| Milvus | 2.4.0 | [Apache-2.0](https://github.com/milvus-io/milvus/blob/master/LICENSE) |
| RustFS | 1.0.0‑beta.3 | [Apache-2.0](https://github.com/rustfs/rustfs/blob/main/LICENSE) |
| etcd | 3.5.5 | [Apache-2.0](https://github.com/etcd-io/etcd/blob/main/LICENSE) |
| MediaMTX | 1.18.2 | [MIT](https://github.com/bluenviron/mediamtx/blob/main/LICENSE) |
| Hugging Face TEI | cpu‑1.6 | [Apache-2.0](https://github.com/huggingface/text-embeddings-inference/blob/main/LICENSE) |
| Lemonade Server | v10.2.0 | [Apache-2.0](https://github.com/lemonade-sdk/lemonade/blob/main/LICENSE) |
| Prometheus | 2.51.2 | [Apache-2.0](https://github.com/prometheus/prometheus/blob/main/LICENSE) |
| AMD Device Metrics Exporter | 1.4.2 | [Apache-2.0](https://github.com/ROCm/device-metrics-exporter/blob/main/LICENSE) |
| node‑exporter | 1.8.2 | [Apache-2.0](https://github.com/prometheus/node_exporter/blob/master/LICENSE) |

#### Models

| Component | License |
|-----------|---------|
| `BAAI/bge-small-en-v1.5` (embeddings) | [MIT](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| `unsloth/Qwen3-30B-A3B-GGUF` (LLM) | [Apache-2.0](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) |

---

### Known Issues

### Application

- **Reports first generation is slow** — 60–120 s on local Lemonade due to model warm‑up.
- **Browser occasionally appears stuck** — if the dashboard freezes or a panel stops updating, perform a hard refresh (`Ctrl+Shift+R` / `Cmd+Shift+R`) to recover.

### Infrastructure

- **Lemonade cold‑start** — first launch of the LLM container can take 5–10 min on a cold GPU as `Qwen3-30B-A3B-GGUF` loads.


---

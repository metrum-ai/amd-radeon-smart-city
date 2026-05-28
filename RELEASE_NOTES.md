<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->
# Release Notes

## v1.2

### Features

* **VA‑API Hardware Decoding Support**: Added hardware‑accelerated H.264 decode via AMD's VCN engine for RTSP output streams. 

* **GPU Allocation Intelligence**: Automatically selects the optimal GPU profile and allocates workloads to the least-utilized or available GPUs.

* **Compliance and Security Hardening**: Resolved high‑severity vulnerability findings. Added SPDX copyright headers to all source files.

---

### Components

#### Python Libraries

| Component | Version | License |
|-----------|---------|---------|
| FastAPI | 0.120.4 | MIT |
| Starlette | 0.49.1 | BSD‑3‑Clause |
| Uvicorn | 0.29.0 | BSD‑3‑Clause |
| Pydantic | 2.9.2 | MIT |
| pydantic‑settings | 2.2.1 | MIT |
| PyYAML | 6.0.1 | MIT |
| python‑multipart | 0.0.27 | MIT |
| asyncpg | 0.29.0 | Apache‑2.0 |
| httpx | 0.27.0 | BSD‑3‑Clause |
| setuptools | 78.1.1 | MIT |
| marshmallow | 3.26.2 | MIT |
| sentence‑transformers | 3.1.0 | Apache‑2.0 |
| pymilvus | 2.4.3 | Apache‑2.0 |
| amd‑gaia | 0.19.0 | Apache‑2.0 |
| OpenCV (headless) | 4.9.0.80 | Apache‑2.0 |
| numpy | 1.26.4 | BSD‑3‑Clause |
| prometheus‑client | 0.20.0 | Apache‑2.0 |
| pypdf | 6.10.2 | BSD‑3‑Clause |
| Markdown | 3.8.1 | BSD‑3‑Clause |
| WeasyPrint | 68.0 | BSD‑3‑Clause |
| PyTorch| 2.10.0 / 2.9.1 | BSD‑3‑Clause |
| onnxruntime‑migraphx | 1.23.2 | MIT |
| onnx | ≥1.16 | Apache‑2.0 |
| ultralytics | ≥8.3.0 | AGPL‑3.0 |

#### System Libraries

| Component | Version | License |
|-----------|---------|---------|
| GStreamer core + plugins | 1.24+ | LGPL‑2.1+ |
| PyGObject / python3‑gi | system | LGPL‑2.1+ |
| FFmpeg (system apt package) | system | LGPL‑2.1+ |
| libva / VA‑API (libva2, libva‑drm2) | system | MIT |
| Mesa GPU drivers (mesa‑va‑drivers, mesa‑vulkan‑drivers) | system | MIT |

#### Frontend

| Component | Version | License |
|-----------|---------|---------|
| React | 19.0.0 | MIT |
| react‑dom | 19.0.0 | MIT |
| react‑redux | 9.2.0 | MIT |
| reduxjs/toolkit | 2.11.2 | MIT |
| mui/material | 7.3.8 | MIT |
| mui/icons‑material | 7.3.8 | MIT |
| emotion/react | 11.14.0 | MIT |
| emotion/styled | 11.14.1 | MIT |
| echarts | 6.0.0 | Apache‑2.0 |
| leaflet | 1.9.4 | BSD‑2‑Clause |
| dompurify | 3.4.0 | Apache‑2.0 / MPL‑2.0 |
| marked | 17.0.5 | MIT |
| motion | 12.34.3 | MIT |
| fontsource‑variable/geist‑mono | 5.2.7 | OFL‑1.1 |
| TypeScript | 5.4.5 | Apache‑2.0 |
| Vite | 5.2.14 | MIT |
| pnpm (build tool) | 10.30.2 | MIT |

#### Base Docker Images

| Image | License |
|-------|---------|
| `rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0` | MIT (ROCm) + BSD‑3‑Clause (PyTorch) |
| `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` | MIT (ROCm) + BSD‑3‑Clause (PyTorch) |
| `jrottenberg/ffmpeg:6-ubuntu` | Apache‑2.0 (image) + LGPL‑2.1+ (FFmpeg) |
| `node:22-alpine` | MIT |
| `nginx:1.30.1-alpine` | BSD‑2‑Clause |

#### Infrastructure Services

| Component | Version | License |
|-----------|---------|---------|
| TimescaleDB | 2.14.2 / PG 16 | Apache‑2.0 / PostgreSQL |
| Milvus | 2.4.0 | Apache‑2.0 |
| RustFS | 1.0.0‑beta.3 | Apache‑2.0 |
| etcd | 3.5.5 | Apache‑2.0 |
| MediaMTX | 1.18.2 | MIT |
| Hugging Face TEI | cpu‑1.6 | Apache‑2.0 |
| Lemonade Server | v10.2.0 | MIT |
| Prometheus | 2.51.2 | Apache‑2.0 |
| AMD Device Metrics Exporter | 1.4.2 | MIT |
| node‑exporter | 1.8.2 | Apache‑2.0 |

#### Models

| Component | License |
|-----------|---------|
| `BAAI/bge-small-en-v1.5` (embeddings) | MIT |
| `unsloth/Qwen3-30B-A3B-GGUF` (LLM) | Apache‑2.0 |

---

## v1.1

### Features

* **Enhanced Multi-GPU Deployment and Profiling Support:** Simplified deployment workflows for both 2-GPU and 4-GPU configurations, along with integrated support for profiling workflows.

* **Ubuntu 24.04 Support**: Updated setup and runtime compatibility for Ubuntu 24.04 environments.

---

## v1.0

### Features

* **GPU‑Accelerated Video Intelligence**: 50 concurrent RTSP streams processed in real time with `YOLOv26` (person/vehicle detection) + `DM‑Count` (crowd density estimation), served by `MIGraphX` on 2× AMD Radeon AI PRO R9700S GPUs (ROCm 7.2).

* **GAIA Agentic Intelligence**: AMD GAIA framework orchestrates an **Investigator Agent** (TimescaleDB queries, alert/trend pattern mining) and an **SOP Advisor Agent** (Milvus RAG over policy documents) under an **Orchestrator Agent**, generating context‑aware incident analyses. Produces policy‑grounded summaries, severity timelines, and recommended actions using local LLM reasoning.

* **Real‑Time Alerting System**: Automatic alerts triggered instantly on threshold breach with **zero operator delay**. Live‑stream auto‑popups with crowd‑density heatmaps appear on the dashboard the moment a zone crosses CRITICAL — no operator intervention required.

* **On‑Prem ROCm‑Optimized LLM Serving**: `unsloth/Qwen3-30B-A3B-GGUF` (UD‑Q4_K_XL, ~17.7 GB) served by **Lemonade Server** on a dedicated R9700S GPU for local incident report generation. Retrieves policy + alert history through the GAIA agents for grounded, auditable outputs — no cloud dependency.

* **City‑Scale Awareness Metrics**: Unified dashboard with live map view, auto‑highlighted critical zones, real‑time analytics for streams, latency, crowd metrics, and system KPIs. WebRTC delivery for sub‑second video playback in any modern browser.

---

### Components

| Component | Version | License |
|-----------|---------|---------|
| FastAPI | 0.111.0 | MIT |
| Uvicorn | 0.29.0 | BSD‑3‑Clause |
| Pydantic | 2.9.2 | MIT |
| asyncpg | 0.29.0 | Apache‑2.0 |
| httpx | 0.27.0 | BSD‑3‑Clause |
| ultralytics (YOLO v26) | 8.2.17 | AGPL‑3.0 |
| onnxruntime‑rocm (MIGraphX EP) | 1.23.2 | MIT |
| OpenCV (headless) | 4.9.0.80 | Apache‑2.0 |
| GStreamer | 1.24+ | LGPL‑2.1+ |
| pymilvus | 2.4.3 | Apache‑2.0 |
| sentence‑transformers | 2.7.0 | Apache‑2.0 |
| WeasyPrint | 62.3 | BSD‑3‑Clause |
| amd‑gaia | 0.17.0 | Apache-2.0 |
| Lemonade Server | v10.2.0 | MIT |
| TimescaleDB | 2.14.2 / PG 16 | Apache‑2.0 / PostgreSQL |
| Milvus | 2.4.0 | Apache‑2.0 |
| RustFS | 1.0.0-beta.3 | Apache‑2.0 |
| etcd | 3.5.5 | Apache‑2.0 |
| MediaMTX | 1.18.2 | MIT |
| Hugging Face TEI | cpu‑1.6 | Apache‑2.0 |
| Prometheus | 2.51.2 | Apache‑2.0 |
| AMD Device Metrics Exporter | 1.4.2 | MIT |
| node‑exporter | 1.8.2 | Apache‑2.0 |
| React + TypeScript + Vite | React 18 / Vite 5 | MIT (React, Vite) + Apache‑2.0 (TypeScript) |
| `rocm/pytorch` | rocm7.1.1_ubuntu22.04_py3.11_pytorch_release_2.10.0  | MIT (ROCm) + BSD‑3‑Clause (PyTorch) |
| `rocm/pytorch` | rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.10.0| MIT (ROCm) + BSD‑3‑Clause (PyTorch) |
| `node` | 22-alpine |  MIT |
| `nginx` | alpine | BSD‑2‑Clause |
| `BAAI/bge-small-en-v1.5` (embeddings) | MIT |
| `unsloth/Qwen3-30B-A3B-GGUF` (LLM) |  Apache‑2.0 |

---

### Known Issues

### Application

- **Reports first generation is slow** — 60–120 s on local Lemonade due to model warm‑up.
- **Browser occasionally appears stuck** — if the dashboard freezes or a panel stops updating, perform a hard refresh (`Ctrl+Shift+R` / `Cmd+Shift+R`) to recover.

### Infrastructure

- **Lemonade cold‑start** — first launch of the LLM container can take 5–10 min on a cold GPU as `Qwen3-30B-A3B-GGUF` loads.


---

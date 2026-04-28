#!/usr/bin/env bash
# Created by Metrum AI for AMD

# =============================================================================
# Smart City Public Safety Platform — Setup & Launch
# Checks prerequisites, configures environment, prepares models, starts services.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[FAIL]${NC}  $*"; }
die()   { error "$*"; exit 1; }
hr()    { echo -e "${BOLD}────────────────────────────────────────────────────${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Hard requirements (no scaling on detected GPU count — we always use 2)
REQUIRED_GPUS=2
YOLO_GPU_ID=0
DMCOUNT_GPU_ID=1
LEMONADE_GPU_ID=1
HSA_VERSION="12.0.1"   # RDNA4 / R9700
DISK_MIN_GB=50
RAM_MIN_GB=32

# Model paths (relative to repo root)
DM_COUNT_PTH="smart_city/models/model_sh_B.pth"
# Pre-exported DM-Count ONNX (name must match pipeline.yaml's density.onnx_path).
# If this is present we skip the .pth requirement and use it directly — the
# pipeline will load ORT + MIGraphX from this file without running the exporter.
DM_COUNT_ONNX="smart_city/models/dm_count_b8_160x120.onnx"
YOLO_ONNX="smart_city/models/yolo26s-384-dynamic.onnx"
YOLO_PT="yolo26s.pt"

echo ""
hr
echo -e "  ${BOLD}Smart City Public Safety Platform — Setup & Launch${NC}"
hr
echo ""

# =============================================================================
# STEP 1: PREREQUISITES
# =============================================================================
echo -e "${BOLD}[1/4] Checking prerequisites...${NC}"
echo ""
PREREQ_FAIL=0

# --- Docker ---
if ! command -v docker &>/dev/null; then
    error "docker not found — install: https://docs.docker.com/get-docker/"
    PREREQ_FAIL=1
elif ! docker info &>/dev/null 2>&1; then
    error "Docker daemon not running — start with: sudo systemctl start docker"
    PREREQ_FAIL=1
else
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
    DOCKER_MAJOR=$(echo "$DOCKER_VER" | cut -d. -f1)
    if [ "${DOCKER_MAJOR:-0}" -lt 25 ]; then
        error "Docker $DOCKER_VER detected — version 25.0+ required"
        PREREQ_FAIL=1
    else
        ok "Docker $DOCKER_VER"
    fi
fi

# --- Docker Compose v2 ---
if docker compose version &>/dev/null 2>&1; then
    ok "Docker Compose $(docker compose version --short 2>/dev/null || echo 'v2')"
else
    error "Docker Compose v2 not found — install: https://docs.docker.com/compose/install/"
    PREREQ_FAIL=1
fi

# --- ROCm + AMD GPUs ---
GPU_COUNT=0
if command -v rocm-smi &>/dev/null; then
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -o "GPU\[[0-9]*\]" | sort -u | wc -l)
    GPU_COUNT=${GPU_COUNT:-0}
    if [ "$GPU_COUNT" -eq 0 ]; then
        GPU_COUNT=$(rocm-smi -i 2>/dev/null | grep -o "GPU\[[0-9]*\]" | sort -u | wc -l)
        GPU_COUNT=${GPU_COUNT:-0}
    fi
fi

# Fallback: count AMD vendor IDs in /sys/class/drm
if [ "$GPU_COUNT" -eq 0 ] && [ -d /sys/class/drm ]; then
    GPU_COUNT=$(for f in /sys/class/drm/card*/device/vendor; do
        [ -f "$f" ] && cat "$f" 2>/dev/null
    done | grep -c "0x1002" || echo 0)
fi

if [ "$GPU_COUNT" -lt "$REQUIRED_GPUS" ]; then
    error "${GPU_COUNT} AMD GPU(s) detected — minimum ${REQUIRED_GPUS} required"
    error "  GPU 0 runs YOLO; GPU 1 runs DM-Count + Lemonade LLM."
    PREREQ_FAIL=1
else
    GPU_NAMES=""
    if command -v rocm-smi &>/dev/null; then
        GPU_NAMES=$(rocm-smi --showproductname 2>/dev/null \
            | grep "Card Series" \
            | sed 's/.*: *//' \
            | tr '\n' ',' \
            | sed 's/,$//' || echo "")
    fi
    if [ -z "$GPU_NAMES" ]; then
        if [ "$GPU_COUNT" -gt "$REQUIRED_GPUS" ]; then
            ok "${GPU_COUNT} AMD GPU(s) detected — using only GPU 0 and GPU 1"
        else
            ok "${GPU_COUNT} AMD GPU(s) detected"
        fi
    else
        if [ "$GPU_COUNT" -gt "$REQUIRED_GPUS" ]; then
            ok "${GPU_COUNT} AMD GPU(s): ${GPU_NAMES} — using only GPU 0 and GPU 1"
        else
            ok "${GPU_COUNT} AMD GPU(s): ${GPU_NAMES}"
        fi
    fi
fi

# --- ROCm device nodes ---
if [ ! -e /dev/kfd ]; then
    error "/dev/kfd not found — ROCm kernel driver not loaded"
    PREREQ_FAIL=1
elif [ ! -d /dev/dri ]; then
    error "/dev/dri not found — DRM subsystem not available"
    PREREQ_FAIL=1
else
    ok "ROCm device nodes (/dev/kfd, /dev/dri)"
fi

# --- Disk ---
FREE_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${FREE_GB:-0}" -lt "$DISK_MIN_GB" ]; then
    error "Only ${FREE_GB} GB free — minimum ${DISK_MIN_GB} GB required"
    PREREQ_FAIL=1
else
    ok "${FREE_GB} GB free disk space"
fi

# --- RAM ---
TOTAL_RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${TOTAL_RAM_GB:-0}" -lt "$RAM_MIN_GB" ]; then
    error "${TOTAL_RAM_GB} GB RAM — minimum ${RAM_MIN_GB} GB required"
    PREREQ_FAIL=1
else
    ok "${TOTAL_RAM_GB} GB total RAM"
fi

echo ""
if [ "$PREREQ_FAIL" -ne 0 ]; then
    die "Fix the errors above then re-run this script."
fi

# =============================================================================
# STEP 2: ENVIRONMENT CONFIGURATION
# =============================================================================
echo -e "${BOLD}[2/4] Environment configuration${NC}"
echo ""

SKIP_ENV=0
if [ -f .env ]; then
    warn ".env already exists."
    read -rp "  Overwrite it? [y/N]: " _ow || _ow="N"
    if [[ ! "${_ow:-N}" =~ ^[Yy]$ ]]; then
        info "Keeping existing .env — only GPU and profile settings will be refreshed."
        SKIP_ENV=1
        echo ""
    else
        rm .env
    fi
fi

if [ "$SKIP_ENV" -eq 0 ]; then
    if [ ! -f .env.example ]; then
        die ".env.example missing — cannot scaffold .env"
    fi
    cp .env.example .env
    echo "  Press Enter to accept the default shown in [brackets]."
    echo ""

    # --- HuggingFace Token (optional — only needed for gated models) ---
    # Public models like the default Qwen3-30B GGUF download fine without one.
    # Provide a token only if you switch LEMONADE_CHECKPOINT to a gated repo
    # (e.g. Llama, Mistral) or hit HF rate limits on a shared IP.
    echo -e "  ${BOLD}Secrets${NC}"
    read -rp "  HuggingFace API Token (hf_..., optional — press Enter to skip): " _hf || _hf=""
    if [ -n "$_hf" ]; then
        # Use | as sed delimiter to avoid issues if the token contains /
        sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${_hf}|" .env
        ok "  HF_TOKEN set"
    else
        # Blank the placeholder so downstream tools see an empty value rather
        # than the literal "hf_your_token_here" string from .env.example.
        sed -i "s|^HF_TOKEN=.*|HF_TOKEN=|" .env
        info "  Skipped — HF_TOKEN left empty. Set it later in .env if you switch to a gated model."
    fi

    echo ""

    # --- Postgres credentials ---
    echo -e "  ${BOLD}TimescaleDB / Postgres${NC}"
    read -rp "  Postgres username [smartcity]: " _pg_user || _pg_user=""
    _pg_user="${_pg_user:-smartcity}"

    while true; do
        read -rsp "  Postgres password (will not be echoed): " _pg_pw
        echo ""
        if [ -z "${_pg_pw:-}" ]; then
            error "  Password cannot be empty."
            continue
        fi
        if [ "$_pg_pw" = "changeme" ] || [ "$_pg_pw" = "password" ]; then
            warn "  '$_pg_pw' is a default placeholder — please pick something stronger."
            continue
        fi
        read -rsp "  Confirm password: " _pg_pw2
        echo ""
        if [ "$_pg_pw" != "$_pg_pw2" ]; then
            error "  Passwords do not match — try again."
            continue
        fi
        break
    done

    read -rp "  Postgres database name [smartcity_db]: " _pg_db || _pg_db=""
    _pg_db="${_pg_db:-smartcity_db}"

    sed -i "s|^POSTGRES_USER=.*|POSTGRES_USER=${_pg_user}|" .env
    sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${_pg_pw}|" .env
    sed -i "s|^POSTGRES_DB=.*|POSTGRES_DB=${_pg_db}|" .env
    sed -i "s|^DATABASE_URL=.*|DATABASE_URL=postgresql://${_pg_user}:${_pg_pw}@timescaledb:5432/${_pg_db}|" .env

    echo ""

    # --- LLM model name ---
    # Default is the Qwen3-30B-A3B MoE (Unsloth UD-Q4_K_XL GGUF). If the user
    # accepts the default we leave LEMONADE_CHECKPOINT alone (template already
    # pairs them); if they pick a custom model name they own the checkpoint
    # mapping themselves and should edit .env after setup.
    DEFAULT_LLM="user.Qwen3-30B-A3B-UD-Q4_K_XL"
    echo -e "  ${BOLD}LLM (Lemonade)${NC}"
    read -rp "  Lemonade model name [${DEFAULT_LLM}]: " _llm || _llm=""
    _llm="${_llm:-${DEFAULT_LLM}}"
    sed -i "s|^LLM_MODEL=.*|LLM_MODEL=${_llm}|" .env
    sed -i "s|^LEMONADE_MODEL=.*|LEMONADE_MODEL=${_llm}|" .env
    sed -i "s|^VLLM_SERVED_MODEL_NAME=.*|VLLM_SERVED_MODEL_NAME=${_llm}|" .env
    if [ "$_llm" != "$DEFAULT_LLM" ]; then
        warn "  Custom LLM '${_llm}' selected — verify LEMONADE_CHECKPOINT in .env"
        warn "  points at the matching upstream HF GGUF or first-boot pull will fail."
    fi

    echo ""
    ok ".env written"
    echo ""
fi

# --- Always (re)apply GPU + ROCm settings ---
info "Applying GPU assignment to .env..."

# pipeline sees both GPUs (it routes YOLO->0, DM-Count->1 internally)
sed -i "s|^ROCR_VISIBLE_DEVICES=.*|ROCR_VISIBLE_DEVICES=0,1|" .env
sed -i "s|^HSA_OVERRIDE_GFX_VERSION=.*|HSA_OVERRIDE_GFX_VERSION=${HSA_VERSION}|" .env
sed -i "s|^NUM_GPUS=.*|NUM_GPUS=${REQUIRED_GPUS}|" .env

# Append per-service GPU IDs if not already present (compose can reference these
# once parameterised; harmless if it doesn't yet).
add_or_replace() {
    local key="$1"; local value="$2"
    if grep -q "^${key}=" .env 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        # Ensure file ends with a newline before appending, otherwise we glue
        # the new key onto the previous last line.
        if [ -s .env ] && [ "$(tail -c 1 .env)" != "" ]; then
            echo "" >> .env
        fi
        echo "${key}=${value}" >> .env
    fi
}
add_or_replace "YOLO_GPU_ID"     "${YOLO_GPU_ID}"
add_or_replace "DMCOUNT_GPU_ID"  "${DMCOUNT_GPU_ID}"
add_or_replace "LEMONADE_GPU_ID" "${LEMONADE_GPU_ID}"

ok "YOLO=GPU ${YOLO_GPU_ID}, DM-Count=GPU ${DMCOUNT_GPU_ID}, Lemonade=GPU ${LEMONADE_GPU_ID}"
ok "ROCR_VISIBLE_DEVICES=0,1, HSA_OVERRIDE_GFX_VERSION=${HSA_VERSION}"
echo ""

# --- WebRTC ICE candidate hosts ---
# MediaMTX advertises these to browsers; the browser MUST be able to reach one
# of them on UDP/TCP 8189. Default is just localhost/127.0.0.1, which only
# works for browsers running on this very host. We auto-add every non-loopback
# IPv4 of the host so any LAN client also gets a reachable candidate.
LAN_IPS=$(ip -4 -o addr show scope global 2>/dev/null \
    | awk '{print $4}' | cut -d/ -f1 | sort -u | paste -sd, -)
if [ -n "$LAN_IPS" ]; then
    PUBLIC_HOSTS="127.0.0.1,localhost,${LAN_IPS}"
else
    PUBLIC_HOSTS="127.0.0.1,localhost"
fi
add_or_replace "PUBLIC_WEBRTC_HOSTS" "${PUBLIC_HOSTS}"
ok "WebRTC ICE hosts: ${PUBLIC_HOSTS}"
echo ""

# =============================================================================
# STEP 3: MODELS
# =============================================================================
echo -e "${BOLD}[3/4] Model bootstrap${NC}"
echo ""

# ---------------------------------------------------------------------------
# pip helper — installs a package if not already importable.
# Tries plain pip first; falls back to --break-system-packages for Ubuntu
# 23.04+ / Debian 12+ where the system Python is "externally managed".
# ---------------------------------------------------------------------------
_pip_install() {
    local pkg="$1"; local import_name="${2:-$1}"
    if python3 -c "import ${import_name}" &>/dev/null 2>&1; then
        return 0
    fi
    info "Installing ${pkg}..."
    if python3 -m pip install --quiet "$pkg" 2>/dev/null; then
        return 0
    fi
    # PEP 668 externally-managed environment — try with override flag
    if python3 -m pip install --quiet --break-system-packages "$pkg" 2>/dev/null; then
        return 0
    fi
    die "Failed to install ${pkg}. Run manually: pip install ${pkg}"
}

# --- DM-Count model ---
# Prefer the pre-exported ONNX if it's already shipped with the repo; in that
# case we don't need the .pth and we don't need to run the exporter. Fall back
# to the .pth → ONNX export path only if the ONNX is absent.
if [ -f "$DM_COUNT_ONNX" ]; then
    SIZE_MB=$(du -m "$DM_COUNT_ONNX" | awk '{print $1}')
    ok "DM-Count ONNX present: ${DM_COUNT_ONNX} (${SIZE_MB} MB) — skipping .pth export"
    if [ -f "$DM_COUNT_PTH" ]; then
        PTH_MB=$(du -m "$DM_COUNT_PTH" | awk '{print $1}')
        info "DM-Count .pth also present (${PTH_MB} MB) — will be ignored; ONNX is authoritative."
    fi
elif [ -f "$DM_COUNT_PTH" ]; then
    SIZE_MB=$(du -m "$DM_COUNT_PTH" | awk '{print $1}')
    ok "DM-Count weights present: ${DM_COUNT_PTH} (${SIZE_MB} MB) — exporting to ONNX now"
    if ! command -v python3 &>/dev/null; then
        die "python3 not found — install python3 and re-run."
    fi
    _pip_install "torch" "torch"
    _pip_install "onnx" "onnx"
    _pip_install "onnxconverter-common" "onnxconverter_common"
    if ! python3 scripts/export_dm_count_onnx.py \
            --weights "$DM_COUNT_PTH" \
            --output  "$DM_COUNT_ONNX"; then
        error "DM-Count export failed — see error above."
        die "Cannot continue without DM-Count ONNX."
    fi
    ok "DM-Count ONNX exported: ${DM_COUNT_ONNX}"
else
    info "DM-Count model missing — downloading weights and exporting ONNX..."
    if ! command -v python3 &>/dev/null; then
        die "python3 not found — install python3 and re-run."
    fi
    _pip_install "gdown"
    _pip_install "torch" "torch"
    _pip_install "onnx" "onnx"
    _pip_install "onnxconverter-common" "onnxconverter_common"

    mkdir -p "$(dirname "$DM_COUNT_PTH")"
    if ! python3 -m gdown 1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru -O "$DM_COUNT_PTH"; then
        error "Download failed."
        echo "         Download manually and re-run:"
        echo "           pip install gdown"
        echo "           gdown 1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru -O ${DM_COUNT_PTH}"
        die "Cannot continue without DM-Count weights."
    fi
    ok "DM-Count weights downloaded: ${DM_COUNT_PTH}"

    if ! python3 scripts/export_dm_count_onnx.py \
            --weights "$DM_COUNT_PTH" \
            --output  "$DM_COUNT_ONNX"; then
        error "DM-Count export failed — see error above."
        die "Cannot continue without DM-Count ONNX."
    fi
    ok "DM-Count ONNX exported: ${DM_COUNT_ONNX}"
fi

# --- YOLO ONNX (export on demand from Ultralytics .pt) ---
if [ -f "$YOLO_ONNX" ]; then
    SIZE_MB=$(du -m "$YOLO_ONNX" | awk '{print $1}')
    ok "YOLO ONNX present: ${YOLO_ONNX} (${SIZE_MB} MB)"
else
    info "YOLO ONNX missing — exporting ${YOLO_PT} → ${YOLO_ONNX}"
    if ! command -v python3 &>/dev/null; then
        die "python3 not found — install python3 and re-run."
    fi
    _pip_install "ultralytics"
    if ! python3 scripts/export_yolo_onnx.py --model "$YOLO_PT" --output "$YOLO_ONNX"; then
        error "YOLO export failed — see error above."
        die "Cannot continue without YOLO ONNX."
    fi
    ok "YOLO ONNX exported: ${YOLO_ONNX}"
fi

echo ""

# =============================================================================
# STEP 4: BUILD & LAUNCH
# =============================================================================
echo -e "${BOLD}[4/4] Building & starting services...${NC}"
echo ""

info "Building all service images (this takes a while on first run)..."
if ! docker compose build; then
    die "Docker build failed — check output above."
fi
ok "All images built"
echo ""

info "Starting services..."
docker compose up -d

echo ""
hr
ok "All services started."
hr
echo ""
echo "  GPU layout:"
echo "    GPU 0  →  YOLO (pipeline)"
echo "    GPU 1  →  DM-Count (pipeline) + Lemonade (LLM)"
echo ""
echo "  Endpoints:"
echo "    Frontend  →  http://localhost:5173    (or whatever 'frontend' maps to)"
echo "    API       →  http://localhost:8000"
echo "    Lemonade  →  http://localhost:13305"
echo ""
echo "  First boot: Lemonade pulls the LLM into its cache (~5 GB), the pipeline"
echo "  exports DM-Count to ONNX (~30 s), and Milvus + Postgres initialise."
echo "  Tail logs with:"
echo "    docker compose logs -f lemonade"
echo "    docker compose logs -f pipeline"
echo "    docker compose logs -f api"
echo ""
echo "  To stop everything:"
echo "    docker compose down -v"
echo ""

# --- Sanity check that compose still honours LEMONADE_GPU_ID ---
if ! grep -q 'LEMONADE_GPU_ID' docker-compose.yml 2>/dev/null; then
    warn "docker-compose.yml does not reference LEMONADE_GPU_ID — the .env value"
    warn "  may not propagate to the lemonade container. Verify the lemonade"
    warn "  service block uses ROCR_VISIBLE_DEVICES: \"\${LEMONADE_GPU_ID:-1}\"."
fi

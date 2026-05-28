#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

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

retry_command() {
    local label="$1"
    local attempts="${2:-3}"
    local delay_s="${3:-15}"
    shift 3

    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$attempts" ]; then
            return 1
        fi
        warn "${label} failed (attempt ${attempt}/${attempts}); retrying in ${delay_s}s..."
        sleep "$delay_s"
        attempt=$((attempt + 1))
    done
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/setup_runtime_defaults.sh
source "$SCRIPT_DIR/scripts/setup_runtime_defaults.sh"

# Hard requirements and runtime defaults
MIN_REQUIRED_GPUS=2
HSA_VERSION="12.0.1"   # RDNA4 / R9700
DISK_MIN_GB=40
RAM_MIN_GB=32
MODEL_EXPORT_VENV_DIR="${MODEL_EXPORT_VENV_DIR:-.venv-model-export}"
MODEL_EXPORT_PYTHON="$(model_export_python_path "$MODEL_EXPORT_VENV_DIR")"
MODEL_EXPORT_VENV_READY=0
SETUP_DOCKER_RETRY_ATTEMPTS="${SETUP_DOCKER_RETRY_ATTEMPTS:-3}"
SETUP_DOCKER_RETRY_DELAY_S="${SETUP_DOCKER_RETRY_DELAY_S:-20}"

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

# --- Python + venv support for model export ---
if ! command -v python3 &>/dev/null; then
    error "python3 not found — install it first:"
    error "  sudo apt-get update && sudo apt-get install -y python3 python3-venv"
    PREREQ_FAIL=1
else
    ok "$(python3 --version 2>/dev/null || echo python3)"
    if python3 -m venv --help &>/dev/null 2>&1; then
        ok "python3 venv module"
    else
        error "python3 venv module not available — install python3-venv:"
        error "  sudo apt-get update && sudo apt-get install -y python3-venv"
        PREREQ_FAIL=1
    fi
fi

# --- ROCm + AMD GPUs ---
DETECTED_GPU_COUNT=0
if command -v rocm-smi &>/dev/null; then
    DETECTED_GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -o "GPU\[[0-9]*\]" | sort -u | wc -l)
    DETECTED_GPU_COUNT=${DETECTED_GPU_COUNT:-0}
    if [ "$DETECTED_GPU_COUNT" -eq 0 ]; then
        DETECTED_GPU_COUNT=$(rocm-smi -i 2>/dev/null | grep -o "GPU\[[0-9]*\]" | sort -u | wc -l)
        DETECTED_GPU_COUNT=${DETECTED_GPU_COUNT:-0}
    fi
fi

# Fallback: count AMD vendor IDs in /sys/class/drm
if [ "$DETECTED_GPU_COUNT" -eq 0 ] && [ -d /sys/class/drm ]; then
    DETECTED_GPU_COUNT=$(for f in /sys/class/drm/card*/device/vendor; do
        [ -f "$f" ] && cat "$f" 2>/dev/null
    done | grep -c "0x1002" || echo 0)
fi

SELECTED_GPU_PROFILE="$(select_gpu_profile "$DETECTED_GPU_COUNT")"
SELECTED_COMPOSE_FILE="$(profile_compose_file "$SELECTED_GPU_PROFILE")"

# Detect which physical GPUs are currently free (lowest utilisation) and
# select the best N for this profile.  Falls back to 0..N-1 when rocm-smi
# is unavailable or returns no usable data.
GPUS_NEEDED=$([ "$SELECTED_GPU_PROFILE" = "4gpu" ] && echo 4 || echo 2)
SELECTED_GPU_IDS="$(detect_free_gpus "$GPUS_NEEDED" "$DETECTED_GPU_COUNT")"

if [ "$DETECTED_GPU_COUNT" -lt "$MIN_REQUIRED_GPUS" ]; then
    error "${DETECTED_GPU_COUNT} AMD GPU(s) detected — minimum ${MIN_REQUIRED_GPUS} required"
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
        if [ "$DETECTED_GPU_COUNT" -ge 4 ]; then
            ok "${DETECTED_GPU_COUNT} AMD GPU(s) detected — selecting 4-GPU profile"
        elif [ "$DETECTED_GPU_COUNT" -gt "$MIN_REQUIRED_GPUS" ]; then
            warn "${DETECTED_GPU_COUNT} AMD GPU(s) detected — no 3-GPU profile; selecting 2-GPU profile"
        else
            ok "${DETECTED_GPU_COUNT} AMD GPU(s) detected — selecting 2-GPU profile"
        fi
    else
        if [ "$DETECTED_GPU_COUNT" -ge 4 ]; then
            ok "${DETECTED_GPU_COUNT} AMD GPU(s): ${GPU_NAMES} — selecting 4-GPU profile"
        elif [ "$DETECTED_GPU_COUNT" -gt "$MIN_REQUIRED_GPUS" ]; then
            warn "${DETECTED_GPU_COUNT} AMD GPU(s): ${GPU_NAMES} — no 3-GPU profile; selecting 2-GPU profile"
        else
            ok "${DETECTED_GPU_COUNT} AMD GPU(s): ${GPU_NAMES} — selecting 2-GPU profile"
        fi
    fi
    # Show which physical GPU IDs were chosen (freest by utilisation).
    ok "Selected GPUs: $(echo "$SELECTED_GPU_IDS" | tr ' ' ',') (freest by utilization)"
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

# --- CPU cores ---
# Prefer physical cores so SMT siblings do not double the stream-count default.
CPU_CORE_COUNT="$(detect_cpu_core_count)"
ok "${CPU_CORE_COUNT} physical CPU core(s) detected"

# --- Kernel / decode-backend detection ---
# 2-GPU profiles always use libav software decode: GPU 1 is already shared by
# DM-Count + Lemonade, so VA-API VCN decode has no stable headroom there.
# 4-GPU profiles can use VA-API hardware decode on kernel 6.17+, using the
# legacy vaapih264dec path validated for non-black RGB frames. The user can
# still force software on 4-GPU by exporting DECODE_MODE=software before
# running setup.sh. ENCODE_MODE is intentionally NOT auto-flipped — it stays on
# libx264 software regardless of kernel.
KERNEL_RELEASE="$(uname -r 2>/dev/null || echo unknown)"
EXPLICIT_DECODE_MODE="${DECODE_MODE:-}"
SELECTED_DECODE_MODE="$(select_decode_mode "$KERNEL_RELEASE" "$EXPLICIT_DECODE_MODE" "$SELECTED_GPU_PROFILE")"
if [ "$SELECTED_GPU_PROFILE" = "2gpu" ]; then
    ok "Profile 2gpu — forcing software decode; hardware decode is reserved for 4-GPU profile"
elif [ -n "$EXPLICIT_DECODE_MODE" ] && [ "$EXPLICIT_DECODE_MODE" != "auto" ]; then
    ok "Kernel ${KERNEL_RELEASE} — DECODE_MODE forced to '${SELECTED_DECODE_MODE}' via env"
elif [ "$SELECTED_DECODE_MODE" = "hardware" ]; then
    ok "Kernel ${KERNEL_RELEASE} (≥ 6.17) — enabling VA-API hardware decode"
else
    warn "Kernel ${KERNEL_RELEASE} (< 6.17) — falling back to software decode"
    warn "  Upgrade to kernel 6.17+ for stable amdgpu VA-API on RDNA4 GPUs."
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

env_file_value() {
    local key="$1"; local file="${2:-.env}"
    [ -f "$file" ] || return 1
    awk -F= -v key="$key" '
        $1 == key {
            sub(/^[^=]*=/, "")
            print
            found = 1
            exit
        }
        END { exit found ? 0 : 1 }
    ' "$file"
}

explicit_or_existing_env_value() {
    local key="$1"; local read_existing="${2:-0}"
    local shell_value="${!key-}"
    if [ -n "$shell_value" ]; then
        echo "$shell_value"
        return
    fi
    if [ "$read_existing" -eq 1 ]; then
        env_file_value "$key" .env || true
    fi
}

add_or_replace() {
    local key="$1"; local value="$2"
    if [ -f .env ] && awk -F= -v key="$key" '$1 == key { found = 1 } END { exit found ? 0 : 1 }' .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        if [ -s .env ]; then
            printf '\n' >> .env
        fi
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

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

    # --- RustFS credentials ---
    echo -e "  ${BOLD}RustFS object store${NC}"
    while true; do
        read -rp "  RustFS access key: " _rustfs_access_key || _rustfs_access_key=""
        if [ -z "${_rustfs_access_key:-}" ]; then
            error "  Access key cannot be empty."
            continue
        fi
        if [ "$_rustfs_access_key" = "rustfsadmin" ] || [ "$_rustfs_access_key" = "changeme" ]; then
            warn "  '$_rustfs_access_key' is a default placeholder — please pick something unique."
            continue
        fi
        break
    done

    while true; do
        read -rsp "  RustFS secret key (will not be echoed): " _rustfs_secret_key
        echo ""
        if [ -z "${_rustfs_secret_key:-}" ]; then
            error "  Secret key cannot be empty."
            continue
        fi
        if [ "${#_rustfs_secret_key}" -lt 8 ]; then
            error "  Secret key must be at least 8 characters."
            continue
        fi
        if [ "$_rustfs_secret_key" = "rustfsadmin" ] || [ "$_rustfs_secret_key" = "changeme" ] || [ "$_rustfs_secret_key" = "password" ]; then
            warn "  '$_rustfs_secret_key' is a default placeholder — please pick something stronger."
            continue
        fi
        read -rsp "  Confirm secret key: " _rustfs_secret_key2
        echo ""
        if [ "$_rustfs_secret_key" != "$_rustfs_secret_key2" ]; then
            error "  Secret keys do not match — try again."
            continue
        fi
        break
    done

    sed -i "s|^RUSTFS_ACCESS_KEY=.*|RUSTFS_ACCESS_KEY=${_rustfs_access_key}|" .env
    sed -i "s|^RUSTFS_SECRET_KEY=.*|RUSTFS_SECRET_KEY=${_rustfs_secret_key}|" .env

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

# --- Always (re)apply GPU profile, stream defaults, and ROCm settings ---
info "Applying ${SELECTED_GPU_PROFILE} runtime profile to .env..."

while IFS= read -r assignment; do
    [ -n "$assignment" ] || continue
    add_or_replace "${assignment%%=*}" "${assignment#*=}"
done < <(profile_env_assignments "$SELECTED_GPU_PROFILE" "$SELECTED_GPU_IDS")
add_or_replace "HSA_OVERRIDE_GFX_VERSION" "${HSA_VERSION}"
add_or_replace "VIDEO_GID" "$(host_group_gid video 44)"
add_or_replace "RENDER_GID" "$(host_group_gid render 109)"

# Shell environment values win. Existing .env values win only when the user kept
# that file; freshly scaffolded .env values are treated as template defaults.
STREAM_OVERRIDE="$(explicit_or_existing_env_value "STREAM_COUNT" "$SKIP_ENV")"
DENSITY_OVERRIDE="$(explicit_or_existing_env_value "DENSITY_DISPLAY_STREAMS" "$SKIP_ENV")"
SELECTED_STREAM_COUNT="$(select_stream_count "$CPU_CORE_COUNT" "$STREAM_OVERRIDE")"
SELECTED_DENSITY_DISPLAY_STREAMS="$(select_density_display_streams "$SELECTED_STREAM_COUNT" "$DENSITY_OVERRIDE")"
add_or_replace "STREAM_COUNT" "${SELECTED_STREAM_COUNT}"
add_or_replace "DENSITY_DISPLAY_STREAMS" "${SELECTED_DENSITY_DISPLAY_STREAMS}"
add_or_replace "DECODE_MODE" "${SELECTED_DECODE_MODE}"

ok "Profile ${SELECTED_GPU_PROFILE}: compose override ${SELECTED_COMPOSE_FILE}"
# Map the selected physical GPU IDs to role names for the status line.
# Role indices are relative to ROCR_VISIBLE_DEVICES; resolve them back to
# the physical IDs we chose so the message reflects reality on this host.
read -ra _sel_gpus <<< "$SELECTED_GPU_IDS"
if [ "$SELECTED_GPU_PROFILE" = "4gpu" ]; then
    # Relative: 0→DM-Count, 1+2→YOLO, 3→Lemonade
    ok "GPU ${_sel_gpus[0]:-0} → DM-Count, GPU ${_sel_gpus[1]:-1},${_sel_gpus[2]:-2} → YOLO, GPU ${_sel_gpus[3]:-3} → Lemonade"
else
    # Relative: 0→YOLO, 1→DM-Count + Lemonade
    ok "GPU ${_sel_gpus[0]:-0} → YOLO, GPU ${_sel_gpus[1]:-1} → DM-Count + Lemonade"
fi
ok "STREAM_COUNT=${SELECTED_STREAM_COUNT}, DENSITY_DISPLAY_STREAMS=${SELECTED_DENSITY_DISPLAY_STREAMS} (${CPU_CORE_COUNT} physical cores)"
ok "DECODE_MODE=${SELECTED_DECODE_MODE} (kernel ${KERNEL_RELEASE}); ENCODE_MODE=software"
ok "GPU device groups: VIDEO_GID=$(host_group_gid video 44), RENDER_GID=$(host_group_gid render 109)"
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
# Model export venv helpers.
# Ubuntu 24 marks system Python as externally managed, so setup never installs
# model-export packages into system Python and never uses --break-system-packages.
# ---------------------------------------------------------------------------
ensure_model_export_venv() {
    if [ "$MODEL_EXPORT_VENV_READY" -eq 1 ]; then
        return
    fi
    if [ ! -x "$MODEL_EXPORT_PYTHON" ]; then
        info "Creating model export venv at ${MODEL_EXPORT_VENV_DIR}..."
        python3 -m venv "$MODEL_EXPORT_VENV_DIR"
    fi
    "$MODEL_EXPORT_PYTHON" -m pip install --quiet --upgrade pip setuptools wheel
    MODEL_EXPORT_VENV_READY=1
}

_venv_pip_install() {
    local pkg="$1"; local import_name="${2:-$1}"
    ensure_model_export_venv
    if "$MODEL_EXPORT_PYTHON" -c "import ${import_name}" &>/dev/null 2>&1; then
        return 0
    fi
    info "Installing ${pkg} into ${MODEL_EXPORT_VENV_DIR}..."
    if ! "$MODEL_EXPORT_PYTHON" -m pip install --quiet "$pkg"; then
        die "Failed to install ${pkg}. The system Python was left untouched; inspect ${MODEL_EXPORT_VENV_DIR} and retry."
    fi
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
    _venv_pip_install "torch" "torch"
    _venv_pip_install "onnx>=1.12.0,<2.0.0" "onnx"
    _venv_pip_install "onnxconverter-common" "onnxconverter_common"
    if ! PYTHONPATH="$SCRIPT_DIR/smart_city" "$MODEL_EXPORT_PYTHON" -m combined_pipeline.inference.export_dm_count \
            --weights "$DM_COUNT_PTH" \
            --output  "$DM_COUNT_ONNX"; then
        error "DM-Count export failed — see error above."
        die "Cannot continue without DM-Count ONNX."
    fi
    ok "DM-Count ONNX exported: ${DM_COUNT_ONNX}"
else
    info "DM-Count model missing — downloading weights and exporting ONNX..."
    _venv_pip_install "gdown" "gdown"
    _venv_pip_install "torch" "torch"
    _venv_pip_install "onnx>=1.12.0,<2.0.0" "onnx"
    _venv_pip_install "onnxconverter-common" "onnxconverter_common"

    mkdir -p "$(dirname "$DM_COUNT_PTH")"
    if ! "$MODEL_EXPORT_PYTHON" -m gdown 1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru -O "$DM_COUNT_PTH"; then
        error "Download failed."
        echo "         Download manually and re-run:"
        echo "           ${MODEL_EXPORT_PYTHON} -m pip install gdown"
        echo "           ${MODEL_EXPORT_PYTHON} -m gdown 1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru -O ${DM_COUNT_PTH}"
        die "Cannot continue without DM-Count weights."
    fi
    ok "DM-Count weights downloaded: ${DM_COUNT_PTH}"

    if ! PYTHONPATH="$SCRIPT_DIR/smart_city" "$MODEL_EXPORT_PYTHON" -m combined_pipeline.inference.export_dm_count \
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
    _venv_pip_install "torch" "torch"
    _venv_pip_install "ultralytics" "ultralytics"
    _venv_pip_install "onnx>=1.12.0,<2.0.0" "onnx"
    _venv_pip_install "onnxruntime" "onnxruntime"
    if ! "$MODEL_EXPORT_PYTHON" scripts/export_yolo_onnx.py --model "$YOLO_PT" --output "$YOLO_ONNX"; then
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

COMPOSE_ARGS=(-f docker-compose.yml -f "$SELECTED_COMPOSE_FILE")

info "Building all service images (this takes a while on first run)..."
if ! retry_command "Docker Compose build" "$SETUP_DOCKER_RETRY_ATTEMPTS" "$SETUP_DOCKER_RETRY_DELAY_S" \
    docker compose "${COMPOSE_ARGS[@]}" build; then
    die "Docker build failed — check output above."
fi
ok "All images built"
echo ""

info "Starting services..."
if ! retry_command "Docker Compose startup" "$SETUP_DOCKER_RETRY_ATTEMPTS" "$SETUP_DOCKER_RETRY_DELAY_S" \
    docker compose "${COMPOSE_ARGS[@]}" up -d; then
    die "Docker Compose startup failed — check output above."
fi

echo ""
hr
ok "All services started."
hr
echo ""
echo "  GPU layout (physical IDs):"
# Resolve physical GPU IDs for the summary.  _sel_gpus[] is already set above.
if [ "$SELECTED_GPU_PROFILE" = "4gpu" ]; then
    echo "    GPU ${_sel_gpus[0]:-0}      →  DM-Count (pipeline)"
    echo "    GPU ${_sel_gpus[1]:-1},${_sel_gpus[2]:-2}    →  YOLO (pipeline, 2 replicas/GPU)"
    echo "    GPU ${_sel_gpus[3]:-3}      →  Lemonade (LLM)"
else
    echo "    GPU ${_sel_gpus[0]:-0}      →  YOLO (pipeline, 2 replicas)"
    echo "    GPU ${_sel_gpus[1]:-1}      →  DM-Count (pipeline) + Lemonade (LLM)"
fi
echo "    profile    →  ${SELECTED_GPU_PROFILE} (${SELECTED_COMPOSE_FILE})"
echo "    streams    →  ${SELECTED_STREAM_COUNT} total, ${SELECTED_DENSITY_DISPLAY_STREAMS} density display"
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

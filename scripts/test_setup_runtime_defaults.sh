#!/usr/bin/env bash
# Created by Metrum AI for AMD

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/setup_runtime_defaults.sh
source "${REPO_ROOT}/scripts/setup_runtime_defaults.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_eq() {
    local expected="$1"
    local actual="$2"
    local label="$3"
    if [ "$actual" != "$expected" ]; then
        fail "${label}: expected '${expected}', got '${actual}'"
    fi
}

assert_contains_line() {
    local haystack="$1"
    local needle="$2"
    local label="$3"
    if ! printf '%s\n' "$haystack" | awk -v needle="$needle" '$0 == needle { found = 1 } END { exit found ? 0 : 1 }'; then
        fail "${label}: missing line '${needle}'"
    fi
}

assert_eq "2gpu" "$(select_gpu_profile 2)" "2 GPUs select original profile"
assert_eq "2gpu" "$(select_gpu_profile 3)" "3 GPUs fall back to original profile"
assert_eq "4gpu" "$(select_gpu_profile 4)" "4 GPUs select optimized profile"
assert_eq "4gpu" "$(select_gpu_profile 8)" "more than 4 GPUs select optimized profile"

assert_eq "docker-compose.profile-2gpu.yml" "$(profile_compose_file 2gpu)" "2-GPU compose file"
assert_eq "docker-compose.profile-4gpu.yml" "$(profile_compose_file 4gpu)" "4-GPU compose file"

sample_lscpu="$(
    cat <<'EOF'
# CPU,Core,Socket
0,0,0
1,0,0
2,1,0
3,1,0
4,0,1
5,0,1
EOF
)"
assert_eq "3" "$(physical_core_count_from_lscpu "$sample_lscpu")" "SMT siblings are counted once per physical core"

assert_eq "25" "$(select_stream_count 64 "")" "64 physical cores default to 25 streams"
assert_eq "50" "$(select_stream_count 65 "")" "65 physical cores default to 50 streams"
assert_eq "37" "$(select_stream_count 96 37)" "explicit STREAM_COUNT is respected"

assert_eq "25" "$(select_density_display_streams 25 "")" "density follows stream default"
assert_eq "12" "$(select_density_display_streams 25 12)" "explicit density stream count is respected"

profile_2gpu="$(profile_env_assignments 2gpu)"
assert_contains_line "$profile_2gpu" "COMPOSE_FILE=docker-compose.yml:docker-compose.profile-2gpu.yml" "2-GPU compose env"
assert_contains_line "$profile_2gpu" "GPU_COUNT=1" "2-GPU YOLO GPU count"
assert_contains_line "$profile_2gpu" "YOLO_START_GPU=0" "2-GPU YOLO start GPU"
assert_contains_line "$profile_2gpu" "DENSITY_PHYSICAL_GPU=1" "2-GPU density GPU"
assert_contains_line "$profile_2gpu" "LEMONADE_GPU_ID=1" "2-GPU Lemonade GPU"

profile_4gpu="$(profile_env_assignments 4gpu)"
assert_contains_line "$profile_4gpu" "COMPOSE_FILE=docker-compose.yml:docker-compose.profile-4gpu.yml" "4-GPU compose env"
assert_contains_line "$profile_4gpu" "GPU_COUNT=2" "4-GPU YOLO GPU count"
assert_contains_line "$profile_4gpu" "YOLO_START_GPU=1" "4-GPU YOLO start GPU"
assert_contains_line "$profile_4gpu" "DENSITY_PHYSICAL_GPU=0" "4-GPU density GPU"
assert_contains_line "$profile_4gpu" "LEMONADE_GPU_ID=3" "4-GPU Lemonade GPU"
assert_contains_line "$profile_4gpu" "YOLO_BATCH_SIZE=48" "4-GPU YOLO batch"
assert_contains_line "$profile_4gpu" "PAIRED_OUTPUT_FPS=15" "4-GPU paired output FPS"

assert_eq ".venv-model-export/bin/python" "$(model_export_python_path ".venv-model-export")" "model export uses venv python"
assert_eq ".custom-export-venv/bin/python" "$(model_export_python_path ".custom-export-venv")" "model export venv path is configurable"

echo "setup runtime defaults tests passed"

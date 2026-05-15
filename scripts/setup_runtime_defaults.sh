#!/usr/bin/env bash
# Created by Metrum AI for AMD

# Pure helpers used by setup.sh and by the shell characterization tests.

physical_core_count_from_lscpu() {
    local lscpu_output="$1"
    printf '%s\n' "$lscpu_output" | awk -F, '
        $0 !~ /^#/ && NF >= 2 {
            # Supports both `lscpu -p=Core,Socket` and wider forms such as
            # `lscpu -p=CPU,Core,Socket`. The physical core identity is the
            # final Core,Socket pair; SMT siblings share that pair.
            key = $(NF - 1) "," $NF
            seen[key] = 1
        }
        END {
            for (key in seen) {
                count++
            }
            print count + 0
        }
    '
}

detect_cpu_core_count() {
    local physical_cores
    if command -v lscpu >/dev/null 2>&1; then
        physical_cores="$(physical_core_count_from_lscpu "$(lscpu -p=Core,Socket 2>/dev/null)")"
        if [ "${physical_cores:-0}" -gt 0 ]; then
            echo "$physical_cores"
            return
        fi
    fi

    # Fallbacks report logical CPUs, but only run on minimal systems without
    # lscpu. Prefer a useful default over failing setup entirely.
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
        return
    fi
    echo 1
}

select_gpu_profile() {
    local gpu_count="$1"
    if [ "${gpu_count:-0}" -ge 4 ]; then
        echo "4gpu"
    else
        echo "2gpu"
    fi
}

profile_compose_file() {
    local profile="$1"
    case "$profile" in
        2gpu) echo "docker-compose.profile-2gpu.yml" ;;
        4gpu) echo "docker-compose.profile-4gpu.yml" ;;
        *) return 1 ;;
    esac
}

select_stream_count() {
    local cpu_cores="$1"
    local explicit_stream_count="${2:-}"
    if [ -n "$explicit_stream_count" ]; then
        echo "$explicit_stream_count"
        return
    fi
    if [ "${cpu_cores:-0}" -le 64 ]; then
        echo 25
    else
        echo 50
    fi
}

select_density_display_streams() {
    local stream_count="$1"
    local explicit_density_display_streams="${2:-}"
    if [ -n "$explicit_density_display_streams" ]; then
        echo "$explicit_density_display_streams"
    else
        echo "$stream_count"
    fi
}

profile_env_assignments() {
    local profile="$1"
    local compose_file
    compose_file="$(profile_compose_file "$profile")" || return 1

    case "$profile" in
        2gpu)
            cat <<EOF
SMARTCITY_GPU_PROFILE=2gpu
COMPOSE_FILE=docker-compose.yml:${compose_file}
ROCR_VISIBLE_DEVICES=0,1
HSA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
GPU_COUNT=1
YOLO_GPU_ID=0
DMCOUNT_GPU_ID=1
LEMONADE_GPU_ID=1
YOLO_START_GPU=0
YOLO_REPLICAS_PER_GPU=2
DENSITY_PHYSICAL_GPU=1
EOF
            ;;
        4gpu)
            cat <<EOF
SMARTCITY_GPU_PROFILE=4gpu
COMPOSE_FILE=docker-compose.yml:${compose_file}
ROCR_VISIBLE_DEVICES=0,1,2,3
HSA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
GPU_COUNT=2
YOLO_GPU_ID=1
DMCOUNT_GPU_ID=0
LEMONADE_GPU_ID=3
YOLO_START_GPU=1
YOLO_REPLICAS_PER_GPU=2
YOLO_BATCH_SIZE=48
YOLO_PREPROCESS_THREADS=2
PAIRED_OUTPUT_FPS=15
DENSITY_PHYSICAL_GPU=0
EOF
            ;;
        *)
            return 1
            ;;
    esac
}

model_export_python_path() {
    local venv_dir="${1:-.venv-model-export}"
    echo "${venv_dir}/bin/python"
}

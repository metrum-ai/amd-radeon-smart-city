#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

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

# detect_free_gpus  — rank all available GPUs by current load and return the
# N least-busy physical GPU indices as a space-separated string.
#
# Algorithm:
#   score(gpu) = gpu_use% + mem_use%   (lower = freer)
#   Sort GPUs by score ascending, take the top N, output their physical IDs
#   in ascending numeric order so ROCR_VISIBLE_DEVICES stays deterministic.
#
# Falls back to sequential IDs (0 .. N-1) when:
#   - rocm-smi is absent or returns no output
#   - total_gpus <= n_needed (no choice to be made)
#   - awk parsing produces no results
#
# Usage: detect_free_gpus <n_needed> <total_gpus>
#   n_needed   — how many GPUs to select (2 or 4)
#   total_gpus — total number of AMD GPUs detected on the host
# Prints a space-separated list, e.g. "2 3 5 7"
detect_free_gpus() {
    local n_needed="$1"
    local total_gpus="$2"
    local i fallback_ids=""

    # Build sequential fallback: "0 1 ... (min(n_needed,total_gpus)-1)"
    # Caps at total_gpus so we never reference physical IDs that don't exist.
    local _actual
    _actual=$(( total_gpus < n_needed ? total_gpus : n_needed ))
    for ((i = 0; i < _actual; i++)); do
        fallback_ids="${fallback_ids}${i} "
    done
    fallback_ids="${fallback_ids% }"  # trim trailing space

    # When there are no spare GPUs to choose from, return all available.
    if [ "${total_gpus:-0}" -le "${n_needed:-0}" ]; then
        echo "$fallback_ids"
        return
    fi

    # Query per-GPU compute and memory utilisation from rocm-smi.
    # Both calls are silenced; an empty result triggers the fallback.
    local gpu_use_raw mem_use_raw
    gpu_use_raw="$(rocm-smi --showuse 2>/dev/null)"    || gpu_use_raw=""
    mem_use_raw="$(rocm-smi --showmemuse 2>/dev/null)" || mem_use_raw=""

    if [ -z "$gpu_use_raw" ] && [ -z "$mem_use_raw" ]; then
        echo "$fallback_ids"
        return
    fi

    # Feed both outputs into a single awk pass.
    # Each rocm-smi line that references a GPU is expected in one of two forms:
    #   GPU[N]   : GPU use (%): 45          (--showuse verbose format)
    #   GPU[N]   : 45%                      (--showmemuse compact format)
    # We strip "%" signs, extract the GPU index and the last integer on the
    # line, and accumulate a total "load score" per GPU.  After scanning all
    # lines we sort by score ascending and return the N lowest (freest) IDs,
    # re-sorted numerically so ROCR_VISIBLE_DEVICES is always well-ordered.
    local selected_ids
    selected_ids="$(printf '%s\n%s\n' "$gpu_use_raw" "$mem_use_raw" | awk -v n="$n_needed" '
        /GPU\[[0-9]+\]/ {
            line = $0
            # Extract GPU index from GPU[N]
            start = index(line, "[")
            stop  = index(line, "]")
            if (start == 0 || stop == 0) next
            gpu_id = substr(line, start + 1, stop - start - 1)

            # Remove "%" so trailing numbers are plain integers
            gsub(/%/, "", line)

            # Grab the last integer token on the line
            if (match(line, /[0-9]+[[:space:]]*$/)) {
                pct = substr(line, RSTART, RLENGTH) + 0
                score[gpu_id] += pct
            }
        }
        END {
            # Collect GPU IDs into a sortable indexed array
            count = 0
            for (g in score) {
                ids[count++] = g
            }

            if (count == 0) exit 1   # no data — caller will use fallback

            # Bubble-sort ids[] by score ascending (N<=8 so O(N^2) is fine)
            for (i = 0; i < count - 1; i++) {
                for (j = i + 1; j < count; j++) {
                    if (score[ids[j]] < score[ids[i]]) {
                        tmp = ids[i]; ids[i] = ids[j]; ids[j] = tmp
                    }
                }
            }

            # Keep only the top-N (freest) IDs, then re-sort numerically
            # so the ROCR_VISIBLE_DEVICES list is in ascending physical order.
            limit = (count < n) ? count : n
            for (i = 0; i < limit; i++) sel[i] = ids[i] + 0

            for (i = 0; i < limit - 1; i++) {
                for (j = i + 1; j < limit; j++) {
                    if (sel[j] < sel[i]) { tmp = sel[i]; sel[i] = sel[j]; sel[j] = tmp }
                }
            }

            out = ""
            for (i = 0; i < limit; i++) out = out (i > 0 ? " " : "") sel[i]
            print out
        }
    ')"

    if [ -z "$selected_ids" ]; then
        echo "$fallback_ids"
        return
    fi

    echo "$selected_ids"
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

host_group_gid() {
    local group_name="$1"
    local fallback_gid="$2"
    local group_entry

    group_entry="$(getent group "$group_name" 2>/dev/null || true)"
    if [ -n "$group_entry" ]; then
        echo "$group_entry" | awk -F: '{print $3}'
        return
    fi

    echo "$fallback_gid"
}

profile_env_assignments() {
    local profile="$1"
    # Optional: space-separated physical GPU IDs selected by detect_free_gpus,
    # e.g. "2 3 5 7".  When provided, ROCR_VISIBLE_DEVICES and
    # HSA_VISIBLE_DEVICES are set to these physical IDs (comma-separated).
    # All role indices (YOLO_GPU_ID, DMCOUNT_GPU_ID, etc.) remain as relative
    # indices into ROCR_VISIBLE_DEVICES — no change needed there.
    local selected_gpu_ids="${2:-}"
    local compose_file
    compose_file="$(profile_compose_file "$profile")" || return 1

    # Convert space-separated IDs ("2 3 5 7") → comma-separated ("2,3,5,7").
    # Falls back to the profile default when no IDs were supplied.
    local rocr_devices
    if [ -n "$selected_gpu_ids" ]; then
        rocr_devices="$(echo "$selected_gpu_ids" | tr ' ' ',')"
    fi

    case "$profile" in
        2gpu)
            local rocr="${rocr_devices:-0,1}"
            cat <<EOF
SMARTCITY_GPU_PROFILE=2gpu
COMPOSE_FILE=docker-compose.yml:${compose_file}
ROCR_VISIBLE_DEVICES=${rocr}
HSA_VISIBLE_DEVICES=${rocr}
NUM_GPUS=2
GPU_COUNT=1
YOLO_GPU_ID=0
DMCOUNT_GPU_ID=1
LEMONADE_GPU_ID=1
YOLO_START_GPU=0
YOLO_REPLICAS_PER_GPU=2
DENSITY_PHYSICAL_GPU=1
DECODE_MODE=software
HW_DECODE_VAAPI_GPU=1
HW_DECODE_MAX_STREAMS=0
HW_DECODE_STAGGER_MS=0
EOF
            ;;
        4gpu)
            local rocr="${rocr_devices:-0,1,2,3}"
            cat <<EOF
SMARTCITY_GPU_PROFILE=4gpu
COMPOSE_FILE=docker-compose.yml:${compose_file}
ROCR_VISIBLE_DEVICES=${rocr}
HSA_VISIBLE_DEVICES=${rocr}
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
HW_DECODE_VAAPI_GPU=0
HW_DECODE_MAX_STREAMS=
HW_DECODE_STAGGER_MS=200
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

# select_decode_mode — choose VA-API hardware decode vs libav software decode
# based on the runtime profile and host kernel version. 2-GPU runs always use
# software decode because GPU 1 is already shared by DM-Count + Lemonade and has
# no VCN headroom. 4-GPU can use VA-API hardware decode on kernel 6.17+.
#
# Arguments:
#   $1 — kernel version string (e.g. "6.17.0-29-generic" from `uname -r`).
#        Tests pass synthetic strings here; setup.sh passes `uname -r`.
#   $2 — optional explicit override. When non-empty (and not "auto") it is
#        honored for 4-GPU profiles, so a user can force a value via env var.
#        2-GPU profiles still force software decode.
#   $3 — runtime GPU profile ("2gpu" or "4gpu").
#
# Echoes "hardware" or "software".
select_decode_mode() {
    local kernel_release="${1:-}"
    local explicit="${2:-}"
    local profile="${3:-4gpu}"

    if [ "$profile" = "2gpu" ]; then
        echo "software"
        return
    fi

    if [ -n "$explicit" ] && [ "$explicit" != "auto" ]; then
        echo "$explicit"
        return
    fi

    # Strip everything after the first non-numeric/non-dot character so
    # "6.17.0-29-generic" becomes "6.17.0", then split on "." for compare.
    local sanitized
    sanitized="$(printf '%s' "$kernel_release" | sed 's/[^0-9.].*$//')"
    local major minor
    major="$(printf '%s' "$sanitized" | cut -d. -f1)"
    minor="$(printf '%s' "$sanitized" | cut -d. -f2)"
    major="${major:-0}"
    minor="${minor:-0}"

    if [ "$major" -gt 6 ] || { [ "$major" -eq 6 ] && [ "$minor" -ge 17 ]; }; then
        echo "hardware"
    else
        echo "software"
    fi
}

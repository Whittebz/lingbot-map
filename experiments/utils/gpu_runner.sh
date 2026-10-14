#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
    echo "Usage: gpu_runner.sh command [args...]" >&2
    exit 2
fi

umask 077
LOCK_DIR="${GPU_SELECTOR_LOCK_DIR:-${TMPDIR:-/tmp}/lingbot-map-gpu-locks-${UID}}"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/selector.lock"

while :; do
    flock -x 9

    if ! gpu_rows=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits); then
        echo "ERROR: unable to query GPU memory; refusing to select a GPU" >&2
        exit 1
    fi
    if ! process_rows=$(nvidia-smi pmon -c 1); then
        echo "ERROR: unable to query GPU processes; refusing to select a GPU" >&2
        exit 1
    fi

    declare -A busy_gpu=()
    while read -r gpu_id; do
        [[ -n "$gpu_id" ]] && busy_gpu["$gpu_id"]=1
    done < <(awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2 > 0 {print $1}' <<< "$process_rows")

    candidates=()
    while IFS=',' read -r gpu_id memory_used; do
        gpu_id="${gpu_id//[[:space:]]/}"
        memory_used="${memory_used//[[:space:]]/}"
        if [[ ! "$gpu_id" =~ ^[0-9]+$ || ! "$memory_used" =~ ^[0-9]+$ ]]; then
            echo "ERROR: malformed GPU query output; refusing to select a GPU" >&2
            exit 1
        fi
        if ((memory_used == 0)) && [[ -z "${busy_gpu[$gpu_id]+x}" ]]; then
            candidates+=("$gpu_id")
        fi
    done <<< "$gpu_rows"

    selected_gpu=""
    selected_lock_fd=""
    if ((${#candidates[@]} > 0)); then
        start=$((RANDOM % ${#candidates[@]}))
        for ((offset = 0; offset < ${#candidates[@]}; offset++)); do
            gpu_id="${candidates[$(((start + offset) % ${#candidates[@]}))]}"
            exec {gpu_lock_fd}>"$LOCK_DIR/gpu-${gpu_id}.lock"
            if flock -n "$gpu_lock_fd"; then
                selected_gpu="$gpu_id"
                selected_lock_fd="$gpu_lock_fd"
                break
            fi
            exec {gpu_lock_fd}>&-
        done
    fi

    if [[ -n "$selected_gpu" ]]; then
        flock -u 9
        export CUDA_VISIBLE_DEVICES="$selected_gpu"
        echo "GPU selector assigned physical GPU ${selected_gpu}; lock held for command lifetime" >&2
        if "$@"; then
            status=0
        else
            status=$?
        fi
        flock -u "$selected_lock_fd"
        exec {selected_lock_fd}>&-
        exit "$status"
    fi

    flock -u 9
    echo "No GPU is fully idle (0 MiB and no process); retrying in 300 seconds" >&2
    sleep 300
done

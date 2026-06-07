#!/usr/bin/env bash
# VRAM watchdog: kill target process if GPU memory exceeds threshold.
# Usage: vram_watchdog.sh <gpu_index> <threshold_pct> <target_pid> <flag_file>
set -u

GPU_IDX="${1:?gpu index}"
THRESHOLD="${2:?threshold pct}"
TARGET_PID="${3:?target pid}"
FLAG_FILE="${4:?flag file}"

while kill -0 "$TARGET_PID" 2>/dev/null; do
    read -r used total <<<"$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$GPU_IDX" | tr ',' ' ')"
    if [ -n "${used:-}" ] && [ -n "${total:-}" ]; then
        pct=$(( used * 100 / total ))
        if [ "$pct" -ge "$THRESHOLD" ]; then
            {
                echo "WATCHDOG_KILL $(date '+%Y-%m-%d %H:%M:%S %Z')"
                echo "gpu=$GPU_IDX used=${used}MiB total=${total}MiB pct=${pct}%"
                echo "killed_pid=$TARGET_PID"
            } > "$FLAG_FILE"
            kill -9 "$TARGET_PID" 2>/dev/null
            exit 1
        fi
    fi
    sleep 5
done
exit 0

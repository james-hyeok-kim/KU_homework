#!/usr/bin/env bash
# 5-way comparison runner (Chaemin-aligned, GPU 0, NF4).
# 16 GPU runs (4 modes x 4 datasets) + Method 3 copied from Chaemin results.
# A VRAM watchdog kills the python process if GPU 0 memory >= 98%.
set -u

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SRC_DIR")"
RESULTS="$ROOT/results/comparison_v2"
CHAEMIN="$(dirname "$ROOT")/AI-Final-project_chaemin"
LOG_DIR="$RESULTS/logs"
FLAG_FILE="$RESULTS/WATCHDOG_KILLED"
mkdir -p "$RESULTS" "$LOG_DIR"
rm -f "$FLAG_FILE"

export CUDA_VISIBLE_DEVICES=0
GPU_IDX=0
THRESHOLD=98

DATASETS=(InfoVQA ChartQA MP-DocVQA SlideVQA)
MODES=(image_only ocr_text_only ocr_text_image selective_llm)

echo "=== 5-way comparison run start: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

# Method 3 (Closed-VDU): copy Chaemin's parsed_text_only results (no GPU).
for ds in "${DATASETS[@]}"; do
    src="$CHAEMIN/results/v12_on_visrag/today/${ds}_parsed_text_only_top1_first100.json"
    dst="$RESULTS/${ds}_closed_text_only.json"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "[method3] copied $ds parsed_text_only -> $(basename "$dst")"
    fi
done

# GPU runs: 4 modes x 4 datasets, sequential on GPU 0.
for mode in "${MODES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        out="$RESULTS/${ds}_${mode}.json"
        if [ -f "$out" ]; then
            echo "[skip] $ds/$mode (exists)"
            continue
        fi
        log="$LOG_DIR/${ds}_${mode}.log"
        echo "=== [$(date '+%H:%M:%S')] $ds / $mode ==="
        python3 "$SRC_DIR/run_comparison_v2.py" --dataset "$ds" --mode "$mode" >"$log" 2>&1 &
        PY_PID=$!
        bash "$SRC_DIR/vram_watchdog.sh" "$GPU_IDX" "$THRESHOLD" "$PY_PID" "$FLAG_FILE" &
        WD_PID=$!
        wait "$PY_PID"
        STATUS=$?
        kill "$WD_PID" 2>/dev/null
        wait "$WD_PID" 2>/dev/null
        if [ -f "$FLAG_FILE" ]; then
            echo "!!! WATCHDOG KILLED $ds/$mode — VRAM >= ${THRESHOLD}% !!!"
            cat "$FLAG_FILE"
            exit 2
        fi
        if [ "$STATUS" -ne 0 ]; then
            echo "!!! FAILED $ds/$mode (exit $STATUS) — see $log (tail below)"
            tail -n 20 "$log"
            exit 1
        fi
        tail -n 3 "$log"
    done
done

echo "=== all runs done: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
python3 "$SRC_DIR/make_table_v2.py"

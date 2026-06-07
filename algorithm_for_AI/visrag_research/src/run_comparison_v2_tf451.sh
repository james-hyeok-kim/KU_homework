#!/usr/bin/env bash
# 5-way comparison under transformers 4.51 (Chaemin-matched env), GPU 1.
# Same 16 GPU runs as run_comparison_v2.sh but: venv python, separate results
# dir (comparison_v2_tf451), separate OCR/route caches. Method 3 copied from
# Chaemin results as before.
set -u

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SRC_DIR")"
RESULTS="$ROOT/results/comparison_v2_tf451"
CHAEMIN="$(dirname "$ROOT")/AI-Final-project_chaemin"
LOG_DIR="$RESULTS/logs"
FLAG_FILE="$RESULTS/WATCHDOG_KILLED"
PYTHON_BIN="/data/jameskimh/venvs/tf451/bin/python"
mkdir -p "$RESULTS" "$LOG_DIR"
rm -f "$FLAG_FILE"

export CUDA_VISIBLE_DEVICES=1
export COMPARISON_RESULTS_DIR="$RESULTS"
GPU_IDX=1
THRESHOLD=98

DATASETS=(InfoVQA ChartQA MP-DocVQA SlideVQA)
MODES=(image_only ocr_text_only ocr_text_image selective_llm)

echo "=== tf451 5-way comparison run start: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
"$PYTHON_BIN" -c "import transformers; print('transformers:', transformers.__version__)"

# Method 3 (Closed-VDU): copy Chaemin's parsed_text_only results (no GPU).
for ds in "${DATASETS[@]}"; do
    src="$CHAEMIN/results/v12_on_visrag/today/${ds}_parsed_text_only_top1_first100.json"
    dst="$RESULTS/${ds}_closed_text_only.json"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "[method3] copied $ds parsed_text_only -> $(basename "$dst")"
    fi
done

for mode in "${MODES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        out="$RESULTS/${ds}_${mode}.json"
        if [ -f "$out" ]; then
            echo "[skip] $ds/$mode (exists)"
            continue
        fi
        log="$LOG_DIR/${ds}_${mode}.log"
        echo "=== [$(date '+%H:%M:%S')] tf451 $ds / $mode ==="
        "$PYTHON_BIN" "$SRC_DIR/run_comparison_v2.py" --dataset "$ds" --mode "$mode" >"$log" 2>&1 &
        PY_PID=$!
        bash "$SRC_DIR/vram_watchdog.sh" "$GPU_IDX" "$THRESHOLD" "$PY_PID" "$FLAG_FILE" &
        WD_PID=$!
        wait "$PY_PID"
        STATUS=$?
        kill "$WD_PID" 2>/dev/null
        wait "$WD_PID" 2>/dev/null
        if [ -f "$FLAG_FILE" ]; then
            echo "!!! WATCHDOG KILLED tf451 $ds/$mode — VRAM >= ${THRESHOLD}% !!!"
            cat "$FLAG_FILE"
            exit 2
        fi
        if [ "$STATUS" -ne 0 ]; then
            echo "!!! FAILED tf451 $ds/$mode (exit $STATUS) — see $log (tail below)"
            tail -n 20 "$log"
            exit 1
        fi
        tail -n 3 "$log"
    done
done

echo "=== tf451 all runs done: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

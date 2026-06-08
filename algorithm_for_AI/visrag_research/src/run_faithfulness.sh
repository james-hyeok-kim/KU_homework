#!/usr/bin/env bash
# Faithfulness evaluation: tf457 and tf451 using Qwen2.5-VL-72B judge on GPU 2,3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/results/faithfulness_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S KST')] $*"; }

log "=== Faithfulness Eval START ==="
log "GPU 2,3 | Qwen2.5-VL-72B | VRAM watchdog 98%"

CUDA_VISIBLE_DEVICES=2,3 python3 "$ROOT/src/faithfulness_eval_v2.py" \
    --result-dir "$ROOT/results/comparison_v2" \
    --physical-gpus 2 3 \
    2>&1 | tee "$LOG_DIR/tf457_${TS}.log"

log "=== tf457 DONE, starting tf451 ==="

CUDA_VISIBLE_DEVICES=2,3 python3 "$ROOT/src/faithfulness_eval_v2.py" \
    --result-dir "$ROOT/results/comparison_v2_tf451" \
    --physical-gpus 2 3 \
    2>&1 | tee "$LOG_DIR/tf451_${TS}.log"

log "=== ALL DONE ==="

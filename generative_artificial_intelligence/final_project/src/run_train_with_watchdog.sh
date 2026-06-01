#!/bin/bash
# GLOW training + GPU OOM watchdog
# Kills training if GPU memory > 98%, sends push notification

GPU_ID=${1:-1}
OOM_THRESHOLD=98  # percent
POLL_SEC=10
LOG_DIR="/data/jameskimh/final_project/logs"
mkdir -p "$LOG_DIR"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PID_FILE="$LOG_DIR/glow_train.pid"
KILLED_FLAG="$LOG_DIR/glow_killed.flag"
rm -f "$KILLED_FLAG"

echo "[watchdog] Starting GLOW training on GPU $GPU_ID"
echo "[watchdog] OOM threshold: $OOM_THRESHOLD%"
echo "[watchdog] Log: $LOG_DIR/glow_train.log"

# Start training in background
CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$SRC_DIR/train.py" \
    --n_blocks 4 \
    --n_flows 32 \
    --hidden 512 \
    --batch_size 16 \
    --lr 1e-4 \
    --n_iter 200000 \
    --save_every 5000 \
    --sample_every 2000 \
    --n_sample 16 \
    --temperature 0.7 \
    --resume_latest \
    >> "$LOG_DIR/glow_train.log" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$TRAIN_PID_FILE"
echo "[watchdog] Training PID: $TRAIN_PID"

# Watchdog loop
while kill -0 $TRAIN_PID 2>/dev/null; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d ' ')
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU_ID | tr -d ' ')
    PCT=$(( MEM_USED * 100 / MEM_TOTAL ))

    if [ "$PCT" -ge "$OOM_THRESHOLD" ]; then
        echo "[watchdog] OOM! GPU $GPU_ID memory ${PCT}% (${MEM_USED}/${MEM_TOTAL} MiB) >= ${OOM_THRESHOLD}% — killing PID $TRAIN_PID"
        kill -9 $TRAIN_PID 2>/dev/null || true
        echo "KILLED_OOM pct=${PCT} mem=${MEM_USED}/${MEM_TOTAL}MiB" > "$KILLED_FLAG"
        break
    fi

    sleep $POLL_SEC
done

if [ -f "$KILLED_FLAG" ]; then
    echo "[watchdog] Training KILLED by OOM watchdog: $(cat $KILLED_FLAG)"
    exit 2
else
    echo "[watchdog] Training ended normally."
    tail -5 "$LOG_DIR/glow_train.log"
    exit 0
fi

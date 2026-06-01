#!/bin/bash
# GLOW training launcher — logs saved to /data/jameskimh/final_project/
set -e

LOG_DIR="/data/jameskimh/final_project/logs"
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")"

echo "Starting GLOW training from scratch on FFHQ-64x64"
echo "Checkpoints → /data/jameskimh/final_project/glow_pretrained/"
echo "Samples     → /data/jameskimh/final_project/samples/glow_train/"
echo "Log         → $LOG_DIR/glow_train.log"

python3 train.py \
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
    2>&1 | tee "$LOG_DIR/glow_train.log"

#!/usr/bin/env bash
# run_experiment.sh — Sequential execution of all three experimental stages.
# Usage: bash run_experiment.sh [--seed SEED] [--data DATA_PATH] [--ckpt CKPT_DIR]

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------
SEED=42
DATA_PATH="./data"
CKPT_DIR="./checkpoints"
RESULTS_DIR="./results"
LOG_DIR="./logs"
TOKEN_BUDGET=2048
DEVICE="cuda"
DATASETS="slidevqa docvqa chartqa"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)      SEED="$2"; shift 2 ;;
        --data)      DATA_PATH="$2"; shift 2 ;;
        --ckpt)      CKPT_DIR="$2"; shift 2 ;;
        --results)   RESULTS_DIR="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --budget)    TOKEN_BUDGET="$2"; shift 2 ;;
        *)           echo "Unknown flag: $1"; exit 1 ;;
    esac
done

EXP_CKPT="${CKPT_DIR}/exp001_seed${SEED}"
mkdir -p "${EXP_CKPT}/stage1" "${EXP_CKPT}/stage2" "${RESULTS_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/exp001_seed${SEED}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo " Experiment 001: Query-Conditioned Visual Token Pruning"
echo " Seed=${SEED}  Budget=${TOKEN_BUDGET}  Device=${DEVICE}"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "============================================================"

# ---------------------------------------------------------------------------
# GPU monitoring in background (log every 30 s)
# ---------------------------------------------------------------------------
GPU_LOG="${LOG_DIR}/gpu_exp001_seed${SEED}.log"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 30 >> "${GPU_LOG}" &
GPU_MON_PID=$!
echo "GPU monitor PID: ${GPU_MON_PID} → ${GPU_LOG}"

cleanup() {
    kill "${GPU_MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Stage 1 — Sanity Check (100 samples)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 1: Sanity Check"
echo "    $(date '+%H:%M:%S KST')"

python train.py \
    --stage 1 \
    --data_path "${DATA_PATH}" \
    --dataset slidevqa \
    --max_steps 50 \
    --lr 1e-4 \
    --lambda_hinge 0.1 \
    --token_budget "${TOKEN_BUDGET}" \
    --batch_size 2 \
    --grad_accum 2 \
    --checkpoint_dir "${EXP_CKPT}" \
    --sanity_check \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "    Stage 1 sanity check complete: $(date '+%H:%M:%S KST')"

# ---------------------------------------------------------------------------
# Stage 2 — Subset training (SlideVQA, DocVQA, ChartQA val-20%)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 2: Selector Training (subset, 2000 steps)"
echo "    $(date '+%H:%M:%S KST')"

for DATASET in slidevqa docvqa chartqa; do
    echo "  Training on ${DATASET}..."
    python train.py \
        --stage 1 \
        --data_path "${DATA_PATH}" \
        --dataset "${DATASET}" \
        --max_steps 2000 \
        --lr 1e-4 \
        --lambda_hinge 0.1 \
        --token_budget "${TOKEN_BUDGET}" \
        --batch_size 4 \
        --grad_accum 4 \
        --checkpoint_dir "${EXP_CKPT}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
done

echo "  Evaluating subset (val 20%)..."
python evaluate.py \
    --checkpoint "${EXP_CKPT}/stage1/final.pt" \
    --datasets ${DATASETS} \
    --data_path "${DATA_PATH}" \
    --split val \
    --token_budget "${TOKEN_BUDGET}" \
    --batch_size 8 \
    --output_json "${RESULTS_DIR}/exp001_stage2_seed${SEED}.json" \
    --device "${DEVICE}"

echo "    Stage 2 complete: $(date '+%H:%M:%S KST')"

# ---------------------------------------------------------------------------
# Stage 3 — Full benchmark (test split, optional Stage 2 end-to-end)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 3: Full Training + Evaluation"
echo "    $(date '+%H:%M:%S KST')"

# Full Stage 1 training
for DATASET in slidevqa docvqa chartqa; do
    python train.py \
        --stage 1 \
        --data_path "${DATA_PATH}" \
        --dataset "${DATASET}" \
        --max_steps 10000 \
        --lr 1e-4 \
        --lambda_hinge 0.1 \
        --token_budget "${TOKEN_BUDGET}" \
        --batch_size 4 \
        --grad_accum 4 \
        --checkpoint_dir "${EXP_CKPT}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
done

# Optional: Stage 2 end-to-end (enable if VRAM >= 40GB)
# python train.py \
#     --stage 2 \
#     --data_path "${DATA_PATH}" \
#     --dataset slidevqa \
#     --max_steps 5000 \
#     --lr 5e-6 \
#     --lambda_hinge 0.05 \
#     --token_budget "${TOKEN_BUDGET}" \
#     --batch_size 2 \
#     --grad_accum 16 \
#     --checkpoint_dir "${EXP_CKPT}" \
#     --device "${DEVICE}" \
#     --seed "${SEED}"

# Full evaluation on test split
python evaluate.py \
    --checkpoint "${EXP_CKPT}/stage1/final.pt" \
    --datasets ${DATASETS} \
    --data_path "${DATA_PATH}" \
    --split test \
    --token_budget "${TOKEN_BUDGET}" \
    --batch_size 8 \
    --output_json "${RESULTS_DIR}/exp001_final_seed${SEED}.json" \
    --device "${DEVICE}"

# K sweep: ablation A5 (SlideVQA only)
echo ""
echo ">>> A5 Ablation: K sweep (K=3,5,7 with same budget)"
for K in 3 5 7; do
    python evaluate.py \
        --checkpoint "${EXP_CKPT}/stage1/final.pt" \
        --datasets slidevqa \
        --data_path "${DATA_PATH}" \
        --split test \
        --token_budget "${TOKEN_BUDGET}" \
        --batch_size 8 \
        --output_json "${RESULTS_DIR}/exp001_ksweep_K${K}_seed${SEED}.json" \
        --device "${DEVICE}"
    # NOTE: pass --top_k ${K} once evaluate.py exposes that argument
done

echo ""
echo "============================================================"
echo " All stages complete."
echo " Results → ${RESULTS_DIR}/"
echo " End: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "============================================================"

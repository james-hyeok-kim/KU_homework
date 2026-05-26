# GPU/CPU 오케스트레이션 계획 — Experiment 001: QCVTP

**작성일**: 2026-05-26 KST  
**대상 실험**: `experiment_001.md` (Query-Conditioned Visual Token Pruning)

---

## 1. 환경 요약

### GPU

| GPU | 모델 | VRAM | 현재 사용 | 사용 가능 |
|-----|------|------|-----------|-----------|
| 0 | NVIDIA B200 | 183 GB | 76,264 MiB (사용 중, python 프로세스) | 약 104 GB 여유 |
| 1 | NVIDIA B200 | 183 GB | 76,264 MiB (사용 중, python 프로세스) | 약 104 GB 여유 |
| 2 | NVIDIA B200 | 183 GB | 1 MiB | **완전 여유 (183 GB)** |
| 3 | NVIDIA B200 | 183 GB | 1 MiB | **완전 여유 (183 GB)** |

- GPU 0, 1은 기존 작업 점유 중 → **GPU 2, 3을 실험 전용으로 사용**
- B200은 BF16 native 지원 (fp8도 지원), CUDA 13.0

### CPU / RAM

| 항목 | 값 |
|------|----|
| CPU 코어 수 | 288 vCPU |
| 총 RAM | 2.2 TiB |
| 사용 중 RAM | 441 GiB |
| 사용 가능 RAM | 1.8 TiB |

---

## 2. Batch Size 최적화

### VRAM 기준 메모리 추정 (GPU 2 or 3 단독 사용, 183 GB)

| 컴포넌트 | 메모리 추정 |
|---------|-----------|
| MiniCPM-V-2.6 (bfloat16) | ~16 GB |
| Visual token selector (W_q, W_v, ~2M params) | ~8 MB (무시 가능) |
| Batch activations: B=16, K=5, N=1024 tokens, D=1152 (bfloat16) | ~16 × 5 × 1024 × 1152 × 2 bytes ≈ 189 GB → 분할 필요 |
| B=8, K=5, N=1024 | ~94 GB → 실현 가능 |
| B=4, K=5, N=1024 | ~47 GB → 안전 마진 확보 |
| B=8, K=3, N=1024 | ~57 GB → 안전 |

### 권장 Batch Size 설정

| Stage | K값 | 권장 batch_size | gradient_accumulation | effective batch |
|-------|-----|----------------|----------------------|-----------------|
| Stage 1 (sanity) | 5 | 4 | ×4 | 16 |
| Stage 2 (subset) | 5 | 8 | ×2 | 16 |
| Stage 2 (subset) | 3 | 8 | ×2 | 16 |
| Stage 3 (full K=5) | 5 | 8 | ×2 | 16 |
| Stage 3 K=7 sweep | 7 | 4 | ×4 | 16 |

### Mixed Precision 권장

- **BF16 강력 권장**: B200은 BF16 native (A100 대비 2배 처리량), NaN 안정성 우수
- `torch.autocast('cuda', dtype=torch.bfloat16)` 사용
- Gradient는 FP32 유지 (selector 학습 시 안정성)
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True` 설정

---

## 3. 병렬화 전략

### GPU 할당

```
GPU 2: Stage 2 학습 + Ablation 실험 (selector 학습)
GPU 3: Baseline inference 병렬 실행 (vanilla VisRAG, Cosine Selector 평가)
```

GPU 2와 GPU 3을 독립 프로세스로 동시 실행 → Stage 2 baseline 측정과 학습을 **동시에** 진행

### Stage 3 (Full): DDP vs DataParallel

- **권장: DDP (DistributedDataParallel)** on GPU 2 + GPU 3
- 이유: 183 GB × 2 = 사실상 무제한 VRAM, DDP는 gradient sync가 DataParallel보다 효율적
- K=7 sweep 시 GPU 2 + 3 DDP로 학습 시간 절반 단축

```bash
# DDP 실행 예시 (GPU 2, 3)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --use_ddp \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --bf16
```

### DataLoader num_workers 설정

| 환경 | 권장 num_workers |
|------|----------------|
| Stage 1 (sanity, 100 samples) | 4 |
| Stage 2 (subset) | 16 |
| Stage 3 (full) | 32 |

- 288 vCPU 환경이므로 num_workers=32까지 여유 있음
- `pin_memory=True`, `prefetch_factor=4` 추가

---

## 4. 실험 실행 순서

### Stage 1: Sanity Check

**목표**: forward pass 오류 없이 100 샘플 처리 확인  
**GPU**: GPU 2 (단독)  
**예상 시간**: 10~20분 (B200은 A100보다 2~3배 빠름)

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --stage sanity \
    --dataset slidevqa \
    --num_samples 100 \
    --batch_size 4 \
    --K 5 \
    --budget 2048 \
    --bf16 \
    --output_dir checkpoints/exp001_sanity
```

**검증 항목**:
- pruned_tokens shape: `(B, K, k_keep, D)` — k_keep = int(r × N)
- VQA loss: 유한값 (NaN/Inf 없음)
- r 값이 K=5 → 0.4, K=3 → 0.67로 동적 계산됨을 로그 확인

---

### Stage 2: Subset Evaluation

**총 예상 시간**: 1.5~3시간 (B200 2장 병렬)

#### Step 2-1, 2-2: Baseline 측정 (GPU 3, inference only)

```bash
# GPU 3에서 병렬 실행
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --method vanilla \
    --dataset slidevqa docvqa chartqa \
    --split val \
    --subset_ratio 0.2 \
    --batch_size 16 \
    --bf16 &

CUDA_VISIBLE_DEVICES=3 python evaluate.py \
    --method cosine_selector \
    --dataset slidevqa docvqa chartqa \
    --split val \
    --subset_ratio 0.2 \
    --batch_size 16 \
    --bf16 &
```

**예상 시간**: 30~60분 (GPU 3)

#### Step 2-3: Token Selector 학습 (GPU 2)

```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
    --stage 1 \
    --dataset slidevqa docvqa \
    --split train \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --max_steps 2000 \
    --lambda_hinge 0.1 \
    --hinge_margin 0.5 \
    --K 5 \
    --budget 2048 \
    --bf16 \
    --gradient_checkpointing \
    --num_workers 16 \
    --save_every 500 \
    --eval_every 500 \
    --output_dir checkpoints/exp001_stage1
```

**예상 시간**: 1~2시간 (GPU 2, B200 기준 A100 대비 2~3배 단축)

#### Step 2-4, 2-5: 학습된 Selector 평가 + Ablation (GPU 2, 학습 완료 후)

```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --method qcvtp \
    --checkpoint checkpoints/exp001_stage1/best.pt \
    --dataset slidevqa docvqa chartqa \
    --split val --subset_ratio 0.2 \
    --batch_size 16 --bf16

# Ablation A1~A3 (inference-only variants)
for ablation in no_hinge fixed_r_0.5 fixed_r_0.25; do
    CUDA_VISIBLE_DEVICES=2 python evaluate.py \
        --method qcvtp_ablation --ablation $ablation \
        --checkpoint checkpoints/exp001_stage1/best.pt \
        --dataset slidevqa --split val --subset_ratio 0.2 --bf16
done
```

**예상 시간**: 30~60분 (GPU 2)

---

### Stage 3: Full Benchmark

**총 예상 시간**: 8~16시간 (GPU 2 + 3 DDP, B200 기준)

#### Step 3-1: Full 학습 (GPU 2 + 3, DDP)

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --stage 1 \
    --dataset slidevqa docvqa chartqa \
    --split train \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --max_steps 10000 \
    --lambda_hinge 0.1 \
    --K 5 \
    --budget 2048 \
    --bf16 \
    --gradient_checkpointing \
    --num_workers 32 \
    --save_every 1000 \
    --eval_every 1000 \
    --output_dir checkpoints/exp001_full_seed0
```

**예상 시간**: 4~6시간 (GPU 2+3 DDP)

#### Step 3-2: 3 Seeds 반복 (병렬)

```bash
# Seed 0: GPU 2+3 DDP (위 명령)
# Seed 1: GPU 2 단독 (seed 0 완료 후 또는 별도 스케줄)
# Seed 2: GPU 3 단독

for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$((seed % 2 + 2)) python train.py \
        --seed $seed \
        --max_steps 10000 \
        --output_dir checkpoints/exp001_full_seed${seed} \
        --bf16 &
done
```

#### Step 3-3: K Sweep Ablation (A5, inference-only)

```bash
for K in 3 5 7; do
    CUDA_VISIBLE_DEVICES=2 python evaluate.py \
        --method qcvtp \
        --checkpoint checkpoints/exp001_full_seed0/best.pt \
        --dataset slidevqa \
        --split test \
        --K $K \
        --budget 2048 \
        --bf16
done
```

---

## 5. 모니터링 전략

### GPU 실시간 모니터링

```bash
# 터미널 1: 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 더 상세한 모니터링 (GPU 2, 3만)
nvidia-smi dmon -s pucvmet -d 5 -i 2,3 | tee gpu_monitor.log
```

### 학습 중 메모리 프로파일링

```python
# 학습 코드 내 삽입 (첫 배치 후)
if step == 1:
    print(f"GPU 2 allocated: {torch.cuda.memory_allocated(2) / 1e9:.1f} GB")
    print(f"GPU 2 reserved:  {torch.cuda.memory_reserved(2) / 1e9:.1f} GB")
```

### 메모리 누수 탐지

```bash
# 10 steps마다 메모리 증가 여부 확인
# 학습 로그에서 'GPU memory' 컬럼 추적
# 증가세가 지속되면 DataLoader의 persistent_workers=False로 전환
```

### 체크포인트 저장 주기

| Stage | 저장 주기 | 보존 정책 |
|-------|----------|----------|
| Stage 1 학습 (2K steps) | 매 500 steps | 최근 3개 + best |
| Stage 3 학습 (10K steps) | 매 1,000 steps | 최근 3개 + best |
| Evaluation | 결과 JSON 즉시 저장 | 전체 보존 |

---

## 6. OOM 방지 전략

### Gradient Checkpointing

```python
# MiniCPM-V generator (frozen이지만 visual encoding 시 activation 큼)
model.gradient_checkpointing_enable()
# selector는 경량 (~2M params)이므로 불필요
```

### Batch Size 자동 축소 로직

```python
def safe_forward(model, batch, batch_size):
    try:
        return model(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        # batch를 반으로 나눠 재시도
        half = batch_size // 2
        if half < 1:
            raise
        out1 = model(batch[:half])
        out2 = model(batch[half:])
        return torch.cat([out1, out2])
```

### B200 환경 특이사항

- 183 GB VRAM: 단일 GPU로도 K=7, B=8까지 충분히 처리 가능
- 실질적 OOM 위험: **낮음** (실험 설계 기준 30~35 GB vs 183 GB 여유)
- K=10 이상의 극단적 sweep 시에도 단일 GPU에서 처리 가능
- CPU offload는 **불필요** (VRAM이 충분)

### 실제 OOM 발생 시 (비상 대응)

```bash
# 1단계: batch_size 절반 감소
# 2단계: gradient_accumulation 배로 증가
# 3단계: K값 줄이기 (K=7→5)
# B200에서는 4단계까지 갈 일이 없어야 함
```

---

## 7. 예상 총 실행 시간

B200 성능은 A100 80GB 대비 약 2~3배 (FLOPS 기준 TF32: A100 312 TFLOPS vs B200 ~900 TFLOPS).

| Stage | A100 기준 | B200 단독 | B200 DDP(2장) |
|-------|----------|----------|--------------|
| Stage 1 (sanity, 100 samples) | 30~60분 | 10~20분 | — |
| Stage 2 baseline inference | 1~2시간 | 30~60분 | — |
| Stage 2 학습 (2K steps) | 2~4시간 | 1~2시간 | — |
| Stage 2 ablation eval | 1~2시간 | 30~60분 | — |
| Stage 3 full 학습 (10K steps × 1 seed) | 12~24시간 | 4~8시간 | 2~4시간 |
| Stage 3 × 3 seeds (병렬) | 12~24시간 | 12~24시간 | 4~8시간 |
| Stage 3 K sweep + ablation eval | 2~4시간 | 1~2시간 | — |
| **합계 (실험 전체)** | **33~63시간** | **12~24시간** | **8~16시간** |

**권장 시나리오**:
- Stage 1 + 2: GPU 2 (학습) + GPU 3 (baseline inference) 병렬 → **2~4시간**
- Stage 3: GPU 2 + 3 DDP (3 seeds 순차) → **8~12시간**
- **전체 예상 완료**: **10~16시간 (KST 기준)**

---

## 8. 핵심 환경 변수 및 설정 요약

```bash
# 실험 시작 전 설정
export CUDA_VISIBLE_DEVICES=2,3      # GPU 0,1은 기존 작업 보존
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# BF16 최적화
export TORCH_ALLOW_TF32=1
```

```python
# 코드 내 설정
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True  # 고정 입력 크기 시 성능 향상
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-26 KST | 초안 작성 (B200 × 4장 환경, GPU 2+3 전용 배정) |

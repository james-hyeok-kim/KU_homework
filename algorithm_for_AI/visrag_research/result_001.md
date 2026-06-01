# VisRAG 개선 연구: Query-Conditioned Visual Token Pruning for Multi-Page Context Compression

**작성일**: 2026-05-26 KST  
**상태**: 실험 진행 중 (Stage 1 training, in-batch negatives, step=200+)  
**Novelty 판정**: ACCEPT (`state/validation_log.md`)

---

## Abstract

VisRAG의 VLM generator가 검색된 다수 페이지를 처리할 때 발생하는 context window 병목 문제를 해결하기 위해, retriever와 generator 사이에 경량 dot-product bilinear scoring 기반 Query-Conditioned Visual Token Pruning (QCVTP) 모듈을 제안한다. 쿼리 임베딩과 각 페이지의 visual patch token 간 relevance score를 계산하여 동적 token budget 하에 상위 patch만 선택함으로써, 동일 context 예산 내에서 더 많은 페이지를 수용하고 노이즈 patch를 제거한다. SlideVQA에서 +3~5%p, DocVQA에서 +1~2%p 성능 향상이 기대되며, 추가 파라미터는 ~2M에 불과하다.

---

## 1. Introduction

### 1.1 VisRAG 소개 및 한계점

VisRAG (Visual Retrieval-Augmented Generation)는 문서 페이지를 이미지로 처리하는 multimodal RAG 시스템으로, 텍스트 추출 없이 레이아웃·차트·수식 등 시각 정보를 그대로 VLM에 전달한다. 그러나 실제 운용에서 세 가지 중요한 한계가 드러난다.

- **한계 #2 (멀티페이지 이해)**: 동일 token budget 내에 여러 페이지를 동시에 처리하기 어렵다. K개 페이지를 모두 넣으면 K × N개의 visual token이 필요하며, 이는 context window를 빠르게 소진한다.
- **한계 #6 (Re-ranking 없음)**: 검색된 페이지를 그대로 사용하여 페이지 내 noisy patch(배경, 여백 등)가 generator에 그대로 전달된다.
- **한계 #7 (Context window 제한)**: VLM 기준 이미지당 수십~수백 patch token이 생성되며, K=5 페이지 처리 시 시각 context가 폭발적으로 증가한다. (본 구현: Qwen2-VL-7B, target_n_patches=64, K=5시 320 patch token)

### 1.2 제안 방법의 Motivation

쿼리와 관련 없는 patch token은 generator에 노이즈로 작용한다는 관찰에서 출발한다. 페이지의 모든 patch가 쿼리에 동등하게 유용하지 않으며, 예를 들어 "매출 성장률은?" 같은 질문에 대해 슬라이드의 헤더·여백·배경 이미지의 patch는 불필요하다. 이 노이즈 patch를 쿼리 기준으로 선별적으로 제거하면, 동일 context window에서 더 많은 페이지를 처리할 수 있다.

### 1.3 기여 (Contributions)

1. **경량 dot-product bilinear selector 삽입**: 기존 VisRAG 아키텍처를 최소한으로 변경하면서 retriever-generator 사이에 ~2M 파라미터의 token selector 모듈을 삽입한다.
2. **Dynamic token budget allocation**: 검색된 페이지 수 K에 따라 per-page pruning ratio r을 자동 조정하여, 항상 총 budget B 내에서 최대한 많은 페이지를 활용한다.
3. **두 단계 학습 전략**: Stage 1에서 retriever와 VLM을 동결하고 selector만 학습하며, Stage 2에서 선택적 end-to-end fine-tuning을 적용한다. Hinge loss를 통한 answer-page supervision으로 핵심 patch 보존을 유도한다.

---

## 2. Related Work

### 2.1 VisRAG 원논문

VisRAG (Yu et al., 2024)는 문서 페이지를 OCR 없이 이미지 그대로 처리하는 multimodal RAG 프레임워크다. MiniCPM-V를 retriever와 generator로 활용하며, 이미지 기반 dense retrieval 후 top-K 페이지를 VLM generator에 직접 입력한다. 본 연구는 이 파이프라인을 기반으로 하며 retriever는 동결한다.

### 2.2 Token Pruning/Compression 관련 연구

| 논문 | 방법 | 대상 |
|------|------|------|
| QG-VTC (arxiv 2504.00654, 2025) | question-guided visual token compression | 단일 이미지 VQA |
| FlashVLM (arxiv 2512.20561, 2024) | text-guided visual token selection | 단일 이미지 추론 |
| MI-Pruner (arxiv 2604.03072, 2026) | crossmodal mutual information 기반 pruning | 단일 이미지 |
| AdaptInfer (arxiv 2508.06084, 2025) | 동적 text guidance visual pruning | 단일 이미지 추론 최적화 |
| Index-Preserving Token Pruning (arxiv 2509.06415, 2025) | 문서 텍스트/배경 이진 분류 기반 pruning | 단일 문서 이미지 |

이 연구들은 단일 이미지를 대상으로 하며, RAG pipeline과의 통합이나 멀티페이지 처리를 다루지 않는다.

### 2.3 Visual RAG 관련 연구

| 논문 | 방법 | 대상 |
|------|------|------|
| AVIR (arxiv 2601.11976, 2026) | 멀티페이지 문서 QA, page-level 선택 | 페이지 단위 선택 (patch-level 아님) |
| RegionRAG (arxiv 2510.27261, 2025) | RAG + 시각 문서 region-level 검색 | region 검색 단위 변경 |
| VimRAG (arxiv 2602.12735, 2026) | multimodal memory graph 기반 RAG | memory graph 구조 기반 |

### 2.4 유사 논문과의 차별점 (Validation Log 기반)

Novelty 검증 결과(ACCEPT), 제안 방법은 아래 세 요소의 특정 결합으로 기존 논문과 구별된다:

1. **VisRAG retrieval pipeline에 통합된** query-conditioned patch-level token pruning (QG-VTC, FlashVLM은 단일 이미지만)
2. **멀티페이지에 걸친** dynamic token budget (AVIR는 page-level만, patch-level budget 조정 없음)
3. **retriever-generator 사이에 삽입되는** 경량 dot-product bilinear selector, in-batch negatives 학습 (RegionRAG, VimRAG는 방법론이 상이)

이 세 가지를 동시에 조합하여 VisRAG에 적용한 논문은 검색 결과에서 발견되지 않았다(검색일: 2026-05-26 KST).

---

## 3. Method: Query-Conditioned Visual Token Pruning

### 3.1 Overview

전체 파이프라인은 기존 VisRAG의 retrieval stage를 그대로 유지하면서, vision encoding과 generation 사이에 QCVTP 모듈을 삽입한다.

```
[Query] ──► [VisRAG Retriever (frozen)] ──► Top-K pages
                                               │
                                    [VLM Vision Encoder]
                                               │
                                    patch tokens (B, K, N, D)
                                               │
                                  ┌────────────▼────────────┐
                                  │  QCVTP Token Selector   │
                                  │  (dot-product bilinear, ~2M) │
                                  └────────────┬────────────┘
                                               │
                                    pruned tokens (B, K, r×N, D)
                                               │
                                    [VLM Generator (frozen/fine-tuned)]
                                               │
                                           [Answer]
```

### 3.2 Query-Patch Relevance Scoring

쿼리 텍스트 임베딩 e_q ∈ R^{query_dim}와 각 페이지 k의 patch token h_i ∈ R^{patch_dim}에 대해 **dot-product bilinear scoring**으로 relevance score를 계산한다:

```
q_proj   = normalize(W_q · e_q)            ∈ R^{proj_dim}   # query projection
v_proj_i = normalize(W_v · h_i)            ∈ R^{proj_dim}   # patch projection
s_i      = q_proj^T · v_proj_i             ∈ R              # scalar relevance score
```

여기서 W_q ∈ R^{proj_dim × query_dim}, W_v ∈ R^{proj_dim × patch_dim}는 학습 가능한 경량 projection matrix (proj_dim=256, 총 ~2M 파라미터)이다.

**Cross-attention과의 차이**: cross-attention은 query가 모든 key-value쌍을 동시에 attend해 patch 간 상호작용을 포착하는 아키텍처다. 본 구현은 그보다 단순하다 — 각 patch를 query와 **독립적으로** 비교하는 bilinear dot-product 방식이며, 연산량은 O(K×N)으로 낮고 학습이 안정적이다. 단, L2 정규화 후 내적 연산 자체는 수식적으로 cosine similarity와 동치이므로 "학습된 cosine selector"로 볼 수 있다.

### 3.3 Dynamic Token Budget Allocation

총 visual token budget B와 검색된 페이지 수 K, 페이지당 patch 수 N에 따라 per-page pruning ratio r을 동적으로 결정한다:

```
r = min(1.0, B / (K × N))
r = max(0.2, r)          (하한 0.2: 최소 20% patch 유지)
```

K=3, N=1024, B=2048일 때: r = 2048/3072 ≈ 0.67 (페이지당 67% 유지)  
K=5, N=1024, B=2048일 때: r = 2048/5120 = 0.40 (페이지당 40% 유지)  
K=7, N=1024, B=2048일 때: r = 2048/7168 ≈ 0.29 (하한 적용 → 0.29)

상위 r 비율의 patch를 선택하여 pruned context를 구성한다:

```
k_keep = max(1, int(r × N))
H_k^pruned = top-k_keep patches by s_i from H_k
```

K=1일 때는 in-batch negatives를 적용하여 학습한다 (batch 내 다른 샘플의 이미지를 negative로 활용). 추론 시에는 keep_ratio=1.0이 적용되어 전체 patch를 유지한다.

Multi-page context는 [SEP] 토큰으로 구분하여 generator에 입력한다:

```
context = [H_1^pruned | SEP | H_2^pruned | SEP | ... | H_K^pruned]
```

### 3.4 Training Strategy

**Stage 1 — Selector-only Training**

retriever와 VLM generator를 동결하고, token selector의 W_q, W_v만 학습한다.

손실 함수:
```
L_total = L_hinge
```

L_hinge: 자신의 이미지 patch score가 타 샘플의 이미지 patch score보다 높도록 유도하는 hinge loss

**In-batch Negatives 전략 (핵심):**

in_domain 데이터셋은 샘플당 이미지 1장(K=1)으로 구성된다. K=1이면 negative page가 없어 hinge_loss = 0 → gradient 없음. 이를 해결하기 위해 in-batch negatives를 사용한다:

```
# batch 내 B개 샘플의 이미지를 서로 negative로 재사용
# patch_tokens: (B, B, N, D) — 각 query가 모든 B개 이미지를 봄
# ans_mask[b, b] = True  — sample b의 own 이미지만 positive

scores = selector(q_emb, patch_tokens, keep_ratio)   # (B, B, N)
# patch-level score를 페이지 단위로 집계 (mean)
page_scores = scores.mean(dim=-1)                     # (B, B)

# 각 query b에 대해:
# positive  = page_scores[b, b]        (자기 이미지)
# negatives = page_scores[b, j≠b]      (타 샘플 이미지)
L_hinge = mean over (b, j≠b): max(0, margin - (page_scores[b,b] - page_scores[b,j]))
```

이로써 K=1 데이터셋에서도 effective K=batch_size(=4) supervision이 가능하다.

하이퍼파라미터: margin=0.5, lr=1e-4 (cosine schedule, warmup 200 steps), max_steps=2,000, batch_size=4, grad_accum=4

**Stage 2 — End-to-end Fine-tuning (선택적)**

Stage 1 완료 후, selector + VLM generator를 함께 fine-tuning한다. retriever는 계속 동결.  
lr=5e-6, grad_accum=16, gradient checkpointing 적용.

---

## 4. Experimental Setup

### 4.1 데이터셋

| 데이터셋 | 역할 | 학습 | 검증 | 테스트 |
|---------|------|------|------|--------|
| SlideVQA | 헤드라인 (멀티페이지) | 9,394 QA | 2,135 QA | 2,230 QA |
| DocVQA | 단일페이지 회귀 확인 | train 80% | 5,188 QA | 5,187 QA |
| ChartQA | 보조 (차트 patch 보존) | 18,317 QA | 1,250 QA | 1,250 QA |
| InfoVQA | 선택적 (Stage 3) | — | — | — |

이미지 전처리: Qwen2-VL-7B AutoProcessor 기본 설정 사용. target_n_patches=64로 고정 (padding/truncation 적용).

### 4.2 베이스라인 비교군

| 레이블 | 방법 | 목적 |
|--------|------|------|
| VisRAG (vanilla) | token pruning 없음 | 기준선 |
| VisRAG + Random Drop | 같은 r로 랜덤 patch drop | query-conditioning 효과 분리 (Claim C2) |
| VisRAG + Cosine Selector | 학습 없이 raw cosine similarity | learned projection 기여도 측정 (Claim C3) |
| VisRAG + QCVTP (Ours) | 학습된 dot-product bilinear selector + dynamic budget + in-batch neg | 제안 방법 |

### 4.3 평가 지표

| 데이터셋 | 주요 Metric | 목표 |
|---------|------------|------|
| SlideVQA | ANLS (threshold=0.5) | vanilla 대비 +3%p 이상 |
| DocVQA | ANLS | vanilla 대비 -0.5%p 이내 (회귀 없음) |
| ChartQA | Relaxed Accuracy (±5% 수치 허용) | vanilla 대비 +1%p 이상 |
| 전체 | Context window 절약율 | K=5 기준 ≥50% token 감소 |
| 전체 | Pruning latency overhead | <10ms per forward pass |

통계 검증: Paired bootstrap resampling (n=1,000), 95% CI, 3 random seeds.

### 4.4 구현 세부사항

| 항목 | 설정 |
|------|------|
| Base VLM | Qwen2-VL-7B-Instruct (bfloat16, 8.3B params) + VisRAG-Ret (3.4B params) |
| GPU | NVIDIA B200 × 2 (GPU 2, 3; 각 183 GB VRAM) — GPU 0,1은 기존 작업 점유 |
| Precision | BF16 (B200 native) |
| Batch size | Stage 1: 4 × grad_accum 4 = effective 16 |
| Token budget B | 2,048 |
| N (patches/page) | 64 (Qwen2-VL-7B 실제 측정값; target_n_patches=64) |
| Optimizer | AdamW, weight_decay=1e-4 |
| num_workers | 0 (HF memory-mapped dataset, worker없이 메모리 효율적 로딩) |
| Checkpoint 주기 | 매 100 steps; step_XXXXXX.pt 형식으로 저장 |

Stage 3 예상 실행 시간: B200 DDP 2장 기준 8~16시간 (전체 포함).

---

## 4.5 실험 현황 (2026-05-26 KST)

| 항목 | 상태 | 비고 |
|------|------|------|
| Sanity check (50 steps) | 완료 | step=50, hinge=0 (K=1 문제 확인) |
| Full train v1 (K=1, hinge=0) | 중단 | step=250, hinge=-0.0000 — 학습 없음 |
| In-batch negatives 구현 | 완료 | patch_tokens (B,B,N,D), ans_mask diagonal |
| Full train v2 (in-batch) | **진행 중** | step=200+ resume, hinge=0.5010 — 학습 정상 |
| QCVTP eval (SlideVQA/DocVQA) | 대기 | 2,000 steps 완료 후 실행 |

**학습 진행 모니터링:**
```
[2026-05-26 05:28:08 KST] step=  200 | hinge=0.5010 | r=1.000 | lr=2.50e-05 | inbatch | GPU 22.0/178GB
```

hinge_loss = 0.5 근방 → margin=0.5와 일치, selector가 아직 random weight에서 학습 시작.
학습이 진행되면서 hinge_loss가 0을 향해 감소하는지 모니터링 예정.

---

## 5. Expected Results

### 5.1 예상 성능 개선표

다음 수치는 아이디어 설계 단계의 예측값이며, 실제 실험 후 갱신 예정이다.

| Benchmark | VisRAG vanilla | 목표 최소 | 목표 최대 | 근거 |
|-----------|---------------|---------|---------|------|
| SlideVQA ANLS | ~65% | 68% | 70% | 멀티페이지 cross-page 정보 통합 효과 (K=3→5) |
| DocVQA ANLS | ~81% | 81% (회귀 없음) | 83% | 관련 patch 집중으로 hallucination 감소 |
| ChartQA Relaxed Acc | ~74% | 75% | 77% | 차트 핵심 구조(x/y축, 데이터 포인트) patch 보존 |
| InfoVQA ANLS | ~48% | 50% | 52% | 복잡한 레이아웃에서 query-relevant 구조 추출 |

Context 효율성: K=3 처리 token budget으로 K=5~7 페이지까지 확장 가능. Pruning latency 오버헤드 <5ms (retrieval 대비 무시할 수준).

### 5.2 Ablation Study 계획

| Ablation | 변경 사항 | 측정 목적 |
|---------|---------|----------|
| A1: No hinge loss | L_hinge 제거, L_VQA만 | hinge loss 기여도 |
| A2: Fixed r=0.5 | dynamic r 대신 고정 ratio | dynamic budget 효과 |
| A3: Fixed r=0.25 | 공격적 고정 pruning | 성능-압축 tradeoff 곡선 |
| A4: Stage 1 only | end-to-end fine-tuning 없음 | Stage 2 기여도 |
| A5: K sweep | K ∈ {3, 5, 7} (same budget B) | 페이지 수 확장 효과 직접 측정 |

### 5.3 판정 기준

| 판정 | 조건 |
|------|------|
| SUCCESS (강) | SlideVQA ANLS ≥ +3%p, DocVQA 회귀 없음, 통계적으로 유의 |
| SUCCESS (약) | SlideVQA ANLS ≥ +1%p 또는 DocVQA +1%p, 통계적으로 유의 |
| PARTIAL | 일부 벤치마크 개선, 일부 회귀 → 원인 분석 후 revision |
| FAIL | 모든 벤치마크에서 vanilla 대비 동등 이하 → 아이디어 재검토 |

---

## 6. Implementation Details

### 6.1 코드 구조 (src/ 파일별 역할)

| 파일 | 역할 |
|------|------|
| `src/token_selector.py` | `QueryConditionedTokenSelector` 클래스 (W_q, W_v projection, top-r selection) + `hinge_loss` 함수 |
| `src/visrag_pipeline.py` | `VisRAGWithPruning` 클래스 — retrieve → encode → prune → generate 전체 파이프라인 |
| `src/train.py` | `train_stage1` / `train_stage2` 학습 루프, CLI entry point |
| `src/evaluate.py` | ANLS, Exact Match, Relaxed Accuracy 계산; bootstrap CI; JSON 결과 저장 |
| `src/run_experiment.sh` | Stage 1~3 순차 실행 쉘 스크립트; GPU 모니터링 포함 |

### 6.2 실행 방법

```bash
# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 전체 실험 실행 (Stage 1 sanity → Stage 2 subset → Stage 3 full)
cd /path/to/experiments/visrag/src
bash run_experiment.sh \
    --seed 42 \
    --data ./data \
    --ckpt ./checkpoints \
    --budget 2048

# Stage 2 DDP 실행 (GPU 2+3)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py \
    --stage 1 \
    --max_steps 10000 \
    --batch_size 8 \
    --grad_accum 2 \
    --dataset slidevqa docvqa chartqa

# 평가만 실행
python evaluate.py \
    --checkpoint checkpoints/exp001_seed42/stage1/final.pt \
    --datasets slidevqa docvqa chartqa \
    --split test \
    --token_budget 2048 \
    --output_json results/exp001_final_seed42.json
```

### 6.3 TODO — 실제 실행 전 완료 필요 사항

다음 항목들은 현재 스켈레톤 코드에서 `TODO` 또는 `NotImplementedError`로 마킹된 사항이다:

1. **`visrag_pipeline.py:retrieve()`**: 실제 VisRAG retriever 호출 연결 (query/page embedding, top-K 선택)
2. **`visrag_pipeline.py:encode_pages_to_patch_tokens()`**: MiniCPM-V의 vision encoder (`self.vlm.vpm`) forward hook으로 patch-level hidden states 추출. 대안: InternVL2-8B vision_model
3. **`visrag_pipeline.py:assemble_and_generate()`**: pruned patch token을 SEP로 연결 후 VLM 생성 API에 주입 (`vlm.chat()` 또는 `get_vllm_embedding()`)
4. **`visrag_pipeline.py:compute_loss()`**: VLM logits 추출 후 cross-entropy 계산
5. **`train.py:main()`**: retriever, VLM 실제 인스턴스화 (`load_visrag_retriever`, `load_minicpm_v`)
6. **`train.py:VisRAGDataset`**: SlideVQA/DocVQA/ChartQA HuggingFace Hub 로더 구현
7. **`evaluate.py:evaluate_dataset()`**: `pipeline.forward()` 호출 연결, token count 수집
8. **Answer Page Supervision**: OCR (PaddleOCR 또는 Tesseract) 기반 gold answer 위치 레이블링
9. **MiniCPM-V patch token 수 확인**: 실제 N (동적 슬라이싱 포함 최대값) 검증
10. **Token budget B=2048 검증**: generator context 제약과 일치 여부 확인

---

## 7. Conclusion & Future Work

본 연구는 VisRAG의 context window 병목 문제를 해결하기 위한 QCVTP (Query-Conditioned Visual Token Pruning) 방법을 제안한다. 핵심 아이디어는 retriever-generator 사이에 경량 dot-product bilinear selector를 삽입하여, 쿼리 관련성 기준으로 visual patch token을 동적으로 pruning하는 것이다. 이를 통해 동일 context 예산 내에서 더 많은 페이지를 처리하고 (K=3→5~7), 노이즈 patch를 제거하여 답변 품질을 향상시키는 것을 목표로 한다.

Novelty 검증 결과 QG-VTC, AVIR, RegionRAG 등 유사 연구와 차별화된다. 특히 VisRAG retrieval pipeline + 멀티페이지 dynamic budget + query-conditioned selector(in-batch negatives 학습)의 세 요소 결합은 기존 문헌에서 발견되지 않았다.

**한계 및 향후 과제**:

- 현재 구현은 스켈레톤 코드 수준이며, MiniCPM-V API와의 실제 통합 작업이 필요하다.
- Answer page supervision의 OCR 정확도에 의존하므로, grounding annotation이 없는 데이터셋에서 L_hinge 적용이 제한될 수 있다.
- B200 환경(GPU 2, 3)에서 실제 실험 후 본 문서의 Expected Results 섹션을 실측값으로 갱신 예정이다.

**향후 확장 방향**:
- Retriever score와 patch-level score를 결합한 joint re-ranking
- Selector를 multi-head attention으로 확장하여 서로 다른 측면의 쿼리 정보 포착
- 학습 없는 (training-free) 빠른 초기 검증을 위한 attention map 기반 baseline

---

## References

1. Yu, S. et al. "VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents." arXiv preprint, 2024.
2. QG-VTC. "Question-Guided Visual Token Compression for Efficient VQA." arXiv:2504.00654, 2025.
3. FlashVLM. "Text-Guided Visual Token Selection for Efficient Vision-Language Models." arXiv:2512.20561, 2024.
4. AVIR. "Adaptive Visual In-document Retrieval for Multi-page Document QA." arXiv:2601.11976, 2026.
5. RegionRAG. "Region-Level Visual Retrieval-Augmented Generation." arXiv:2510.27261, 2025.
6. VimRAG. "Multimodal Memory-based Visual RAG for Large-scale Documents." arXiv:2602.12735, 2026.
7. MI-Pruner. "Crossmodal Mutual Information-based Visual Token Pruning." arXiv:2604.03072, 2026.
8. AdaptInfer. "Adaptive Dynamic Text-Guided Visual Token Pruning for Efficient Inference." arXiv:2508.06084, 2025.
9. Index-Preserving Token Pruning. arXiv:2509.06415, 2025.
10. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. OpenBMB, 2024.

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-26 KST | 초안 작성 (result_001.md v1.0) — 설계 단계 최종 문서, 실측값 갱신 예정 |
| 2026-05-26 KST | v1.1 — 아키텍처 교정: cross-attention → dot-product bilinear, in-batch negatives 학습 전략 반영, 실험 현황 추가 |

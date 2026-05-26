# Experiment 001: Query-Conditioned Visual Token Pruning for Multi-Page Context Compression

**작성일**: 2026-05-26 KST  
**상태**: 설계 완료 (실행 대기)  
**아이디어 출처**: `state/idea_candidate.md`  
**Novelty 검증**: ACCEPT (`state/validation_log.md`)

---

## 1. 실험 개요

### 가설 (Hypothesis)

> VisRAG의 retriever와 generator 사이에 경량 cross-attention 기반 token selector 모듈을 삽입하여, 검색된 멀티페이지 이미지의 visual patch token을 query-conditioned하게 pruning하면 — 동일한 context window 예산 내에서 더 많은 페이지를 수용하고 노이즈 patch를 제거하여 — 답변 정확도(ANLS/Accuracy)가 향상된다.

### 검증하려는 핵심 Claims

| Claim | 측정 방법 |
|-------|----------|
| C1: 동일 token budget에서 더 많은 페이지 처리 시 멀티페이지 QA 성능 향상 | SlideVQA (K=3 vs K=5 with same budget) |
| C2: query-conditioned pruning이 query-agnostic pruning보다 우수 | DocVQA ablation: random drop vs learned selector |
| C3: 학습된 cross-attention projection이 raw cosine similarity보다 유효 | DocVQA ablation: cosine baseline vs learned W_q/W_v |
| C4: pruning이 성능 저하 없이 context window를 절약 | 모든 벤치마크에서 baseline 대비 ≤ -0.5%p 이내 |

---

## 2. 데이터셋

### 선택 및 이유

| 데이터셋 | 역할 | 이유 |
|---------|------|------|
| **SlideVQA** | 1차 헤드라인 벤치마크 | 유일한 실제 멀티페이지(슬라이드 덱) QA — C1 핵심 claim을 직접 검증 가능 |
| **DocVQA** | 2차 단일페이지 벤치마크 | 노이즈 patch 제거 효과(C2, C3) 검증 및 회귀 확인용 |
| **ChartQA** | 3차 보조 벤치마크 | 차트 핵심 구조 patch 보존 효과 검증 |
| InfoVQA | 선택적 (Stage 3) | 복잡한 레이아웃에서 추가 검증 — 리소스 여유 시 실행 |
| ArxivQA | 선택적 (Stage 3) | 2차 멀티페이지 벤치마크 — SlideVQA 결과 재현 확인 |

ChartQA와 InfoVQA는 단일 이미지 기반이므로 멀티페이지 C1 claim은 테스트 불가. 단, 단일 페이지 내 patch-level 압축의 단독 효과를 분리하는 데 사용.

### Split 전략

| 데이터셋 | Train | Val | Test |
|---------|-------|-----|------|
| SlideVQA | train split (9,394 QA) | val split (2,135 QA) | test split (2,230 QA) |
| DocVQA | train (80%) | val (공식 val, 5,188 QA) | test (공식 test, 5,187 QA) |
| ChartQA | train (18,317 QA) | val (set 포함, 1,250 QA) | test (1,250 QA) |

**Stage 1 (sanity check)**: 각 데이터셋에서 랜덤 100 샘플 (forward pass 검증용)  
**Stage 2 (소규모)**: 각 val split의 20% (≈500~1,000 QA)  
**Stage 3 (풀 benchmark)**: 공식 test split 전체

### 데이터 전처리

1. **이미지 해상도**: VisRAG 원본 설정 유지 (MiniCPM-V-2.6 기준, 동적 슬라이싱 448×448)
2. **Answer Page Supervision (L_hinge 학습용)**:
   - **1차 방법 (Distant Supervision)**: 각 페이지의 OCR 텍스트가 gold answer 문자열을 포함하면 positive page로 레이블링
   - **SlideVQA 예외**: 공식 슬라이드 grounding annotation이 있는 경우 해당 annotation 사용 (섹션 7 위험 요소 참조)
   - **OCR 도구**: Tesseract 또는 PaddleOCR (전처리 중 1회 실행하여 캐시)
   - **실패 시 fallback**: gold answer가 어느 페이지에도 없으면 해당 샘플에서 L_hinge 손실 제외 (VQA loss만 사용)
3. **Token Budget 계산**:
   - MiniCPM-V-2.6의 visual token 수: 이미지당 최대 N = 1024 patch tokens (동적 슬라이싱 포함)
   - 총 budget B = 2048 (generator context 중 visual token 할당 예산)
   - K 페이지일 때 per-page ratio: r = min(1.0, B / (K × N))
   - **구체적 예시**: K=5, N=1024, B=2048 → r = 2048/5120 = 0.4 (각 페이지당 40% patch 유지 = 410 tokens)
   - **비교**: K=3이면 r = 2048/3072 = 0.67 (각 페이지당 67% patch 유지)

---

## 3. 베이스라인 비교군

### Baseline 1: 기존 VisRAG (변경 없음)
- Top-K 페이지를 token pruning 없이 그대로 VLM generator에 입력
- K 값이 커지면 context 초과로 페이지 수 줄이거나 truncation 발생
- **레이블**: `VisRAG (vanilla)`

### Baseline 2: Random Patch Drop (query-agnostic)
- 같은 r ratio로 patch를 랜덤하게 drop (query 정보 미사용)
- learned selector의 query-conditioning 효과를 분리 (C2 검증)
- **레이블**: `VisRAG + Random Drop`

### Baseline 3: Cosine Similarity Selector (학습 없는 query-conditioned)
- 학습 없이 raw cosine similarity로 score: `s_i = cosine(e_q_mean, h_i)`
- learned projection (W_q, W_v)의 기여를 분리 (C3 검증)
- **레이블**: `VisRAG + Cosine Selector`

### 제안 방법 (Ours)
- Cross-attention selector (학습된 W_q, W_v) + dynamic token budget
- **레이블**: `VisRAG + QCVTP (Ours)`

### Ablation 설계

| Ablation | 변경 사항 | 목적 |
|---------|---------|------|
| A1: No hinge loss | L_hinge 제거, VQA loss만 사용 | hinge loss 기여도 측정 |
| A2: Fixed r=0.5 | dynamic r 대신 고정 ratio | dynamic budget의 효과 측정 |
| A3: Fixed r=0.25 | 더 공격적인 고정 pruning | 성능-압축 tradeoff 곡선 |
| A4: Stage 1 only | selector만 학습, end-to-end 없음 | Stage 2 fine-tuning 기여도 |
| A5: K sweep | K ∈ {3, 5, 7} (same budget B) | 페이지 수 확장 효과 직접 측정 |

---

## 4. 평가 지표 (Metrics)

### 주요 Metrics

| 데이터셋 | 주요 Metric | 임계값 (목표) |
|---------|------------|-------------|
| SlideVQA | ANLS (Average Normalized Levenshtein Similarity) | VisRAG vanilla 대비 +3%p 이상 |
| DocVQA | ANLS | vanilla 대비 -0.5%p 이내 (회귀 없음) + 가능하면 +1%p |
| ChartQA | Relaxed Accuracy | vanilla 대비 +1%p 이상 |
| 전체 | Context Window 절약율 | K=5 기준 ≥50% token 감소 (r≤0.5) |
| 전체 | Pruning latency overhead | <10ms per forward pass (무시할 수준) |

### 절대 수치 목표 (idea_candidate.md 기반)

| Benchmark | VisRAG 기존 | 목표 최소 | 목표 최대 |
|-----------|------------|---------|---------|
| SlideVQA | ~65% ANLS | 68% | 70% |
| DocVQA | ~81% ANLS | 81% (회귀 없음) | 83% |
| ChartQA | ~74% Acc | 75% | 77% |
| InfoVQA | ~48% ANLS | 50% | 52% |

### 통계적 유의성 검증

- **방법**: Paired bootstrap resampling (n=1,000 iterations) on same question set
- **Minimum**: 3 random seeds (학습 시), seed별 결과 → mean ± std 보고
- **CI**: 95% bootstrap CI 보고
- **유의성 기준**: CI lower bound가 0 이상일 때 개선으로 판정

### 실패 판정 기준

- SlideVQA ANLS < vanilla VisRAG: 실험 실패 (C1 기각)
- DocVQA ANLS 대비 -2%p 이상 하락: 단일 페이지 회귀 (revision 필요)
- Pruning latency > 100ms: 실용성 없음 → 모듈 최적화 필요

---

## 5. 실험 단계별 계획

### Stage 1: Sanity Check (기능 검증)

**목표**: 전체 파이프라인이 오류 없이 forward pass를 완료하는지 확인  
**데이터**: 각 데이터셋 100 샘플  
**검증 항목**:
- Token selector 모듈이 올바른 shape의 pruned token 반환
- K×r×N token이 generator에 전달되는지 확인
- VQA loss가 NaN/Inf 없이 계산
- Pruning ratio r이 K에 따라 동적으로 변하는지 확인

**성공 기준**: 100 샘플 forward pass 완료, loss 수렴 시작 확인

```
예상 시간: 30분~1시간 (GPU 1장)
```

### Stage 2: 소규모 실험 (Subset Evaluation)

**목표**: 학습 가능성과 개선 신호 조기 확인  
**데이터**: 각 val split의 20%  
**실행 순서**:

1. **Step 2-1**: Baseline 측정 (vanilla VisRAG) — 학습 없이 inference만
2. **Step 2-2**: Cosine Selector 평가 (학습 없음) — Baseline 3 측정
3. **Step 2-3**: Token Selector 학습 (Stage 1 학습: selector만, retriever 동결)
   - Optimizer: AdamW, lr=1e-4, warmup 500 steps
   - Loss: VQA cross-entropy + λ·L_hinge (λ=0.1, hinge margin=0.5)
   - Batch size: 4 (gradient accumulation ×4 = effective 16)
   - Max steps: 2,000
4. **Step 2-4**: 학습된 selector 평가 (Ours Stage 1)
5. **Step 2-5**: Ablation A1~A3 평가 (각 variant 학습 없이 또는 별도 학습)

**성공 기준**: SlideVQA에서 vanilla 대비 +1%p 이상 개선 신호 확인

```
예상 시간: 4~8시간 (A100 80GB 1장 기준)
```

### Stage 3: 풀 Benchmark 실행

**목표**: 공식 test split에서 최종 성능 측정 및 통계 검증  
**데이터**: 공식 test split 전체  
**실행 순서**:

1. **Step 3-1**: Full 학습 (train split 전체, 최대 10,000 steps)
   - Stage 1 학습 완료 후 Stage 2 (end-to-end fine-tuning) 선택적 수행
2. **Step 3-2**: 3 random seeds로 반복 학습 및 평가
3. **Step 3-3**: K sweep 실험 (A5 ablation, SlideVQA)
4. **Step 3-4**: 전체 ablation 표 완성

```
예상 시간: 24~48시간 (A100 80GB 1장 기준), seed 3개 병렬 시 8~16시간
```

---

## 6. 리소스 요구사항

### GPU Memory

| 컴포넌트 | 메모리 |
|---------|-------|
| MiniCPM-V-2.6 (base VLM) | ~16GB (bfloat16) |
| Visual token selector (W_q, W_v ~2M params) | ~8MB (무시 가능) |
| Batch size 4 × K=5 pages × 1024 tokens | ~12GB 추가 (activation) |
| **총 추정** | **30~35GB** |

권장: A100 40GB 이상. 40GB 부족 시 gradient checkpointing + batch_size=2로 대응.

### 예상 실행 시간 (A100 80GB 1장 기준)

| 단계 | 시간 |
|------|------|
| Stage 1 (sanity, 100 samples) | 0.5~1시간 |
| Stage 2 (subset, 2K steps) | 4~8시간 |
| Stage 3 (full, 10K steps × 3 seeds) | 24~48시간 |
| 전체 ablation evaluation | 4~6시간 |
| **총계** | **33~63시간** |

### 필요 라이브러리

```
# Core
torch >= 2.0
transformers >= 4.40
Pillow, numpy

# MiniCPM-V
git+https://github.com/OpenBMB/MiniCPM-V

# OCR (answer page supervision용)
paddleocr  # 또는 pytesseract

# Evaluation
anls  # pip install anls-metric
evaluate  # HuggingFace evaluate

# Benchmarks
datasets  # SlideVQA, DocVQA HuggingFace Hub
```

---

## 7. 위험 요소 및 대응

### R1: Answer Page Supervision 레이블 불가 (HIGH)

**위험**: OCR 기반 distant supervision이 정확하지 않거나, gold answer가 어느 페이지에도 OCR되지 않는 경우 L_hinge 적용 불가.

**대응**:
- **기본 전략**: L_hinge 없이 VQA loss만으로 Stage 1 학습 (Ablation A1이 실질적 기본값이 됨)
- **L_hinge는 데이터가 가능한 샘플에만 선택적 적용** (answer page 확인된 샘플만 hinge loss 추가)
- SlideVQA 공식 annotation 확인 필수 (train split에 slide-level grounding 있는지)

### R2: K>3 학습 시 OOM (MEDIUM)

**위험**: K=5~7 페이지를 batch에 넣으면 GPU 메모리 초과.

**대응**:
- gradient checkpointing 활성화
- batch_size=1, gradient accumulation ×16
- Stage 3 K sweep은 inference-only (학습은 K=3으로, 평가는 K=5,7로 분리)

### R3: Pruning이 성능 하락 유발 (MEDIUM)

**위험**: patch를 제거하면 관련 시각 정보가 손실되어 오히려 성능 저하.

**대응**:
- r의 최솟값 0.2 (80% 이상 제거 불허) 하한 설정
- Hinge loss로 answer-page patch 보호
- 실패 시: pruning ratio를 완화하거나 stage 2 fine-tuning 생략

### R4: MiniCPM-V의 vision encoder 출력 접근성 (LOW~MEDIUM)

**위험**: MiniCPM-V-2.6이 patch token sequence를 직접 노출하지 않는 API 구조일 수 있음.

**대응**:
- MiniCPM-V의 `encode_img` 또는 `get_vllm_embedding` 함수에서 visual hidden states 추출
- 필요 시 vision encoder forward hook 등록으로 hidden state 가로채기
- 불가 시: InternVL2-8B (오픈 아키텍처) 대체 검토

### R5: Edge Case — 페이지 수 K=1인 쿼리

**위험**: K=1 이면 r=1.0 (pruning 불필요), dynamic budget 기능이 무의미함.

**대응**: K=1 샘플에서는 selector bypass → 원본 tokens 그대로 사용. SlideVQA는 다수 멀티페이지이므로 비중 낮음.

---

## 8. 검증 체크리스트

### 구현 전 확인 사항

- [ ] MiniCPM-V-2.6에서 patch-level visual token sequence 추출 가능한지 코드 레벨 확인
- [ ] SlideVQA 공식 데이터셋에 slide-level answer grounding annotation 존재 여부 확인
- [ ] DocVQA, SlideVQA HuggingFace Hub에서 직접 로드 가능한지 확인
- [ ] OCR 도구 설치 및 샘플 페이지에서 answer string 포함 여부 정확도 사전 테스트
- [ ] Token budget B=2048이 MiniCPM-V-2.6 generator의 실제 context 제약과 일치하는지 확인

### 실험 중 모니터링 항목

- [ ] 학습 loss (VQA loss + hinge loss) 10 steps마다 로깅
- [ ] Pruning ratio r이 K에 따라 동적으로 변하는지 각 배치에서 확인
- [ ] Visual token 수 (pruning 전/후) 배치별 기록
- [ ] GPU 메모리 사용량 (OOM 위험 모니터링)
- [ ] 학습 500 steps마다 val subset에서 ANLS 체크 (early stopping 판단)

### 결과 판단 기준

| 판정 | 조건 |
|------|------|
| SUCCESS (강) | SlideVQA ANLS ≥ +3%p, DocVQA 회귀 없음, 통계적으로 유의 |
| SUCCESS (약) | SlideVQA ANLS ≥ +1%p OR DocVQA +1%p, 통계적으로 유의 |
| PARTIAL | 일부 벤치마크 개선, 일부 회귀 → 원인 분석 후 revision |
| FAIL | 모든 벤치마크에서 vanilla 대비 동등 이하 → 아이디어 재검토 |

---

## 9. 스켈레톤 코드 (Pseudocode)

### 9.1 Token Selector 모듈

```python
import torch
import torch.nn as nn

class QueryConditionedTokenSelector(nn.Module):
    """
    Retriever-Generator 사이에 삽입되는 경량 cross-attention token selector.
    visual patch tokens를 query embedding 기준으로 pruning.
    """
    def __init__(self, query_dim: int, patch_dim: int, proj_dim: int = 256):
        super().__init__()
        self.W_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(patch_dim, proj_dim, bias=False)

    def forward(
        self,
        query_embedding: torch.Tensor,   # (B, query_dim)
        patch_tokens: torch.Tensor,       # (B, K, N, patch_dim)
        keep_ratio: float,                # r = budget / (K * N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pruned_tokens: (B, K, int(r*N), patch_dim)  — 선택된 patch tokens
            scores: (B, K, N)                            — relevance scores (L_hinge용)
        """
        B, K, N, D = patch_tokens.shape
        q_proj = self.W_q(query_embedding)            # (B, proj_dim)
        v_proj = self.W_v(patch_tokens.view(B*K, N, D))  # (B*K, N, proj_dim)
        
        # Query-Patch relevance score
        scores = torch.einsum('bd,bnd->bn', 
                              q_proj.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1), 
                              v_proj)                  # (B*K, N)
        scores = scores.view(B, K, N)
        
        # Top-r selection
        k_keep = max(1, int(keep_ratio * N))
        topk_indices = scores.topk(k_keep, dim=-1).indices  # (B, K, k_keep)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        pruned = patch_tokens.gather(2, topk_indices_expanded)  # (B, K, k_keep, D)
        
        return pruned, scores


def compute_keep_ratio(budget_total: int, K: int, N: int) -> float:
    """dynamic token budget: r = min(1.0, B / (K * N))"""
    return min(1.0, budget_total / (K * N))
```

### 9.2 Hinge Loss

```python
def hinge_loss(
    scores: torch.Tensor,       # (B, K, N) relevance scores
    answer_page_mask: torch.Tensor,  # (B, K) bool: True if page contains answer
    margin: float = 0.5,
) -> torch.Tensor:
    """
    answer page의 평균 score가 noise page보다 margin만큼 높도록 유도.
    answer_page_mask에 유효 페이지가 없으면 0 반환 (skip).
    """
    B, K, N = scores.shape
    page_mean_scores = scores.mean(dim=-1)  # (B, K)
    
    loss = torch.tensor(0.0, device=scores.device)
    count = 0
    for b in range(B):
        pos_mask = answer_page_mask[b]   # (K,) bool
        neg_mask = ~pos_mask
        if not pos_mask.any() or not neg_mask.any():
            continue  # supervision 불가 샘플 skip
        pos_score = page_mean_scores[b][pos_mask].mean()
        neg_score = page_mean_scores[b][neg_mask].mean()
        loss += torch.clamp(margin - pos_score + neg_score, min=0.0)
        count += 1
    
    return loss / max(count, 1)
```

### 9.3 학습 루프 (Stage 1 skeleton)

```python
def train_stage1(
    model,           # VisRAG generator (frozen)
    retriever,       # VisRAG retriever (frozen)
    selector,        # QueryConditionedTokenSelector (학습 대상)
    dataloader,
    optimizer,
    lambda_hinge: float = 0.1,
    budget_total: int = 2048,
    max_steps: int = 2000,
):
    retriever.eval()
    model.eval()
    selector.train()

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        queries, images_per_page, labels, answer_page_labels = batch
        # images_per_page: (B, K, H, W, C)

        with torch.no_grad():
            # 1. Retriever: query embedding
            q_emb = retriever.encode_query(queries)  # (B, query_dim)
            # 2. Vision encoder: patch tokens
            patch_tokens = model.encode_images(images_per_page)  # (B, K, N, D)

        K, N = patch_tokens.shape[1], patch_tokens.shape[2]
        r = compute_keep_ratio(budget_total, K, N)

        # 3. Token selector
        pruned_tokens, scores = selector(q_emb, patch_tokens, r)

        # 4. Generator: assemble context and forward
        # pruned_tokens를 SEP로 연결하여 generator 입력
        logits = model.generate_with_visual_tokens(queries, pruned_tokens)

        # 5. Loss
        vqa_loss = cross_entropy(logits, labels)
        h_loss = hinge_loss(scores, answer_page_labels)
        loss = vqa_loss + lambda_hinge * h_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: vqa_loss={vqa_loss.item():.4f}, hinge={h_loss.item():.4f}")
```

### 9.4 평가 루프

```python
def evaluate(model, retriever, selector, dataloader, budget_total=2048):
    selector.eval()
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in dataloader:
            queries, images_per_page, gold_answers, _ = batch
            q_emb = retriever.encode_query(queries)
            patch_tokens = model.encode_images(images_per_page)
            K, N = patch_tokens.shape[1], patch_tokens.shape[2]
            r = compute_keep_ratio(budget_total, K, N)
            pruned_tokens, _ = selector(q_emb, patch_tokens, r)
            preds = model.generate_answer(queries, pruned_tokens)
            all_preds.extend(preds)
            all_golds.extend(gold_answers)

    anls = compute_anls(all_preds, all_golds)
    return anls
```

---

## 10. 결과 저장 계획

- 학습 체크포인트: `checkpoints/exp001_seed{seed}/step{step}.pt`
- 평가 결과: `results/exp001_results.json` (benchmark × method × seed 테이블)
- 결과 분석: `result_001.md` (실측값 / 통계 / 판정 / 다음 단계) — 실험 완료 후 작성

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-26 | 초안 작성 (experiment_001.md v1.0) |

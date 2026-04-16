# 핵심 알고리즘: UCE & UCE-EWC

## 1. UCE (Unified Concept Editing) — Baseline

### 개요

Gandikota et al., *Unified Concept Editing in Diffusion Models*, 2024

Stable Diffusion UNet의 cross-attention (attn2) **value projection weight W_v**를  
closed-form solution으로 직접 수정하여 개념을 소거.

### 수식

소거할 개념 embedding $c_e$, 보존할 개념 embeddings $\{c_p^i\}$, anchor (빈 프롬프트) embedding $c_*$가 주어질 때:

**Key matrix** (개념 embedding들을 열로 쌓음):
$$K = [c_e, c_p^1, c_p^2, \ldots, c_p^n] \in \mathbb{R}^{d_k \times (1+n)}$$

**Target value matrix** (소거 개념 → anchor로, 보존 개념 → 현재 W_v 출력 유지):
$$V = [W_v c_*, W_v c_p^1, \ldots, W_v c_p^n] \in \mathbb{R}^{d_v \times (1+n)}$$

**Closed-form update**:
$$W_v^{\text{new}} = V K^\top (K K^\top + \lambda I)^{-1}$$

### 구현 핵심

```python
def apply_uce_erasure(pipeline, concept_to_erase, preserve_concepts, lamb=0.1):
    c_erase  = get_text_embed(pipeline, concept_to_erase)   # [d_k]
    c_pres   = stack([get_text_embed(p) for p in preserve_concepts])  # [n, d_k]
    v_anchor = get_text_embed(pipeline, "")                 # [d_k]

    for each attn2 layer:
        W_v = module.to_v.weight.data                       # [d_v, d_k]

        K    = cat([c_erase, c_pres]).T                     # [d_k, 1+n]
        V    = cat([W_v @ v_anchor, W_v @ c_pres.T])        # [d_v, 1+n]
        KT   = K.T                                          # [1+n, d_k]

        W_new = V @ KT @ inverse(K @ KT + λ·I)
        module.to_v.weight.data = W_new
```

### 임베딩 추출 (수정됨)

```python
def get_text_embed(pipeline, prompt):
    tokens = tokenizer(prompt, max_length=77, padding="max_length")
    embeds = text_encoder(tokens)[0]          # [1, 77, 768]
    # BOS / EOS / PAD 제외한 실제 토큰 평균
    mask = ~(token is BOS or EOS or PAD)
    if mask.sum() > 0:
        return embeds[0][mask].mean(dim=0)    # 다중 토큰 평균
    else:
        return embeds[0, EOS_position]        # 빈 문자열 fallback
```

> **수정 이유**: 기존 코드 `[0, 1]` 방식은 "Van Gogh"에서 "Van" 토큰만 추출.  
> 개선 후 다중 토큰 개념 (Van Gogh, Wonder Woman 등) 올바르게 처리.

### 한계

순차 적용 시 **Catastrophic Forgetting** 발생:

```
Superman 소거 → Van Gogh 소거 시, Superman 소거 효과 일부 복원
Van Gogh 소거 → Snoopy 소거 시, Superman + Van Gogh 소거 효과 추가 복원
```

---

## 2. UCE-EWC (Ours) — Catastrophic Forgetting 방지

### 아이디어

EWC (Elastic Weight Consolidation, Kirkpatrick et al., 2017)의 핵심 원리를 UCE의 closed-form에 통합:

> 이전 erasure에서 **많이 변화한 weight 방향**은 그 erasure에 중요한 방향이므로,  
> 다음 erasure에서 해당 방향의 변화를 억제 → Forgetting 방지

### 수식

**Importance 계산** (step $t$ 완료 후):
$$F^{(t)}[l] = \left(W_v^{(t)}[l] - W_v^{(t-1)}[l]\right)^2 \in \mathbb{R}^{d_v \times d_k}$$

**누적 importance** (모든 이전 step 합산):
$$F_{\text{acc}}[l] = \sum_{t=1}^{T-1} F^{(t)}[l]$$

**Input dimension별 중요도** (output 차원 평균):
$$f[l] = \frac{1}{d_v} \sum_{i} F_{\text{acc}}[l]_{i,:} \in \mathbb{R}^{d_k}$$

**수정된 정규화 행렬**:
$$R = \lambda I_{d_k} + \alpha \cdot \text{diag}\left(\frac{f[l]}{\max f[l]}\right)$$

**UCE-EWC closed-form**:
$$W_v^{\text{new}} = V K^\top (K K^\top + R)^{-1}$$

> UCE 대비 유일한 변화: $\lambda I$ → $\lambda I + \alpha \cdot \text{diag}(F)$

### 구현

```python
def apply_uce_ewc_erasure(pipeline, concept_to_erase, preserve_concepts,
                           importance_dict=None, alpha=1.0, lamb=0.1):
    # K, V 구성은 UCE와 동일
    ...
    for each attn2 layer:
        # EWC 정규화 항
        reg = lamb * I(d_k)
        if importance_dict[layer] exists:
            f_imp = importance_dict[layer].mean(dim=0)   # [d_k]
            f_imp = f_imp / f_imp.max()                  # [0, 1] 정규화
            reg  += alpha * diag(f_imp)

        W_new = V @ KT @ inverse(K @ KT + reg)
        module.to_v.weight.data = W_new


def compute_importance(prev_weights, curr_weights):
    """Weight 변화량 제곱 = EWC importance proxy."""
    return {name: (curr[name] - prev[name]) ** 2
            for name in prev_weights}


def accumulate_importance(imp_a, imp_b):
    """여러 step importance 누적 합산."""
    return {name: imp_a.get(name, 0) + imp_b.get(name, 0)
            for name in (imp_a | imp_b)}
```

### 순차 소거 루프

```python
accumulated_imp = None

for concept in ["Superman", "Van Gogh", "Snoopy"]:
    prev_w = snapshot(W_v)                              # 소거 전 weight 저장
    apply_uce_ewc_erasure(pipe, concept, ...,
                           importance_dict=accumulated_imp,
                           alpha=1.0)
    curr_w = snapshot(W_v)                              # 소거 후 weight
    step_imp = compute_importance(prev_w, curr_w)       # 이번 step importance
    accumulated_imp = accumulate(accumulated_imp,       # 누적
                                  step_imp)
```

### UCE vs UCE-EWC 비교

| 항목 | UCE | UCE-EWC |
|------|-----|---------|
| 정규화 행렬 | $\lambda I$ | $\lambda I + \alpha \cdot \text{diag}(F)$ |
| 이전 erasure 고려 | ✗ | ✓ |
| Catastrophic forgetting | 발생 | 완화 |
| 추가 계산 비용 | 없음 | weight 차분 제곱 (무시 가능) |
| 학습 불필요 | ✓ | ✓ |

### 하이퍼파라미터

| 파라미터 | 의미 | 기본값 |
|----------|------|--------|
| `lamb` | UCE 기본 정규화 강도 | 0.1 |
| `alpha` | EWC importance 가중치 | 1.0 |

**alpha 효과**:
- 작을수록 (≈0): UCE와 동일 (forgetting 발생)
- 클수록: 이전 erasure 보호 강화, 단 새 개념 소거 약화 가능
- 실험상 alpha=1.0이 균형점

---

## 3. UCE-Batch — 단일 Shot 동시 소거 (비교군)

### 개요

모든 소거 개념을 K 행렬에 한꺼번에 포함하여 단일 closed-form으로 해결.  
순차 업데이트가 없으므로 forgetting이 원천 차단되나, 소거 집중도가 희석됨.

### 수식

모든 소거 개념 $\{c_{e1}, c_{e2}, c_{e3}\}$와 전체 보존 개념 $\{c_p^j\}$를 동시에 처리:

$$K = [c_{e1}, c_{e2}, c_{e3}, c_p^1, \ldots, c_p^{12}] \in \mathbb{R}^{d_k \times 15}$$

$$V = [\underbrace{W_v c_*, W_v c_*, W_v c_*}_{\text{소거 → anchor}}, \underbrace{W_v c_p^1, \ldots, W_v c_p^{12}}_{\text{보존 → 현재 출력 유지}}]$$

$$W_v^{\text{new}} = V K^\top (K K^\top + \lambda I)^{-1}$$

UCE와 수식 구조가 동일하며 유일한 차이는 K에 소거 개념이 3개 동시 포함된다는 점.

### UCE vs UCE-EWC vs UCE-Batch 비교

| 항목 | UCE | UCE-EWC | UCE-Batch |
|------|-----|---------|-----------|
| 업데이트 횟수 | 3회 (순차) | 3회 (순차) | 1회 |
| Forgetting | 발생 | 완화 | 없음 |
| 소거 강도 | 강 | 강 | 약 (집중도 희석) |
| 보존 품질 | 중 | 중 | 강 |
| K 행렬 크기 | d_k × 5 (1소거+4보존) | d_k × 5 | d_k × 15 (3소거+12보존) |

### 구현

```python
def apply_uce_batch_erasure(pipeline, erase_concepts, preserved_dict, lamb=0.1):
    v_anchor      = get_text_embed(pipeline, "")
    erase_embeds  = [get_text_embed(pipeline, c) for c in erase_concepts]
    all_preserve  = [get_text_embed(p) for p in 보존 개념 전체 (중복 제거)]

    for each attn2 layer:
        K    = stack(erase_embeds + all_preserve).T     # [d_k, 15]
        V    = cat([W_v @ anchor] * 3 + [W_v @ pres.T])  # [d_v, 15]
        W_new = V @ K^T @ inverse(K @ K^T + λ·I)
```

---

## 알고리즘 의사코드 (보고서용)

```
Algorithm: UCE-EWC Sequential Concept Erasure

Input:  SD model M, concepts C = [c1, c2, c3], preserve P = {c: [p1..p4]}
Output: Edited model M* with all concepts in C erased

Initialize: accumulated_importance F = None

For t = 1, 2, 3:
    W_prev ← snapshot of attn2 to_v weights in M
    
    Compute embeddings:
        e_t   ← TextEmbed(c_t)          // average over non-special tokens
        e_p   ← TextEmbed(p) for p in P[c_t]
        e_*   ← TextEmbed("")            // anchor

    For each attn2 layer l in M.UNet:
        K  ← [e_t, e_p^1, ..., e_p^n]^T       // [d_k, 1+n]
        V  ← [W_v^l e_*, W_v^l e_p^1, ..., W_v^l e_p^n]  // [d_v, 1+n]

        // EWC regularization
        If F is not None:
            f ← F[l].mean(dim=0) / max(F[l].mean(dim=0))  // [d_k]
            R ← λI + α·diag(f)
        Else:
            R ← λI

        W_v^l ← V K^T (K K^T + R)^{-1}        // closed-form update

    W_curr ← snapshot of updated attn2 to_v weights
    F      ← F + (W_curr - W_prev)^2           // accumulate importance

Return M
```

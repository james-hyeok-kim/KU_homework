# Continual Concept Erasure — 실험 설계

## 과제 개요

- **과목**: Algorithm for AI (고려대학교 AI학과, Gyeong-Moon Park 교수)
- **과제**: Continual Concept Erasure (Continual CE)
- **마감**: 2026-04-21
- **Reference**: Lyu et al., *One-dimensional Adapter to Rule Them All*, CVPR 2024 (SPM)

---

## 문제 정의

세 사용자가 순차적으로 개념 소거를 요청하는 시나리오:

```
Superman → Van Gogh → Snoopy
```

**목표**: 마지막 소거 후에도 세 개념이 모두 지워진 상태 유지 + 나머지 개념 보존

**핵심 문제**: weight를 직접 수정하는 방법(UCE 등)을 순차 적용하면,  
새 개념을 지울 때 이전에 지운 개념이 복원되는 **Catastrophic Forgetting** 발생

---

## 실험 설정

| 항목 | 설정 |
|------|------|
| Backbone | Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4) |
| Precision | float16 + CPU offload + attention slicing |
| 소거 순서 | Superman → Van Gogh → Snoopy |
| 평가 metrics | CLIP Score (CS), FID |

### 평가 개념 목록

| 소거 개념 | 관련 보존 개념 (_r) |
|-----------|---------------------|
| Superman | Batman, Thor, Wonder Woman, Shazam |
| Van Gogh | Picasso, Monet, Paul Gauguin, Caravaggio |
| Snoopy | Mickey, Spongebob, Pikachu, Hello Kitty |

### 평가 프롬프트

- **소거 개념**: `"A photo of {concept}"` / Van Gogh는 `"A painting in the style of {concept}"`
- **보존 개념**: 개념명 직접 사용 (e.g., `"Batman"`, `"Picasso"`)
- **일반 품질**: MS-COCO captions 50개 (`sentence-transformers/coco-captions`)

---

## 방법론

### Baseline: UCE (Unified Concept Editing)

Cross-attention (attn2) to_v weight를 closed-form으로 수정.  
순차 적용 시 catastrophic forgetting 발생 → 분석 대상.

### Ours: UCE-EWC

UCE에 EWC(Elastic Weight Consolidation) 스타일 정규화 추가.  
이전 erasure에서 변화량이 큰 weight를 보호하여 forgetting 방지.

### 비교군: UCE-Batch (단일 shot 동시 소거)

Superman, Van Gogh, Snoopy를 순차적으로 지우는 대신 **한 번의 closed-form으로 동시에 처리**:

```
K = [c_superman, c_vangogh, c_snoopy, 모든 보존 개념들(12개)]
V = [W_v c_*,    W_v c_*,   W_v c_*,  W_v c_p1, ..., W_v c_p12]
W_new = V K^T (K K^T + λI)^{-1}
```

**역할**: Sequential 방법들과의 비교군.  
- Forgetting이 원천적으로 발생하지 않음 (단일 업데이트)
- 소거와 보존 간 trade-off의 한 극단을 대표
- Sequential UCE/UCE-EWC 대비 소거가 약하고 보존이 강한 특성 → "보존 품질 ceiling"

---

## 실험 구성

### 메인 실험

| Stage | 방법 | 내용 |
|-------|------|------|
| 0 | SD v1.4 원본 | baseline 측정 + FID reference 이미지 생성 |
| 1 | UCE (baseline) | 3개 순차 소거 + **step-by-step 중간 평가** |
| 2 | UCE-EWC (Ours) | 3개 순차 소거 + **step-by-step 중간 평가** |
| 3 | UCE-Batch | 3개 동시 단일 shot 소거 |

### Step-by-step 중간 평가 (Forgetting Curve)

각 소거 step 직후, **이전에 지운 개념들을 모두 재평가**:

```
Superman 소거 후  → Superman_e CS 측정
Van Gogh 소거 후  → Superman_e CS, Van Gogh_e CS 측정
Snoopy 소거 후    → Superman_e CS, Van Gogh_e CS, Snoopy_e CS 측정
```

UCE는 Van Gogh를 지우면 Superman_e CS가 다시 상승(forgetting),  
UCE-EWC는 낮게 유지(protection) → 핵심 비교 지점

### Ablation 1: alpha 강도 sweep

EWC 정규화 강도 alpha ∈ {0.01, 0.1, 1.0, 10.0}

- **측정**: 소거 CS + **보존 CS** (Van Gogh_r 역전 현상 분석 포함)
- **목적**: forgetting 방지 vs. erasure 효과 trade-off 파악

### Ablation 2: 소거 순서 민감도

역순 소거: **Snoopy → Van Gogh → Superman**

- UCE vs. UCE-EWC 각각 실험
- step-by-step forgetting curve 포함
- UCE-EWC가 순서에 더 강건한지 확인

---

## 샘플 수 설정

| 평가 종류 | 샘플 수 | 이유 |
|-----------|---------|------|
| 메인 테이블 (concept) | 20 | FID 신뢰도 향상 (기존 10) |
| MS-COCO | 50 | 일반 품질 평가 |
| Step-by-step | 8 | 빠른 CS 추세 파악 |
| Ablation | 6 | 신속한 sweep |

---

## FID 계산 방법

- **기존 문제**: `torch.zeros` (검은 이미지) 대비 FID → 600~800, 무의미
- **수정**: Stage 0에서 원본 SD v1.4가 생성한 이미지를 real reference로 저장
- **의미**: 각 방법이 원본 SD의 생성 분포에서 얼마나 벗어났는지 측정

---

## 임베딩 추출 수정

- **기존**: `encoder_output[0, 1]` → 첫 번째 토큰만 사용 ("Van Gogh"에서 "Van"만 추출)
- **수정**: BOS/EOS/PAD 제외한 실제 토큰 전체 평균 → 다중 토큰 개념 올바르게 처리

---

## 코드 파일

- **메인 구현**: `continual_CE.py`
- **실험 실행**: `run_continual_CE.sh`
- **결과 저장**: `results/quantitative_table.json`
- **로그**: `logs/experiment.log`

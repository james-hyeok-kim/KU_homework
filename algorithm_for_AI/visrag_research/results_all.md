# VisRAG 개선 실험 전체 결과 요약

**프로젝트**: VisRAG (Vision-based RAG) 성능 개선 연구  
**실험 기간**: 2026-05-26 ~ 2026-05-29 KST  
**환경**: NVIDIA B200 (183GB) × GPU 0~3  
**Generator**: Qwen2-VL-7B-Instruct | **Retrieval**: Oracle (gold pages, BEIR qrels)

---

## 방법별 아키텍처 다이어그램

> **VLM 내부 구조**: Qwen2-VL-7B는 두 단계로 구성됨
> - **Vision Encoder** (ViT 계열): 이미지 픽셀 → visual patch tokens (N × D)
> - **LLM Generator** (language decoder): text tokens + patch tokens → answer
>
> OCR 단계도 동일한 Vision Encoder → LLM 파이프라인을 거친다.
> QCVTP (Exp 001)의 token selector는 두 단계 사이에 삽입된다.

---

### Baseline — 순수 VisRAG (이미지만)

```
Query ──► [VisRAG Retriever]
               │
          Top-K pages (oracle gold pages)
               │
         [Images × K]
               │
   [Vision Encoder (ViT)]       ← 이미지 → patch tokens
               │
      patch tokens × K
               │
   [LLM Generator (Decoder)]    ← text query + patch tokens
               │
           Answer
```

---

### Exp 002 — DocParse Hybrid (일괄 OCR)

```
Query ──► [VisRAG Retriever]
               │
          Top-K pages
          /          \
    [Images × K]   [Vision Encoder]  ← OCR 전용 pass
                        │
                   patch tokens × K
                        │
                   [LLM Generator]   ← "이미지를 텍스트로 추출" 프롬프트
                        │             (일반 OCR, 모든 페이지 일괄)
                   OCR Text × K
          \              /
     [Images + OCR Text] × K         ← 이미지 + OCR 텍스트 동시 입력
               │
   [Vision Encoder (ViT)]
               │
      patch tokens + text tokens
               │
   [LLM Generator (Decoder)]
               │
           Answer

장점: InfoVQA +1.19%p (혼합 콘텐츠 보조)
단점: ChartQA -1.84%p (차트 수치 OCR 오류가 혼란 유발)
```

---

### Exp 003 — Selective Routing (페이지 유형별 OCR 선택)

```
Query ──► [VisRAG Retriever]
               │
          Top-K pages
               │
         [Vision Encoder]       ← 분류 전용 pass
               │
          patch tokens
               │
   [LLM Generator]  ← "chart / text / mixed?" 프롬프트
               │
          Page Label
       /        |        \
  chart(83%)  text(~50%) mixed(17%~100%)
  [skip]      [skip]    [Vision Encoder]   ← OCR pass
                               │
                          patch tokens
                               │
                        [LLM Generator]    ← 일반 OCR 프롬프트
                               │
                          OCR Text
       \        |            /
   [Image]  [Image]   [Image + Text]
   (chart)  (text)      (mixed)
               │
   [Vision Encoder (ViT)]
               │
      patch tokens (+ text tokens if mixed)
               │
   [LLM Generator (Decoder)]
               │
           Answer

장점: DocVQA +1.04%p (text 스킵 + mixed OCR)
단점: ChartQA -3.17%p (17% mixed 오판이 핵심 손실)
```

---

### Exp 004 — Chart-Aware Routing (차트 특화 OCR)

```
Query ──► [VisRAG Retriever]
               │
          Top-K pages
               │
         [Vision Encoder]       ← 분류 전용 pass
               │
          patch tokens
               │
   [LLM Generator]  ← "chart / text / mixed?" 프롬프트
               │
          Page Label
       /        |        \
  chart(83%)  text(~50%) mixed(17%~100%)
[Vis.Enc.]   [skip]    [Vision Encoder]
      │                      │
 patch tokens           patch tokens
      │                      │
[LLM Gen.]             [LLM Generator]
 차트 특화 OCR 프롬프트    일반 OCR 프롬프트
 (제목/축/범례/수치)
      │                      │
 Chart Text             General Text
       \        |            /
 [Image+Chart  [Image]  [Image+Text]
   Text]       (text)     (mixed)
  (chart)
               │
   [Vision Encoder (ViT)]
               │
      patch tokens (+ text tokens)
               │
   [LLM Generator (Decoder)]
               │
           Answer

차이 vs Exp 003: chart pass를 skip → 차트 특화 OCR로 교체
효과: ChartQA에서 Selective 대비 +0.69%p 개선, 그러나 baseline 미달
```

---

### Phase 2 — Generator 교체 (Oracle Retrieval, OCR 없음)

```
Query ──► [VisRAG Retriever]
               │
          Top-K pages (oracle)
               │
         [Images × K]
               │
   [Vision Encoder (ViT)]       ← 모델별로 구조 상이
               │
      patch tokens × K
               │
   [LLM Generator (Decoder)]    ← 모델만 교체, OCR 없음
   ┌───────┬──────────┬──────────┐
  [7B]  [InternVL  [Qwen2-VL  [Qwen2.5-VL
        78B]        72B]        72B]
 base  -6.27%p    +5.86%p     +5.39%p
   └───────┴──────────┴──────────┘
               │
           Answer

핵심: 모델 계열(Qwen vs InternVL)이 스케일보다 결정적
```

---

### 방법 비교 요약

```
                    VisRAG Pipeline
┌──────────────────────────────────────────────────────┐
│  Query ──► Retriever ──► Top-K pages                 │
│                               │                      │
│  ┌────────────────────────────▼──────────────────┐   │
│  │     OCR Pre-pass (선택적, 별도 VLM 추론)       │   │
│  │  Baseline:   없음                             │   │
│  │  Hybrid:     Vision Enc. → LLM (일반 OCR)    │   │
│  │  Selective:  mixed만 → Vision Enc. → LLM     │   │
│  │  ChartAware: chart→chart OCR, mixed→일반 OCR │   │
│  └────────────────────────────┬──────────────────┘   │
│                    OCR Text (optional)                │
│                               │                      │
│  ┌────────────────────────────▼──────────────────┐   │
│  │     Answer Generation pass                    │   │
│  │  [Vision Encoder] image → patch tokens        │   │
│  │           +                                   │   │
│  │  [LLM Decoder]   patch tokens + text → answer │   │
│  └────────────────────────────┬──────────────────┘   │
│                               │                      │
│                           Answer                     │
└──────────────────────────────────────────────────────┘

※ OCR pre-pass와 answer generation pass 모두 동일한
  Qwen2-VL-7B 모델을 사용 (Vision Encoder + LLM 공유)
```

---

## 실험 목록

| # | 실험명 | 접근 방식 | 상태 |
|---|--------|----------|------|
| 002 | DocParse Hybrid | Qwen2-VL OCR 텍스트 + 이미지 동시 입력 | ✅ 완료 |
| 003 | Selective Routing | 페이지 유형 분류 후 mixed 페이지만 OCR | ✅ 완료 |
| 004 | Chart-Aware Routing | 페이지 유형 분류 후 차트 특화 OCR + mixed 일반 OCR | ✅ 완료 |
| 005 | CoT (Chain-of-Thought) | max_new_tokens 32→256, 단계적 추론 프롬프트 | ❌ 실패 (ChartQA -7.7%p) |
| 006 | PoT (Program-of-Thought) | 차트→JSON→Python 코드 생성→실행 | ❌ 실패 (100% fallback) |

---

---

## 전체 실험 통합 비교표

> **측정 지표**: ANLS (Average Normalized Levenshtein Similarity)  
> **Retrieval**: Oracle (gold pages)  
> **Generator**: Qwen2-VL-7B (Phase 1) / 교체 모델 (Phase 2)

### Phase 1 — OCR Routing 방법 비교 (Generator: Qwen2-VL-7B 고정)

| Method | InfoVQA | ChartQA | DocVQA | SlideVQA | **평균 Δ vs Baseline** |
|--------|--------:|--------:|-------:|---------:|----------------------:|
| **Baseline** (순수 VisRAG) | 0.7835 | **0.6941** | 0.9303 | 0.7458 | — |
| **Hybrid (002)** (일괄 OCR) | **0.7954** | 0.6757 | 0.9259 | **0.7472** | **-0.24%p** |
| **Selective (003)** (mixed만 OCR) | **0.7954** | 0.6624 | **0.9407** | 0.7469 | **-0.21%p** |
| **Chart-Aware (004)** (chart 특화 OCR) | **0.7954** | 0.6693 | 0.9341 | 0.7412 | **-0.34%p** |

#### Baseline 대비 Δ (%p)

| Method | InfoVQA | ChartQA | DocVQA | SlideVQA | **평균 Δ** |
|--------|--------:|--------:|-------:|---------:|-----------:|
| Hybrid (002) | +1.19 | -1.84 | -0.44 | +0.14 | **-0.24** |
| Selective (003) | +1.19 | -3.17 | **+1.04** | +0.11 | **-0.21** |
| Chart-Aware (004) | +1.19 | -2.48 | +0.38 | -0.46 | **-0.34** |

---

### Phase 2 — Generator 교체 비교 (OCR 없음, Oracle Retrieval)

| Generator Model | ChartQA | DocVQA | InfoVQA | SlideVQA | **평균 Δ vs 7B** |
|-----------------|--------:|-------:|--------:|---------:|----------------:|
| **Qwen2-VL-7B** (baseline) | 0.6941 | 0.9303 | 0.7835 | 0.7458 | — |
| **InternVL2.5-78B** | 0.6709 | 0.8939 | 0.6076 | 0.7304 | **-6.27%p** ✗ |
| **Qwen2-VL-72B** | **0.7770** | **0.9657** | 0.8690 | **0.7762** | **+5.86%p** ★ |
| **Qwen2.5-VL-72B** | 0.7349 | 0.9664 | **0.8854** | 0.7664 | **+5.39%p** |

#### Generator 대비 Δ (%p)

| Generator Model | ChartQA | DocVQA | InfoVQA | SlideVQA | **평균 Δ** |
|-----------------|--------:|-------:|--------:|---------:|-----------:|
| InternVL2.5-78B | -2.32 | -3.64 | -17.59 | -1.54 | **-6.27** |
| Qwen2-VL-72B | **+8.29** | +3.54 | +8.55 | **+3.04** | **+5.86** |
| Qwen2.5-VL-72B | +4.08 | **+3.61** | **+10.19** | +2.06 | **+5.39** |

---

### 데이터셋별 최고 성능 정리

| Dataset | 최고 방법 | ANLS | Phase |
|---------|---------|-----:|-------|
| **InfoVQA** | Qwen2.5-VL-72B (Gen 교체) | **0.8854** | Phase 2 |
| **ChartQA** | Qwen2-VL-72B (Gen 교체) | **0.7770** | Phase 2 |
| **DocVQA** | Qwen2.5-VL-72B (Gen 교체) | **0.9664** | Phase 2 |
| **SlideVQA** | Qwen2-VL-72B (Gen 교체) | **0.7762** | Phase 2 |

> Phase 1 OCR 방법은 7B generator 한계 내에서 +1%p 수준 개선 / Phase 2 generator 업그레이드가 압도적

---


## 전체 ANLS 결과표

| Dataset | Baseline | DocParse Hybrid (002) | Selective (003) | Chart-Aware (004) |
|---------|---------|----------------------|-----------------|-------------------|
| SlideVQA | 0.7458 | **0.7472** | 0.7469 | 0.7412 |
| DocVQA | 0.9303 | 0.9259 | **0.9407** | 0.9341 |
| InfoVQA | 0.7835 | **0.7954** | **0.7954** | **0.7954** |
| ChartQA | **0.6941** | 0.6757 | 0.6624 | 0.6693 |

### Baseline 대비 Δ (ANLS %p)

| Dataset | DocParse Hybrid (002) | Selective (003) | Chart-Aware (004) |
|---------|----------------------|-----------------|-------------------|
| SlideVQA | +0.14%p | +0.11%p | -0.46%p |
| DocVQA | -0.44%p | **+1.04%p ★★** | +0.38%p |
| InfoVQA | **+1.19%p ★** | **+1.19%p ★** | **+1.19%p ★** |
| ChartQA | -1.84%p | -3.17%p ✗ | -2.48%p ✗ |
| **평균** | **-0.24%p** | **-0.21%p** | **-0.34%p** |

### Exact Match

| Dataset | Baseline | DocParse Hybrid (002) | Selective (003) | Chart-Aware (004) |
|---------|---------|----------------------|-----------------|-------------------|
| SlideVQA | 0.5791 | **0.5845** | 0.5881 | 0.5827 |
| DocVQA | 0.8951 | 0.8934 | **0.9069** | 0.8968 |
| InfoVQA | 0.7312 | **0.7382** | **0.7382** | **0.7382** |
| ChartQA | **0.6667** | 0.6190 | 0.6349 | 0.6032 |

---

## 실험별 상세 요약

### Exp 002 — DocParse Hybrid

**접근**: Qwen2-VL 7B를 OCR 엔진으로 활용해 이미지 + OCR 텍스트를 생성기에 동시 제공.

**결과**:

| Dataset | Δ vs Baseline | 원인 |
|---------|--------------|------|
| InfoVQA | **+1.19%p** ★ | 인포그래픽 복잡 텍스트 OCR 보완 |
| SlideVQA | +0.14%p | 슬라이드 텍스트 추출이 페이지 식별 보조 |
| DocVQA | -0.44%p | 선명한 텍스트에 OCR 중복 → 노이즈 |
| ChartQA | -1.84%p | 차트 수치 OCR 오류가 생성기 혼란 유발 |

**특이사항**: Text-only(OCR만, 이미지 없음)는 -15%p~-30%p 급락 → 시각 정보 필수

---

### Exp 003 — Selective OCR Routing

**접근**: Qwen2-VL 분류기로 각 페이지를 chart/text/mixed 분류 → mixed만 OCR 적용.

**페이지 분류 통계**:

| Dataset | chart | text | mixed |
|---------|-------|------|-------|
| ChartQA | 83% | 0% | 17% |
| DocVQA | ~0% | ~50% | ~50% |
| InfoVQA | 0% | 0% | ~100% |

**결과**:

| Dataset | Δ vs Baseline | 원인 |
|---------|--------------|------|
| DocVQA | **+1.04%p** ★★ | text(50%) 스킵 + mixed(50%) OCR → 일괄 OCR보다 정교 |
| InfoVQA | **+1.19%p** ★ | 전 페이지 mixed → Hybrid와 동일 경로 |
| SlideVQA | +0.11%p | baseline에 근접 |
| ChartQA | -3.17%p ✗ | mixed 17%에 일반 OCR → 어려운 케이스 심각 손실 |

**핵심**: DocVQA에서 일괄 OCR(-0.44%p) → 선택적 OCR(+1.04%p)로 **+1.48%p 역전**

---

### Exp 004 — Chart-Aware OCR Routing

**접근**: chart 페이지에 차트 특화 OCR 프롬프트 (축/수치/범례 구조화 추출) 적용.  
text → 스킵, mixed → 일반 OCR (Exp003과 동일).

**결과**:

| Dataset | Δ vs Baseline | vs Selective (003) |
|---------|--------------|-------------------|
| InfoVQA | **+1.19%p** | =(동일, 캐시 재활용) |
| ChartQA | -2.48%p | **+0.69%p 개선** |
| DocVQA | +0.38%p | -0.66%p |
| SlideVQA | -0.46%p | -0.57%p |

**특이사항**: ChartQA에서 차트 특화 OCR이 일반 OCR보다 효과적이지만 baseline 미달

---

## 방법별 총평

| 방법 | 핵심 강점 | 핵심 약점 | 추천 용도 |
|------|---------|---------|---------|
| **DocParse Hybrid (002)** | InfoVQA +1.19%p | ChartQA -1.84%p | 인포그래픽/복합 문서 |
| **Selective Routing (003)** | DocVQA +1.04%p ★★, 평균 최고 | ChartQA -3.17%p ✗ | 텍스트/혼합 문서 |
| **Chart-Aware (004)** | ChartQA에서 Selective보다 +0.69%p | 다른 데이터셋에서 Selective 이하 | 차트 중심 데이터 한정 |

---

## 데이터셋별 최적 전략

| Dataset | 최고 방법 | ANLS | 전략 |
|---------|---------|------|------|
| **InfoVQA** | Hybrid = Selective = Chart-Aware | **0.7954** | OCR 적용 (방법 무관) |
| **DocVQA** | Selective Routing | **0.9407** | 선택적 OCR (text 스킵) |
| **SlideVQA** | DocParse Hybrid | **0.7472** | 전체 OCR 약한 보조 |
| **ChartQA** | Baseline (OCR 없음) | **0.6941** | OCR 미사용 — 시각 직접 인식 |

---

## 핵심 인사이트

1. **선택적 OCR > 일괄 OCR**: Selective Routing이 DocVQA에서 Hybrid 대비 +1.48%p 역전
2. **차트는 OCR이 역효과**: ChartQA에서 모든 OCR 방법이 baseline 이하
3. **분류기 오류가 성능 좌우**: ChartQA 17% "mixed" 오판이 핵심 손실 원인
4. **InfoVQA는 전략 무관**: 분류기가 전 페이지를 mixed 판정 → 세 방법 동일
5. **시각 정보 필수**: Text-only에서 -15%p~-30%p → VisRAG 핵심은 이미지

---

---

## 부정적 실험 결과: CoT / PoT (Exp 005, 006)

### Exp 005 — CoT (Chain-of-Thought), ChartQA

**접근**: max_new_tokens 32→256, "Think step by step. Answer: [value]" 프롬프트.

| 지표 | Baseline | CoT | Δ |
|------|---------|-----|---|
| ChartQA ANLS | **0.6941** | 0.6169 | **-7.7%p ✗** |
| ChartQA EM | **0.6667** | 0.5556 | -11.1%p ✗ |

**실패 원인**:
- 41% 샘플이 문장형 출력: "The percentage is 3.6%." → ANLS 계산 시 "3.6" 대비 ≈0점
- 모델이 "Answer:" 태그 지시를 무시하고 설명형 답변 생성
- 오히려 CoT 추론 과정에서 잘못된 값을 읽는 경우 발생 (33→83 등)
- **근본 원인**: ANLS 메트릭은 문자열 유사도 기반 → 긴 답변이 짧은 정답 대비 치명적 페널티

**교훈**: max_new_tokens=32 baseline은 짧은 답변을 강제하는 암묵적 정규화 역할을 함. 이를 늘리면 출력 형식이 바뀌어 메트릭 상 역효과.

---

### Exp 006 — PoT (Program-of-Thought), ChartQA (15샘플 후 중단)

**접근**: Qwen2-VL로 차트→JSON 추출 → Python 코드 생성 → subprocess 실행.

| 경로 | 비율 |
|------|------|
| fallback_json_parse_error | 20% (JSON 파싱 실패) |
| fallback_code_exec_error | 80% (코드 생성 실패) |
| pot_success | **0%** |

**실패 원인**:
- Stage 1 (JSON): 7B VLM이 복잡한 차트를 완전한 JSON으로 표현 실패 (512 token 내 truncation)
- Stage 2 (Code): 모델이 Python 코드 대신 답 숫자("68")를 직접 출력 — 코드 생성 지시 무시
- **근본 원인**: Qwen2-VL-7B는 범용 VLM으로 코드 생성에 최적화되지 않음. GPT-4V/Claude급 모델 필요

---

## 핵심 인사이트

1. **선택적 OCR > 일괄 OCR**: Selective Routing이 DocVQA에서 Hybrid 대비 +1.48%p 역전
2. **차트는 OCR이 역효과**: ChartQA에서 모든 OCR 방법이 baseline 이하
3. **CoT/PoT는 메트릭과 충돌**: ANLS는 문자열 유사도 기반 → 긴 추론 출력이 치명적 페널티
4. **7B VLM의 한계**: 구조화 출력(JSON) 및 코드 생성은 7B 모델로 신뢰할 수 없음
5. **Oracle retrieval 제약**: 데이터셋이 gold pages 제공 → retriever 교체(ColQwen2 등) 비교 불가
6. **InfoVQA는 전략 무관**: 분류기가 전 페이지를 mixed 판정 → 세 방법 동일

---

## ChartQA 한계 분석 (21개 오답 기준)

| 오류 유형 | 개수 | 해결책 |
|---------|-----|------|
| 산수 실패 (비율, 평균, 합계) | ~12 | PoT (단, 더 큰 모델 필요) |
| 시각적 정밀도 (bar 높이 misread) | ~6 | 고해상도 처리, 전문 모델 |
| 비교/논리 실패 | ~2 | CoT (단, 출력 형식 문제 해결 후) |
| 색상/레이블 혼동 | ~1 | 색상 강조 전처리 |

**현 Qwen2-VL-7B로 ChartQA를 이기는 것은 구조적으로 어려움** — 모델 자체 교체가 가장 효과적.

---

## 후속 실험 제안 (업데이트)

| 우선순위 | 실험 | 기대 효과 | 난이도 |
|---------|------|---------|------|
| 1 | **Generator 업그레이드**: Qwen2-VL-72B 또는 InternVL2-76B | ChartQA +10~15%p 예상 | 중 (모델 다운로드) |
| 2 | **Real retrieval 평가**: ColQwen2로 전체 corpus 인덱싱 후 비교 | SlideVQA 실제 retrieval 개선 측정 | 높음 |
| 3 | **Few-shot prompting**: 차트 읽기 예시 2-3개 포함 (images 없이 text만) | ChartQA 일부 개선 가능 | 낮음 |
| 4 | **정밀 페이지 분류기** | ChartQA mixed 오판 감소 | 중 |

---

*세부 결과: `result_002.md`, `result_003_004.md`*

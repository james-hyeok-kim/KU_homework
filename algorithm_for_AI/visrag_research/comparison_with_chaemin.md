# VisRAG 실험 비교: 내 실험 vs Chaemin (AI-Final-project)

**작성일**: 2026-06-06 KST  
**내 실험 경로**: `/home/jovyan/workspace/KU_homework/algorithm_for_AI/visrag_research`  
**비교 대상**: https://github.com/chaemin-0824/AI-Final-project

---

## 1. 프로젝트 개요 비교

| 항목 | 내 실험 (visrag_research) | Chaemin (AI-Final-project) |
|------|--------------------------|---------------------------|
| **목표** | VisRAG 내부 OCR 라우팅 + Generator 교체로 성능 개선 | VisRAG(image-only) vs Parsed-Visual RAG(PV-RAG) 비교 |
| **핵심 가설** | 페이지 유형(chart/text/mixed)에 따른 선택적 OCR이 일괄 OCR보다 효과적 | Upstage로 파싱한 텍스트+이미지 융합이 이미지 단독보다 월등 |
| **데이터셋** | InfoVQA, ChartQA, DocVQA, SlideVQA | InfoVQA, ChartQA, MP-DocVQA, SlideVQA |
| **Retrieval** | Oracle (gold pages, BEIR qrels) | Oracle (qrels) |
| **Generator** | Qwen2-VL-7B-Instruct (full precision, B200 GPU) | Qwen2-VL-7B-Instruct (NF4 4-bit 양자화) |
| **평가 지표** | **ANLS** (Average Normalized Levenshtein Similarity) | **Relaxed Exact Match** (숫자 5% tolerance) |
| **OCR 방식** | Qwen2-VL-7B 자체를 OCR 엔진으로 인라인 활용 | **Upstage Document Parse API** (외부 상용 파서) |
| **실험 GPU** | NVIDIA B200 (183GB) × GPU 0~3 | 일반 GPU (NF4 양자화로 ~5-6GB 사용) |

---

## 2. 실험 방법론 비교

### 2.1 내 실험 — OCR Routing 접근

```
Query → [Oracle Retriever] → Top-K pages
                                  │
                    [페이지 유형 분류 pass]  ← Qwen2-VL-7B
                     chart / text / mixed
                          │
           ┌──────────────┼──────────────┐
        chart           text           mixed
      [skip OCR]     [skip OCR]   [OCR pass]  ← Qwen2-VL-7B
                                       │
                                   OCR Text
           └──────────────┼──────────────┘
                    [Image (+ Text if mixed)]
                          │
                   [Qwen2-VL-7B Generator]
                          │
                        Answer
```

**실험 목록 (Phase 1)**

| 실험 | 방식 | 핵심 변경점 |
|------|------|-----------|
| Baseline | 이미지만 입력 | OCR 없음 |
| Exp 002 Hybrid | 모든 페이지에 일괄 OCR | Qwen → 일반 OCR → 이미지+텍스트 |
| Exp 003 Selective | mixed 페이지만 OCR | 분류기 → mixed만 OCR |
| Exp 004 Chart-Aware | chart 특화 OCR + mixed 일반 OCR | 축/수치/범례 구조화 추출 |
| Exp 005 CoT | Chain-of-Thought 프롬프트 (실패) | max_new_tokens 32→256 |
| Exp 006 PoT | Program-of-Thought, JSON+코드 (실패) | 차트→JSON→Python 실행 |

**Phase 2 — Generator 교체 (OCR 없음)**

| Generator | 평균 Δ vs 7B |
|-----------|------------|
| InternVL2.5-78B | -6.27%p |
| **Qwen2-VL-72B** | **+5.86%p** |
| Qwen2.5-VL-72B | +5.39%p |

---

### 2.2 Chaemin — Parsed-Visual RAG (PV-RAG) 접근

```
Query → [Oracle Retriever (qrels)] → Top-K pages
                                          │
                          ┌───────────────┘
                          │
                [Upstage Document Parse API]  ← 외부 상용 OCR
                     (텍스트 + 표 구조화)
                          │
                    Parsed Text/Table
                          │
           ┌──────────────┴──────────────┐
        image_only   parsed_text_only   parsed_visual
        (이미지만)   (텍스트만)         (이미지 + 텍스트)
           └──────────────┬──────────────┘
                          │
              [Qwen2-VL-7B-Instruct NF4]
                          │
                        Answer
```

**실험 목록**

| 모드 | 방식 | 설명 |
|------|------|------|
| `image_only` | 순수 VisRAG (Baseline) | 이미지만 입력 |
| `parsed_text_only` | Ablation | Upstage 텍스트만, 이미지 없음 |
| `parsed_visual` (PV-RAG) | 제안 방법 | Upstage 텍스트 + 이미지 동시 입력 |

---

## 3. 성능 결과 비교

### 3.1 내 실험 결과 (ANLS 기준)

**Phase 1 — OCR Routing (Generator: Qwen2-VL-7B 고정)**

| Method | InfoVQA | ChartQA | DocVQA | SlideVQA | 평균 Δ |
|--------|--------:|--------:|-------:|---------:|------:|
| **Baseline** | 0.7835 | **0.6941** | 0.9303 | 0.7458 | — |
| Hybrid (002) | **0.7954** | 0.6757 | 0.9259 | **0.7472** | -0.24%p |
| Selective (003) | **0.7954** | 0.6624 | **0.9407** | 0.7469 | -0.21%p |
| Chart-Aware (004) | **0.7954** | 0.6693 | 0.9341 | 0.7412 | -0.34%p |

**Phase 2 — Generator 교체 (OCR 없음)**

| Generator | ChartQA | DocVQA | InfoVQA | SlideVQA | 평균 Δ |
|-----------|--------:|-------:|--------:|---------:|------:|
| Qwen2-VL-7B (base) | 0.6941 | 0.9303 | 0.7835 | 0.7458 | — |
| InternVL2.5-78B | 0.6709 | 0.8939 | 0.6076 | 0.7304 | -6.27%p |
| **Qwen2-VL-72B** | **0.7770** | **0.9657** | 0.8690 | **0.7762** | **+5.86%p** |
| Qwen2.5-VL-72B | 0.7349 | 0.9664 | **0.8854** | 0.7664 | +5.39%p |

---

### 3.2 Chaemin 실험 결과 (Relaxed Exact Match 기준)

| Method | InfoVQA | ChartQA | DocVQA | SlideVQA | 평균 Δ |
|--------|--------:|--------:|-------:|---------:|------:|
| **Baseline** (image_only) | 0.2000 | 0.3651 | 0.1700 | 0.1800 | — |
| Parsed-Text Only | 0.5400 | 0.5238 | 0.7600 | 0.5200 | +35.72%p |
| **PV-RAG** (parsed_visual) | 0.5400 | 0.5079 | **0.8000** | 0.5200 | **+36.32%p** |

---

### 3.3 Baseline 직접 비교 (측정 지표 차이 주의)

| Dataset | 내 Baseline (ANLS) | Chaemin Baseline (Relaxed EM) |
|---------|------------------:|------------------------------:|
| InfoVQA | **0.7835** | 0.2000 |
| ChartQA | **0.6941** | 0.3651 |
| DocVQA | **0.9303** | 0.1700 |
| SlideVQA | **0.7458** | 0.1800 |

> **주의**: 수치 차이는 metric 차이가 주된 원인. ANLS는 부분 점수(편집거리 기반)를 부여하므로 같은 답이라도 Relaxed EM보다 높게 나옴. 직접 비교 불가.

---

## 4. 코드 구조 비교

### 4.1 내 실험 코드 구조

```
visrag_research/src/
├── visrag_pipeline.py     # VisRAG + QCVTP(Query-Conditioned Visual Token Pruning) 모듈
│                          #   retrieve → encode_pages → prune_tokens → generate
├── token_selector.py      # QueryConditionedTokenSelector (학습 가능한 relevance scoring)
├── evaluate.py            # ANLS 계산, 실험 결과 평가
├── train.py               # Token selector 학습 스크립트
├── smoke_test.py          # 환경/의존성 검증
└── run_experiment.sh      # 실험 실행 스크립트
```

**특이점**: `visrag_pipeline.py`에 QCVTP(token pruning) 아키텍처가 설계되어 있으나 핵심 메서드(`retrieve`, `encode_pages_to_patch_tokens`, `assemble_and_generate`)는 `NotImplementedError` — 설계 방향은 잡혔으나 실제 실험은 OCR 라우팅 방식으로 전환.

### 4.2 Chaemin 코드 구조

```
AI-Final-project/
├── benchmark/
│   ├── v12_on_visrag.py       # 핵심 생성 파이프라인 (end-to-end, 800줄)
│   │                          #   image_only / parsed_text_only / parsed_visual 3가지 모드
│   ├── metrics.py             # relaxed_exact_match + ANLS + MRR@10 + Recall@10
│   ├── text_retrieval.py      # BM25 텍스트 검색
│   ├── parse_cache.py         # Upstage 파싱 결과 캐싱
│   └── compare_results.py     # 결과 비교 유틸
├── scripts/
│   ├── parse_visrag_with_upstage.py   # Upstage API 호출 → JSONL 캐시 생성
│   ├── prepare_visrag_datasets.py     # HuggingFace 데이터셋 다운로드
│   ├── evaluate_retrieval.py          # MRR/Recall 평가
│   └── run_domain_free_v12_experiment.sh  # 전체 파이프라인 실행
├── data/parsed/               # Upstage 파싱 캐시 (4개 데이터셋 × first100, ~3.7MB)
└── results/v12_on_visrag/today/  # 12개 결과 파일 (4 datasets × 3 modes)
```

---

## 5. 핵심 차이점 분석

### 5.1 OCR 품질이 결과를 가름

| 항목 | 내 실험 | Chaemin |
|------|---------|---------|
| OCR 도구 | Qwen2-VL-7B (범용 VLM, 무료) | Upstage API (전문 문서 파서, 유료) |
| 표 인식 | 취약 (7B 모델 한계) | 강함 (레이아웃+표 구조화 전용) |
| ChartQA 영향 | -1.84%p ~ -3.17%p (역효과) | +14.28%p (큰 개선) |
| DocVQA 영향 | -0.44%p ~ +1.04%p (소폭) | +63.00%p (압도적 개선) |

**결론**: Upstage의 고품질 파싱이 PV-RAG 성능 향상의 핵심. 내 Qwen self-OCR은 차트에서 오히려 역효과.

### 5.2 Generator 양자화

| 항목 | 내 실험 | Chaemin |
|------|---------|---------|
| 양자화 | 없음 (full precision) | NF4 4-bit (bitsandbytes) |
| VRAM | B200 183GB (충분) | ~5-6GB |
| 속도 | 빠름 | 느림 (CPU offload 가능성) |
| 품질 손실 | 없음 | 소폭 있을 수 있음 |

### 5.3 평가 지표 차이

| 지표 | 특성 | 유리한 경우 |
|------|------|-----------|
| **ANLS** | 편집거리 기반, 연속값, 부분 점수 | 긴 답변, 오타/변형 허용 |
| **Relaxed EM** | 이진값, 숫자 5% tolerance | 짧고 정확한 수치 답변 |

ANLS가 부분 점수를 주기 때문에 내 baseline이 0.78~0.93으로 높고, Chaemin baseline이 0.17~0.37로 낮은 것. **같은 모델, 같은 데이터여도 metric만으로 수치가 크게 달라짐.**

### 5.4 아키텍처 방향성

| 방향 | 내 실험 | Chaemin |
|------|---------|---------|
| 접근 방식 | VisRAG 내부 개선 (token pruning + OCR routing) | 외부 파서 통합 (Upstage → 텍스트 모달 추가) |
| 모달리티 전략 | 이미지 중심, 텍스트 보조 (선택적) | 텍스트 우선, 이미지 검증용 |
| 학습 요소 | QueryConditionedTokenSelector (설계됨) | 없음 (inference only) |
| 확장성 | 모델 내부 최적화 가능 | API 의존, 비용 발생 |

---

## 6. 공통점

- 동일 4개 데이터셋 (InfoVQA, ChartQA, DocVQA/MP-DocVQA, SlideVQA)
- Oracle retrieval (gold pages 사용)
- Qwen2-VL-7B-Instruct를 generator로 활용
- 이미지 + 텍스트 융합 방향 탐색
- VisRAG 논문 (arXiv:2410.10594) 기반 벤치마크

---

## 7. 인사이트 및 교훈

### 내 실험에서 배운 것

1. **Qwen self-OCR은 차트에 역효과**: ChartQA에서 모든 OCR 방법이 baseline 이하 → 차트는 이미지 직접 인식이 우수
2. **선택적 OCR > 일괄 OCR**: DocVQA에서 Selective가 Hybrid 대비 +1.48%p
3. **Generator 교체가 가장 효과적**: OCR 라우팅 ±1%p vs 72B 모델 +5.86%p → 스케일이 지배적
4. **CoT/PoT는 ANLS와 충돌**: 긴 추론 출력이 문자열 유사도 metric에서 치명적 페널티
5. **7B VLM의 구조화 출력 한계**: JSON/코드 생성은 더 큰 모델 필요

### Chaemin 실험에서 배울 것

1. **전문 문서 파서(Upstage)의 위력**: 같은 이미지 + 텍스트 융합이어도 OCR 품질이 결과를 결정
2. **Relaxed EM이 실용적**: 문서 QA의 정확한 수치 평가에 ANLS보다 적합할 수 있음
3. **BM25 텍스트 검색**: `parsed_text_only`가 PV-RAG(parsed_visual)와 거의 동일 → 텍스트 검색만으로도 강력
4. **MRR@10 / Recall@10 추가 평가**: 검색 품질 분리 측정이 분석에 유용

---

## 8. 향후 비교 실험 제안

| 제안 | 기대 효과 | 구현 난이도 |
|------|---------|-----------|
| 동일 metric(ANLS)으로 Chaemin 결과 재평가 | 직접 수치 비교 가능 | 낮음 (chaemin 결과 JSON 재파싱) |
| Upstage 파싱 캐시를 내 파이프라인에 적용 | OCR 품질 격차 제거 후 라우팅 효과만 측정 | 중 (API 키 필요) |
| Chaemin `parsed_visual` + 내 Selective Routing 결합 | Upstage OCR + 선택적 적용 | 중 |
| 72B generator에서 PV-RAG 재실험 | 모델 스케일 × OCR 품질 교차 효과 | 높음 |

---

*세부 실험 결과: `result_001.md`, `result_002.md`, `result_003_004.md`, `results_all.md`*  
*Chaemin 결과: `AI-Final-project/results/ppt_format/today_results_table.md`*

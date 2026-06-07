# Plan 002 — 5-Way Comparison Table (Chaemin 포맷 맞추기)

**Created**: 2026-06-06 KST  
**목표**: Chaemin(AI-Final-project_chaemin)과 동일한 조건(Relaxed EM, NF4 양자화, first100, Oracle retrieval)으로  
5개 방법을 비교하는 결과 테이블 생성

---

## 목표 결과 테이블

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|--------|--------:|--------:|----------:|---------:|------------------:|
| **Baseline** (Image only) | | | | | — |
| **Text — OCR(Open-영혁-Hybrid)** | | | | | |
| **Text — OCR(Closed-VDU)** | | | | | |
| **Text — OCR(Open-영혁-Hybrid) + Image** | | | | | |
| **Selective (LLM)** | | | | | |

---

## 각 방법 정의

| # | 이름 | 입력 | OCR 방식 | 비고 |
|---|------|------|---------|------|
| 1 | Baseline | 이미지만 | 없음 | Chaemin `image_only`와 동일 |
| 2 | Text — OCR(Open-영혁-Hybrid) | 텍스트만 | Qwen2-VL-7B self-OCR (모든 페이지) | 이미지 없음, OCR 텍스트만 generator 입력 |
| 3 | Text — OCR(Closed-VDU) | 텍스트만 | Upstage Document Parse API | Chaemin `parsed_text_only` 결과 재활용 |
| 4 | Text — OCR(Open-영혁-Hybrid) + Image | 텍스트 + 이미지 | Qwen2-VL-7B self-OCR (모든 페이지) | 내 Exp002 Hybrid와 동일 로직, 재실행 |
| 5 | Selective (LLM) | 유형별 선택 | Qwen2-VL-7B self-OCR (mixed만) | LLM 분류기 → chart/text→이미지만, mixed→OCR+이미지 |

---

## 환경 및 설정

| 항목 | 설정값 |
|------|--------|
| **모델** | `Qwen/Qwen2-VL-7B-Instruct` |
| **양자화** | NF4 4-bit (bitsandbytes) — Chaemin 동일 조건 |
| **Metric** | Relaxed Exact Match (숫자 5% tolerance) |
| **샘플 수** | first100 (dataset당 100 query) |
| **Retrieval** | Oracle (gold pages, qrels) |
| **데이터셋** | InfoVQA, ChartQA, MP-DocVQA, SlideVQA |
| **GPU** | CUDA 사용, `device_map="auto"` |
| **max_new_tokens** | 20 (Chaemin 동일) |

---

## 구현 계획

### Step 0: 의존성 설치
```bash
pip install bitsandbytes  # NF4 양자화용 (현재 미설치)
```

### Step 1: 메인 스크립트 작성
`visrag_research/src/run_comparison_v2.py`

Chaemin의 `benchmark/v12_on_visrag.py`를 베이스로,  
아래 5개 `--mode`를 구현:

```
mode=image_only           → Method 1 (Baseline)
mode=ocr_text_only        → Method 2 (Open OCR, 텍스트만)
mode=closed_text_only     → Method 3 (Upstage 캐시 재활용, 인퍼런스 불필요)
mode=ocr_text_image       → Method 4 (Open OCR + 이미지)
mode=selective_llm        → Method 5 (LLM 분류 → 선택적 OCR)
```

**Method 2/4 OCR pass (Qwen self-OCR)**:
```
[페이지 이미지] → Qwen2-VL-7B → "Extract all visible text..." 프롬프트 → OCR 텍스트
```

**Method 5 Selective 로직**:
```
[페이지 이미지] → Qwen2-VL-7B → "Classify: chart / text / mixed" 프롬프트 → 레이블
  chart  → 이미지만 입력 (OCR 스킵)
  text   → 이미지만 입력 (OCR 스킵)
  mixed  → OCR pass → OCR 텍스트 + 이미지 입력
```

**NF4 양자화 로딩**:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

**Relaxed Exact Match**: Chaemin `benchmark/metrics.py`의 `relaxed_exact_match` 함수 그대로 사용

**Method 3 (Closed-VDU)**: Upstage 캐시 경로  
`AI-Final-project_chaemin/data/parsed/{dataset}_first100_qrels_upstage.jsonl`  
인퍼런스 없이 텍스트만 로드 → generator 입력 (이미지 없음)

### Step 2: 실험 실행 스크립트
`visrag_research/src/run_comparison_v2.sh`

```bash
DATASETS="InfoVQA ChartQA MP-DocVQA SlideVQA"
MODES="image_only ocr_text_only closed_text_only ocr_text_image selective_llm"
```

각 조합 (4 datasets × 5 modes = 20 runs) 순차 실행  
결과 저장: `visrag_research/results/comparison_v2/{dataset}_{mode}.json`

### Step 3: 결과 테이블 생성
`visrag_research/src/make_table.py`

20개 JSON을 읽어 위 목표 테이블 형식 출력  
→ `visrag_research/results/comparison_v2/table.md` + `table.json`

---

## 예상 실행 시간

| Method | 인퍼런스 pass 수 | 400 query 예상 시간 |
|--------|----------------|-------------------|
| Baseline (이미지) | 1 pass/page | ~30분 |
| OCR text only | 1 OCR pass/page | ~30분 |
| Closed-VDU | 0 (캐시 재활용) | <1분 |
| OCR text + Image | 1 OCR + 1 gen/page | ~60분 |
| Selective (LLM) | 1 분류 + 0~1 OCR + 1 gen | ~50분 |
| **전체** | | **~3시간** |

---

## 위험 요소

| 위험 | 대응 |
|------|------|
| `bitsandbytes` CUDA 버전 호환 | torch 2.9.1+cu130 → 최신 bitsandbytes 설치 후 테스트 |
| OCR 텍스트 품질 (Qwen self-OCR) | Method 2/4 결과가 Chaemin보다 낮을 수 있음 → 예상된 차이 |
| Method 3 캐시 corpus_id 매핑 | Upstage 캐시의 corpus_id 형식 확인 필요 (현재: `3960.png` 형태) |
| first100 oracle 샘플 범위 | Chaemin qrels와 동일 100 query 사용 확인 필요 |

---

## 사용자 확인 필요 사항

1. **샘플 수**: first100 (Chaemin 동일)으로 고정? or 더 많이?
2. **Selective (LLM) 분류 로직**: chart/text → 이미지만, mixed → OCR+이미지 (Exp003 방식) 유지?
3. **Method 3 (Closed-VDU)**: Chaemin 결과 JSON 재활용 vs. Upstage 캐시로 내 파이프라인 재실행?
4. **GPU 지정**: 특정 GPU 사용 (예: GPU 0만 사용, NF4니까 6GB로 충분)?
5. **실행 순서**: 전체 20 runs 일괄 실행? or 데이터셋별로 확인하며 진행?

---

## 파일 구조 (완료 후)

```
visrag_research/
├── plan_002.md                        # 이 파일
├── src/
│   ├── run_comparison_v2.py           # 메인 실험 스크립트 (5 modes)
│   └── run_comparison_v2.sh           # 실행 스크립트
└── results/
    └── comparison_v2/
        ├── {dataset}_{mode}.json      # 20개 결과 파일
        ├── table.md                   # 최종 비교 테이블
        └── table.json                 # 테이블 JSON
```

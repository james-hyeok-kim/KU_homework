# Experiment 007 — 5-Way Comparison Table (Chaemin 조건 정렬)

**Created**: 2026-06-06 KST  
**Plan**: `plan_002.md`  
**Status**: 🔄 진행 중  
**Result file**: `result_007.md`

---

## 1. 목표

Chaemin(AI-Final-project_chaemin)과 **완전히 동일한 조건**(샘플, 모델, 양자화, metric)에서
5개 방법을 비교하는 결과 테이블 생성. 내 OCR 라우팅 방법(Open self-OCR)과
Chaemin의 Upstage 파싱(Closed-VDU)을 공정하게 직접 비교한다.

### 목표 결과 테이블

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|--------|--------:|--------:|----------:|---------:|------------------:|
| Baseline (Image) | | | | | — |
| Text — OCR(Open-영혁-Hybrid) | | | | | |
| Text — OCR(Closed-VDU) | | | | | |
| Text — OCR(Open-영혁-Hybrid) + Image | | | | | |
| Selective (LLM) | | | | | |

---

## 2. 가설

1. **H1**: Closed-VDU(Upstage)가 Open self-OCR(Qwen)보다 텍스트 품질이 높아
   text-only 비교(Method 2 vs 3)에서 우위일 것 (특히 표/차트 수치).
2. **H2**: OCR + Image 결합(Method 4)이 text-only(Method 2)보다 높을 것
   — 이전 실험에서 text-only는 -15~-30%p 급락했음 (시각 정보 필수).
3. **H3**: Selective(LLM) 라우팅(Method 5)은 chart 페이지에서 이미지 직접 인식,
   text/mixed 페이지에서 OCR 텍스트를 사용하므로 양쪽 강점을 결합할 것.
4. **H4**: Relaxed EM 기준으로 Baseline(image_only)은 Chaemin 측정치
   (0.20/0.37/0.17/0.18)와 유사하게 재현될 것 (파이프라인 정합성 검증).

---

## 3. 방법 정의 (5 methods)

| # | Method | Generator 입력 | OCR 엔진 | 실행 방식 |
|---|--------|---------------|---------|----------|
| 1 | **Baseline (Image)** | 페이지 이미지만 | 없음 | GPU run |
| 2 | **Text — OCR(Open-영혁-Hybrid)** | OCR 텍스트만 (이미지 X) | Qwen2-VL-7B self-OCR (전체 페이지) | GPU run |
| 3 | **Text — OCR(Closed-VDU)** | 파싱 텍스트만 (이미지 X) | Upstage Document Parse API | Chaemin `parsed_text_only` 결과 JSON 재활용 |
| 4 | **Text — OCR(Open-영혁-Hybrid) + Image** | OCR 텍스트 + 페이지 이미지 | Qwen2-VL-7B self-OCR (전체 페이지) | GPU run |
| 5 | **Selective (LLM)** | chart→이미지만 / text·mixed→OCR 텍스트만 | Qwen2-VL-7B (분류 + OCR) | GPU run |

### Method 5 (Selective LLM) 흐름

```
Query → Top-1 page (oracle)
            │
     [LLM 분류 pass]  "chart / text / mixed?"
            │
   chart ──→ [이미지만 generator 입력]
   text  ──→ [OCR pass] → OCR 텍스트만 입력
   mixed ──→ [OCR pass] → OCR 텍스트만 입력
```

---

## 4. 실험 설정

| 항목 | 값 | 비고 |
|------|----|----|
| **Generator** | Qwen2-VL-7B-Instruct | 로컬: `/data/jameskimh/visrag_experiment/models/Qwen2-VL-7B-Instruct` |
| **양자화** | bitsandbytes NF4 4-bit (double quant, fp16 compute) | Chaemin과 동일 |
| **Metric** | Relaxed Exact Match (숫자 5% tolerance) | Chaemin `benchmark/metrics.py` import |
| **샘플** | Chaemin first100 (oracle qrels, topk=1) | Chaemin image_only 결과 JSON에서 qid/query/answer/docids 추출 → 완전 정렬 |
| **샘플 수** | InfoVQA 100 / ChartQA 63 / MP-DocVQA 100 / SlideVQA 100 (총 363) | |
| **이미지 캡** | max_pixels=1280×1024, min_pixels=256×256 | Chaemin과 동일 |
| **max_new_tokens** | 답변 20 / OCR 512 / 분류 8 | 답변은 Chaemin과 동일 |
| **GPU** | GPU 0 단독 (B200 183GB) | `CUDA_VISIBLE_DEVICES=0` |
| **OOM 보호** | VRAM watchdog — GPU 0 메모리 ≥98% 시 프로세스 kill + 사용자 알람 | `vram_watchdog.sh`, 5초 간격 폴링 |
| **데이터** | 로컬 parquet (`/data/jameskimh/visrag_experiment/data/test/`) | HF openbmb/VisRAG-Ret-Test-* 와 동일, qid 순서 일치 검증 완료 |

### OCR / 분류 프롬프트

이전 실험(Exp 002/003, `evaluate_docparse_v2.py`)에서 그대로 포팅:
- **OCR**: "Extract all text content from this document image. Preserve the structure..."
- **분류**: "Classify it into one of these types: chart / text / mixed..."

### 답변 프롬프트

Chaemin `v12_on_visrag.py`의 `build_prompt` 그대로 포팅 (evidence 조합별 분기).

---

## 5. 평가 지표

- **주 지표**: Relaxed Exact Match accuracy (dataset별 + macro 평균 Δ vs Baseline)
- **보조 기록**: per-query elapsed_sec, OCR elapsed_sec, input/output tokens, Method 5 route 분포
- **정합성 검증**: Method 1 결과 vs Chaemin image_only 결과 (H4) — ±2%p 이내면 파이프라인 정합 판정

---

## 6. 실행 계획

```bash
# 전체 실행 (16 GPU runs + Method 3 복사 + 테이블 생성)
bash visrag_research/src/run_comparison_v2.sh
```

| 단계 | 내용 | 예상 시간 |
|------|------|---------|
| 0 | smoke test (ChartQA image_only --limit 2) | ~5분 (모델 로드 포함) |
| 1 | Method 3 복사 (GPU 불필요) | <1분 |
| 2 | image_only × 4 datasets | ~30분 |
| 3 | ocr_text_only × 4 (OCR 캐시 생성) | ~60분 |
| 4 | ocr_text_image × 4 (OCR 캐시 재활용) | ~30분 |
| 5 | selective_llm × 4 (분류 + 캐시 재활용) | ~40분 |
| 6 | 테이블 생성 (`make_table_v2.py`) | <1분 |
| **계** | | **~2.5~3시간** |

---

## 7. 위험 요소 및 대응

| 위험 | 대응 |
|------|------|
| OOM | NF4 ~6GB vs B200 183GB → 사실상 불가능. watchdog 98% kill + push 알람 |
| bitsandbytes B200(sm_100) 호환 | v0.49.2 NF4 forward smoke test 통과 확인 완료 |
| NF4 양자화로 OCR 품질 저하 | Method 2/4/5 모두 동일 조건이므로 내부 비교는 공정. Chaemin 정렬 우선 |
| Baseline 재현 불일치 (H4 실패) | transformers 버전 차이 가능 → result_007.md에 양쪽 수치 병기 |
| ChartQA 63개 (100 미만) | Chaemin과 동일 현상 (qrels 있는 query가 63개) — 그대로 사용 |

---

## 8. 산출물

```
visrag_research/results/comparison_v2/
├── {dataset}_{mode}.json        # 20개 결과 (16 GPU run + 4 Method3 복사)
├── cache/ocr_{dataset}.jsonl    # OCR 텍스트 캐시 (corpus_id 키)
├── cache/route_{dataset}.jsonl  # 페이지 분류 캐시
├── logs/{dataset}_{mode}.log    # run별 로그
├── table.md                     # 최종 5-way 비교 테이블
└── table.json
```

**코드**: `src/run_comparison_v2.py`, `src/run_comparison_v2.sh`, `src/make_table_v2.py`, `src/vram_watchdog.sh`

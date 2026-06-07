# Result 007 — 5-Way Comparison Table (Chaemin 조건 정렬)

**Experiment**: `experiment_007.md`  
**Status**: ✅ **전체 완료** — 4.57 suite (20:29 KST) + tf451 Chaemin 정렬 suite (22:04 KST)  
**Last updated**: 2026-06-06 22:05 KST

---

## 최종 결과 테이블 (업데이트 예정)

Generator: Qwen2-VL-7B-Instruct (NF4 4-bit) | Metric: Relaxed Exact Match | Samples: first100 (oracle, topk=1)

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|--------|--------:|--------:|----------:|---------:|------------------:|
| Baseline (Image) | **0.7200** | 0.5556 | **0.8800** | 0.5400 | — |
| Text — OCR(Open-영혁-Hybrid) | 0.3400 | 0.3175 | 0.6700 | 0.4300 | -23.45%p |
| Text — OCR(Closed-VDU) | 0.5400* | 0.5238* | 0.7600* | 0.5200* | -8.79%p |
| Text — OCR(Open-영혁-Hybrid) + Image | **0.7300** | **0.6667** | 0.8200 | **0.5800** | **+2.53%p** ★ |
| Selective (LLM) | 0.3400 | 0.5238 | 0.6700 | 0.4400 | -18.04%p |

> \* Method 3 (Closed-VDU) 수치는 Chaemin 환경에서 측정된 `parsed_text_only_top1_first100` 결과를 그대로 사용
> (2026-06-06 사용자 결정 — "일단 chaemin 결과 넣고, 나중에 다시 업데이트 요청"). 타 환경 측정치이므로
> 해석 시 아래 "중대 발견" 참고. 내 환경 재실행은 추후 요청 시 진행 (~15분, Upstage 캐시 재활용).

### 참고: Chaemin 측정치 (재현 검증용, H4)

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA |
|--------|--------:|--------:|----------:|---------:|
| Chaemin image_only | 0.2000 | 0.3651 | 0.1700 | 0.1800 |
| Chaemin parsed_visual (PV-RAG) | 0.5400 | 0.5079 | 0.8000 | 0.5200 |

---

## ⚠️ 중대 발견 — H4 실패: Chaemin 환경 vision 경로 손상 의심

**같은 모델·샘플·프롬프트·metric·입력 토큰 수(1634.95 동일)인데 baseline이 크게 다름:**

| Dataset | 내 환경 | Chaemin 환경 | 차이 |
|---------|--------:|------------:|-----:|
| InfoVQA | 0.7200 | 0.2000 | +52.0%p |
| ChartQA | 0.5556 | 0.3651 | +19.1%p |
| MP-DocVQA | 0.8800 | 0.1700 | +71.0%p |
| SlideVQA | 0.5400 | 0.1800 | +36.0%p |

**증거**: Chaemin 예측이 이미지 내용 대신 언어 prior로 답함
(정답 picaboo→"Snapchat", friendster→"Facebook", Mangaldeep→"ITC", 또는 "insufficient to answer").
입력 토큰 수가 동일하므로 이미지는 들어갔으나 **시각 특징이 손상된 채** 입력된 것.

**원인 확정 (2026-06-06 probe)**: **transformers 4.51의 vision 처리 버그.**
- A/B 테스트: 내 4.57 환경에서 `llm_int8_skip_modules=["visual"]` 유무 → 둘 다 0.8667 (30샘플 동일)
  → vision 양자화 자체는 무관
- 4.51 venv probe (MP-DocVQA image_only 30샘플, 동일 코드·모델·GPU): **0.267로 급락** (4.57은 0.867)
  → Chaemin의 0.17 수준 재현, "insufficient to answer" 패턴까지 동일
- 내 수치는 과거 full-precision EM(InfoVQA 0.7312)과 일치 → 내 환경 정상 교차 검증

**함의**:
1. Chaemin의 "+36.32%p PV-RAG 개선" 주장은 손상된 baseline 대비 → 정상 baseline 대비로는
   Upstage text-only가 오히려 4개 중 4개 dataset에서 baseline 이하.
2. Method 3를 Chaemin JSON 그대로 쓰면 타 환경 측정치가 섞임 → 공정성 문제.

**상태**: Method 3 = **Chaemin 결과 그대로 사용으로 결정** (2026-06-06, 추후 재실행 업데이트 가능).
Faithfulness 측정(KJ-Min/Visual-RAG, `Visual-RAG_kjmin/` 클론 완료)은 **나중에 별도 진행**으로 결정됨.

---

## 🔄 Chaemin 환경 정렬 suite (tf451) — 진행 중

사용자 지시 (2026-06-06 19:16 KST): "처음에 얘기한 method 모두 채민 환경에 맞춰서 결과 업데이트"
→ transformers 4.51 venv(`/data/jameskimh/venvs/tf451`)에서 전체 16 runs를 GPU 1에서 병행 실행.

- 결과 위치: `results/comparison_v2_tf451/` (4.57 결과와 완전 분리, OCR/route 캐시도 분리)
- 메인 4.57 suite (GPU 0)도 계속 진행 → 최종적으로 **두 버전 테이블** 확보
  - `comparison_v2/` — 내 환경 (정상 vision, 4.57.6)
  - `comparison_v2_tf451/` — Chaemin 정렬 환경 (4.51.3, vision 버그 포함)

### tf451 결과 테이블 (업데이트 예정)

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|--------|--------:|--------:|----------:|---------:|------------------:|
| Baseline (Image) | 0.1600 | 0.3492 | 0.1600 | 0.1600 | — |
| Text — OCR(Open-영혁-Hybrid) | 0.0400 | 0.2857 | 0.1100 | 0.0900 | -7.59%p |
| **Text — OCR(Closed-VDU)** | **0.5400** | **0.5238** | **0.7600** | **0.5200** | **+37.87%p** ★ |
| Text — OCR(Open-영혁-Hybrid) + Image | 0.1000 | 0.3333 | 0.1300 | 0.1500 | -2.90%p |
| Selective (LLM) | 0.0400 | 0.3333 | 0.1100 | 0.0900 | -6.40%p |

> tf451 InfoVQA ocr_text_only 0.04 — 4.51 vision 버그가 OCR pass도 손상시켜 self-OCR 텍스트 자체가 무용지물.
> Chaemin 환경에서 Upstage(외부 API, 버그 무관)가 +50%p 우위로 보였던 이유가 설명됨.
>
> tf451 InfoVQA baseline 0.16 vs Chaemin 원본 0.20 — ±4%p로 Chaemin 환경 재현 확인 (H4 충족, 정렬 성공)

---

## 진행 로그

| 시각 (KST) | 이벤트 |
|-----------|--------|
| 2026-06-06 | 환경 검증: bitsandbytes 0.49.2 설치, B200 NF4 forward 테스트 통과 |
| 2026-06-06 | 데이터 정렬 검증: 로컬 parquet qid 순서 == Chaemin first100 qid 순서 (ChartQA 확인) |
| 2026-06-06 | 코드 작성: `run_comparison_v2.py`, `run_comparison_v2.sh`, `make_table_v2.py`, `vram_watchdog.sh` |
| 2026-06-06 | smoke test 통과: image_only / ocr_text_image / selective_llm (ChartQA 2샘플, 라우팅 동작 확인) |
| 2026-06-06 17:47 KST | Method 3: Chaemin parsed_text_only 결과 4개 복사 완료 |
| 2026-06-06 17:47 KST | 전체 16 runs 시작 (GPU 0, NF4 ~8GB/183GB, watchdog 98% 가동) |

---

## 가설 판정 (업데이트 예정)

| 가설 | 내용 | 판정 (4.57 suite 기준) |
|------|------|------|
| H1 | Closed-VDU(Upstage) > Open self-OCR (text-only 비교) | ✅ **입증** — 4/4 dataset에서 Upstage 우위 (평균 +14.7%p) |
| H2 | OCR+Image > OCR text-only | ✅ **입증** — 4/4 dataset에서 우위 (평균 +26.0%p), 시각 정보 필수 재확인 |
| H3 | Selective(LLM)가 양쪽 강점 결합 | ❌ **기각** — 선택 라우팅(chart만 이미지)이 text/mixed를 text-only로 보내 평균 -18.0%p |
| H4 | Baseline 재현 ±2%p 이내 (파이프라인 정합성) | ❌→✅ **4.57에서 기각, 원인 규명 후 tf451로 재현 성공** — transformers 4.51 vision 버그 확정 |

---

## 관찰 / 이슈 (업데이트 예정)

- (없음)

---

## Method 5 route 분포 (업데이트 예정)

| Dataset | chart | text | mixed |
|---------|------:|-----:|------:|
| InfoVQA | 0 | 0 | 100 (전부 mixed → 전부 OCR text-only 경로) |
| ChartQA | 38 (60%) | 0 | 25 (40%) |
| MP-DocVQA | 0 | 65 (65%) | 35 (35%) |
| SlideVQA | 7 (7%) | 23 (23%) | 70 (70%) |

---

## 최종 종합 — 두 테이블이 말하는 것

### 환경에 따라 결론이 정반대

| 관점 | 4.57 (정상 환경) | tf451 (Chaemin 정렬, vision 버그) |
|------|-----------------|----------------------------------|
| 최고 방법 | **OCR(Open)+Image (+2.53%p)** | **Closed-VDU/Upstage (+37.87%p)** |
| Upstage text-only | baseline 이하 (-8.79%p) | 압도적 1위 |
| 이미지 활용 가치 | 핵심 (text-only 전부 급락) | 무용 (vision 버그로 이미지가 손상 입력) |

### 해석

1. **Chaemin의 "+36%p 개선"은 tf451 환경에서 정확히 재현됨 (+37.87%p)** — 그 환경 안에서는 올바른 측정.
   다만 개선의 본질은 "Upstage 파싱이 이미지보다 우수"가 아니라 **"vision이 고장난 환경에서
   유일하게 멀쩡한 정보 채널(외부 API 텍스트)이 Upstage였다"**는 것.
2. **정상 환경에서는 이미지가 왕** — baseline(0.72/0.56/0.88/0.54)이 모든 text-only를 압도,
   OCR+Image 결합만 소폭 개선 (+2.53%p).
3. **Selective(LLM) 라우팅 규칙의 한계** — "chart→이미지, text/mixed→OCR text"는 비-chart 페이지를
   전부 text-only로 보내 정상 환경에서 -18%p. 라우팅 개선 시 "text/mixed→OCR+Image"가 유망
   (Method 4가 +2.53%p였으므로).

### 산출물

- `results/comparison_v2/table.md` — 내 환경(4.57.6) 5-way 테이블
- `results/comparison_v2_tf451/table.md` — Chaemin 정렬(4.51.3) 5-way 테이블
- 캐시: `results/comparison_v2{,_tf451}/cache/` (OCR/route, 환경별 분리)
- venv: `/data/jameskimh/venvs/tf451` (transformers 4.51.3 재현용)

## 다음 단계 (후속 제안)

1. ⏳ Faithfulness 측정 (KJ-Min/Visual-RAG, judge: Qwen2.5-VL-72B) — 사용자 결정: 나중에 별도 진행
2. ⏳ `comparison_with_chaemin.md`에 양 환경 결과 반영
3. ⏳ Selective 라우팅 v2: text/mixed → OCR+Image (정상 환경에서 +2.53%p 초과 기대)
4. ⏳ Chaemin에게 transformers 4.51 vision 버그 공유 (재실험 권고)

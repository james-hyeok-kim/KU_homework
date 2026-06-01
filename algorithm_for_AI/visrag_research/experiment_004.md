# Experiment 004: Chart-Aware OCR Prompt Routing (Experiment C, redesigned)

## 설계 변경 경위

원래 설계: rednote-hilab/dots.mocr (3B, chart→SVG)를 차트 페이지 OCR에 활용

**실측 문제**: dots.mocr가 853초/샘플 → ChartQA 63샘플에 14.9시간 예상 (비실용적)
- 원인: max_pixels=11.3M 설정으로 840×788 차트 이미지가 3360 패치 생성
- 3360 visual token에 대한 attention이 O(n²) 연산 → 생성당 14분 소요
- Flash Attention 2.8.3 설치되어 있어도 patches 수가 과다

**재설계**: dots.mocr 대신 동일한 Qwen2-VL을 차트 특화 프롬프트로 사용
- 모델 추가 로딩 없음, 기존 캐시 재활용 구조 유지
- 과학적 가설 변환: "모델 특화" → "프롬프트 특화" 효과 검증

---

## 가설

차트 페이지에 차트 특화 OCR 프롬프트(수치·축·범례 추출 명시)를 적용하면,
일반 OCR 프롬프트보다 정확한 수치 정보를 추출하여 ChartQA 성능이 향상될 것이다.

- `chart` 페이지: Qwen2-VL + 차트 특화 프롬프트
- `text` 페이지: OCR 건너뜀 (VLM이 직접 읽음)
- `mixed` 페이지: Qwen2-VL + 일반 OCR 프롬프트

vs Experiment D (selective_routing):
- chart → **skip OCR** (D)  vs  chart → **차트 특화 OCR** (C)
- 둘의 차이가 ChartQA 성능 차이를 설명

## 프롬프트 비교

| 상황 | Exp002 (DocParse Hybrid) | Exp003 (Selective) | **Exp004 (Chart-Aware)** |
|------|------------------------|-------------------|----------------------|
| chart 페이지 | 일반 OCR 텍스트 | OCR 없음 | 차트 특화 OCR |
| text 페이지 | 일반 OCR 텍스트 | OCR 없음 | OCR 없음 |
| mixed 페이지 | 일반 OCR 텍스트 | 일반 OCR 텍스트 | 일반 OCR 텍스트 |

### 차트 특화 OCR 프롬프트 (영어)

```
This is a chart or graph. List ALL data precisely:
1. Chart title
2. X-axis label and all tick values
3. Y-axis label and all tick values
4. Legend entries
5. All data series values (bar heights, line points, pie slices, etc.)
Be exact with numbers. Output only this structured list — no commentary.
```

## 설정

| 항목 | 값 |
|------|-----|
| OCR 모델 | Qwen2-VL-7B (chart + mixed 페이지 모두) |
| GPU | NVIDIA B200 GPU 1 (183GB) |
| 프롬프트 (chart) | 차트 특화 (5-step list) |
| 프롬프트 (mixed) | 일반 OCR |
| OCR max_new_tokens | 256 |
| 차트 OCR 캐시 | `/data/jameskimh/visrag_experiment/ocr_cache/{dataset}_chart/` |
| 라우팅 캐시 | `/data/jameskimh/visrag_experiment/route_cache/` (Exp003과 공유) |
| 실행일 | 2026-05-28 KST |

## 실측 속도

| 항목 | 수치 |
|------|------|
| Sanity (5 ChartQA) avg | 8.9s/샘플 |
| 예상 전체 (1928샘플) | ~4.8시간 |

## 예상 결과

| Dataset | Baseline | Exp002 Hybrid | Exp003 Selective | **Exp004 Chart-Aware** |
|---------|---------|--------------|-----------------|----------------------|
| ChartQA | 0.6941 | 0.6757 | ≈0.6941 (회복 기대) | ≥0.6941 (차트 OCR로 추가 향상?) |
| InfoVQA | 0.7835 | 0.7954 | ≈0.7954 | ≈0.7954 (mixed 동일 처리) |
| DocVQA | 0.9303 | 0.9259 | ≈0.9303 | ≈0.9303 |
| SlideVQA | 0.7458 | 0.7472 | ≈0.7472 | ≈0.7472 |

## 위험 요소

- "mixed"로 분류된 차트 페이지가 일반 OCR 받아 Exp002 손실 일부 계승
- 차트 특화 프롬프트가 숫자를 더 잘 추출하지만 VLM이 이미 이미지에서 직접 읽는 것이 더 정확할 수도 있음
- Exp003과 Exp004의 chart 처리 차이(skip vs OCR)가 ANLS에 유의미하게 반영되지 않을 수도 있음

## 결론적 측정 목표

- 핵심 비교: Exp003 ChartQA vs Exp004 ChartQA → 차트 특화 OCR의 순수 기여
- 부차 비교: Exp002 ChartQA vs Exp004 ChartQA → 일반 OCR vs 차트 특화 OCR

## 코드 위치

- 평가 스크립트: `/data/jameskimh/visrag_experiment/src/evaluate_docparse_v2.py`
  - method: `chart_aware_routing`
- 실행 스크립트: `/data/jameskimh/visrag_experiment/run_docparse_v2.sh`

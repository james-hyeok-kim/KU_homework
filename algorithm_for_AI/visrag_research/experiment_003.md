# Experiment 003: Selective OCR Routing (Experiment D)

## 가설

페이지 유형(차트 / 텍스트 / 혼합)을 자동 분류하여 OCR을 선택적으로 적용하면,
DocParse Hybrid (일괄 OCR) 대비 ChartQA·DocVQA 하락을 막으면서 InfoVQA 개선을 유지할 수 있다.

- `chart` 페이지: OCR 건너뜀 → 시각 모델이 직접 읽음 (ChartQA -1.84%p 방지)
- `text` 페이지: OCR 건너뜀 → VLM이 이미 잘 읽음 (DocVQA -0.44%p 방지)
- `mixed` 페이지: Qwen2-VL OCR 적용 → InfoVQA +1.19%p 효과 보존

## 배경

- Experiment 002 결과: DocParse Hybrid가 InfoVQA에서 +1.19%p이지만 ChartQA에서 -1.84%p
- 문제: OCR이 모든 페이지에 일괄 적용되어 차트 수치가 잘못 추출되는 경우 성능 하락
- 해결: Qwen2-VL 자체를 분류기로 활용해 페이지 유형 판단 후 선택적 OCR

## 설정

| 항목 | 값 |
|------|-----|
| 모델 | VisRAG-Ret (검색) + Qwen2-VL-7B (분류 + OCR + 생성) |
| GPU | NVIDIA B200 GPU 1 (183GB) |
| OCR max_new_tokens | 256 |
| 분류 max_new_tokens | 10 |
| 라우팅 캐시 | `/data/jameskimh/visrag_experiment/route_cache/` |
| Top-K 페이지 | 5 |
| 실행일 | 2026-05-28 KST |

## 분류 기준

| 레이블 | 설명 | OCR 적용 |
|--------|------|---------|
| `chart` | 주로 차트·그래프·데이터 시각화 | ✗ 건너뜀 |
| `text` | 주로 읽기 가능한 일반 텍스트 | ✗ 건너뜀 |
| `mixed` | 텍스트 + 시각 요소 혼합 (인포그래픽) | ✓ OCR 적용 |

## 비교 메서드

| 메서드 | 설명 |
|--------|------|
| `baseline` | 순수 VisRAG (이미지만) |
| `docparse` | Hybrid: 이미지 + Qwen2-VL OCR 전 페이지 (Exp 002) |
| `selective_routing` | 이미지 + mixed 페이지만 OCR (이번 실험) |

## 데이터셋

| Dataset | 샘플 수 | 기대 효과 |
|---------|---------|---------|
| InfoVQA | 718 | mixed 분류 → OCR 적용 → +1%p 유지 |
| ChartQA | 63 | chart 분류 → OCR 건너뜀 → 회복 |
| DocVQA | 591 | text 분류 → OCR 건너뜀 → 회복 |
| SlideVQA | 556 | 혼합 → mixed가 많을 경우 +0.14%p 유지 |

## 평가 메트릭

- ANLS (주요), Exact Match, Relaxed Accuracy

## 코드 위치

- 평가 스크립트: `/data/jameskimh/visrag_experiment/src/evaluate_docparse_v2.py`
- 실행 스크립트: `/data/jameskimh/visrag_experiment/run_docparse_v2.sh`

## 예상 결과

| Dataset | Baseline | Exp002 Hybrid | 예상 Selective |
|---------|---------|--------------|---------------|
| ChartQA | 0.6941 | 0.6757 (-1.84%) | ≥ 0.6941 (회복) |
| InfoVQA | 0.7835 | 0.7954 (+1.19%) | ≈ 0.7954 (유지) |
| DocVQA | 0.9303 | 0.9259 (-0.44%) | ≈ 0.9303 (회복) |
| SlideVQA | 0.7458 | 0.7472 (+0.14%) | ≈ 0.7472 (유지) |

## 위험 요소

- 분류기(Qwen2-VL) 오분류 시 효과 상쇄 (chart → mixed 오판)
- 분류 추가로 샘플당 처리 시간 증가 (10 token 추론 × 5페이지)
- ChartQA 63샘플로 통계적 유의성 낮음

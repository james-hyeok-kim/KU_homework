# Experiment 002: DocParse + VisRAG Hybrid

## 가설

VisRAG 생성 단계에서 이미지와 함께 OCR로 추출한 구조화 텍스트를 제공하면 (Hybrid 접근), 순수 시각 RAG (baseline VisRAG)보다 문서 VQA 성능이 향상될 것이다. 특히 텍스트와 시각 요소가 혼재된 문서 유형(InfoVQA, SlideVQA)에서 효과가 클 것이다.

## 배경

- Upstage Document Parse 개념에서 영감: OCR + 레이아웃 파싱 결과를 RAG에 활용
- API 키 없어 동등한 오픈소스 구현: Qwen2-VL 7B를 OCR 엔진으로 활용
- 동료 실험(Upstage Document Parse API 사용)과의 비교를 위한 사전 실험

## 설정

| 항목 | 값 |
|------|-----|
| 모델 | VisRAG-Ret (retrieval) + Qwen2-VL-7B (OCR + generation) |
| GPU | NVIDIA B200 GPU 1 (183GB) |
| OCR max_new_tokens | 256 |
| Top-K 페이지 | 5 |
| OCR 캐시 | `/data/jameskimh/visrag_experiment/ocr_cache/` |
| 실행일 | 2026-05-27 ~ 2026-05-28 KST |

## 비교 메서드

| 메서드 | 설명 |
|--------|------|
| `baseline` | 순수 VisRAG: 이미지만 생성기에 전달 |
| `qcvtp` | QCVTP: 쿼리 기반 토큰 프루닝 (Experiment 001) |
| `docparse` | Hybrid: 이미지 + Qwen2-VL OCR 텍스트 함께 전달 |
| `docparse_text` | Text-only: OCR 텍스트만 전달 (이미지 없음) |

## 데이터셋

| Dataset | 샘플 수 | 특성 |
|---------|---------|------|
| SlideVQA | 556 | 멀티페이지 슬라이드 |
| DocVQA | 591 | 단일페이지 텍스트 문서 |
| InfoVQA | 718 | 인포그래픽 (텍스트+시각 혼합) |
| ChartQA | 63 | 차트/그래프 |

## 평가 메트릭

- ANLS (Average Normalized Levenshtein Similarity) — 주요 지표
- Exact Match (EM)
- Relaxed Accuracy (±5% 허용)

## 코드 위치

- 평가 스크립트: `/data/jameskimh/visrag_experiment/src/evaluate_docparse.py`
- 실행 스크립트: `/data/jameskimh/visrag_experiment/run_docparse.sh`
- 결과: `/data/jameskimh/visrag_experiment/results/docparse_*.jsonl`

## 예상 결과

- `docparse` (hybrid)가 InfoVQA, SlideVQA에서 baseline 대비 +2~5%p 개선
- `docparse_text`는 VisRAG 핵심 강점(시각 정보)을 제거하므로 하락 예상
- 텍스트만 있는 단순 문서(DocVQA)에서는 효과 미미 또는 오히려 노이즈

## 위험 요소

- Qwen2-VL을 OCR과 생성 양쪽에 쓰는 구조 → OCR 텍스트가 생성 시 중복 정보가 될 수 있음
- OCR 오류가 있을 경우 생성 성능 저하 가능
- Upstage Document Parse 대비 OCR 정밀도 차이 존재

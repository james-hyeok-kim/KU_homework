# Result 002: DocParse + VisRAG Hybrid

## 실험 요약

Qwen2-VL 7B를 OCR 엔진으로 사용하여 페이지 이미지에서 텍스트를 추출하고,
이를 VisRAG 생성 단계에 함께 제공하는 Hybrid 접근법의 효과를 검증.
→ 문서 유형에 따라 효과가 상이함을 확인.

**실행 환경**: GPU 1 (NVIDIA B200, 183GB) | 2026-05-27 20:04 ~ 2026-05-28 KST

---

## 최종 결과 (ANLS)

| Dataset | Baseline | QCVTP | DocParse Hybrid | DocParse Text | Hybrid Δ |
|---------|---------|-------|-----------------|---------------|----------|
| SlideVQA | 0.7458 | 0.6943 | **0.7472** | 0.5922 | **+0.14%p** |
| DocVQA | 0.9303 | 0.9303 | 0.9259 | 0.6729 | -0.44%p |
| InfoVQA | 0.7835 | 0.7835 | **0.7954** | 0.4993 | **+1.19%p ★** |
| ChartQA | 0.6941 | 0.6941 | 0.6757 | 0.3714 | -1.84%p |

### Exact Match

| Dataset | Baseline | DocParse Hybrid | DocParse Text |
|---------|---------|-----------------|---------------|
| SlideVQA | 0.5791 | 0.5845 | 0.4550 |
| DocVQA | 0.8951 | 0.8934 | 0.6176 |
| InfoVQA | 0.7312 | 0.7382 | 0.4220 |
| ChartQA | 0.6667 | 0.6190 | 0.2381 |

---

## 분석

### 1. DocParse Hybrid (이미지 + OCR 텍스트)

**효과적인 경우:**
- **InfoVQA (+1.19%p)**: 인포그래픽 문서는 텍스트와 시각 요소가 복잡하게 혼재.
  OCR 텍스트가 모델이 시각적으로 놓칠 수 있는 작은 텍스트 요소를 보완.
- **SlideVQA (+0.14%p)**: 멀티페이지 슬라이드에서 소폭 개선.
  슬라이드 제목·본문 텍스트 추출이 정확한 페이지 식별을 돕는 효과.

**효과 없거나 해로운 경우:**
- **DocVQA (-0.44%p)**: 텍스트가 이미지에 선명하게 존재 → OCR 텍스트가 중복/노이즈.
  Qwen2-VL은 이미 이미지에서 직접 텍스트를 읽을 수 있어 추가 정보가 불필요.
- **ChartQA (-1.84%p)**: 차트 수치는 OCR로 정확히 추출하기 어려움.
  잘못된 OCR이 오히려 혼란을 유발.

### 2. DocParse Text-only (OCR 텍스트만)

전 데이터셋에서 큰 폭 하락 (-15%p ~ -30%p).
VisRAG의 핵심 강점은 시각 정보 보존에 있으며, 이를 제거하면 성능이 급락.
→ OCR 텍스트 단독 사용은 VisRAG 대체재로 부적합.

### 3. QCVTP와의 비교

| | DocVQA | InfoVQA | ChartQA | SlideVQA |
|--|--------|---------|---------|---------|
| QCVTP vs Baseline | 0 | 0 | 0 | **-5.1%p** |
| DocParse Hybrid vs Baseline | -0.44%p | **+1.19%p** | -1.84%p | +0.14%p |

- QCVTP: 단일 페이지 데이터셋에서 효과 없음 (K=1 bypass), SlideVQA에서 성능 저하
- DocParse Hybrid: 인포그래픽에서 유의미한 개선, 그 외 혼재

---

## OCR 속도

| Dataset | avg OCR 시간/샘플 | docparse_text (캐시) |
|---------|----------------|---------------------|
| SlideVQA | 17.0초 | 0.003초 |
| DocVQA | 9.9초 | 0.001초 |
| InfoVQA | 8.7초 | 0.001초 |
| ChartQA | 2.9초 | 0.001초 |

OCR 캐시 덕분에 `docparse_text`는 사실상 무료로 실행됨.

---

## 결론

1. **DocParse Hybrid는 문서 유형 선택적으로 효과적**: 인포그래픽·복합 문서(InfoVQA)에서 +1.19%p 개선.
2. **텍스트 선명 문서(DocVQA)나 차트(ChartQA)에서는 오히려 역효과**: VLM이 이미지에서 직접 읽는 것이 더 정확.
3. **Text-only는 VisRAG 대체 불가**: 시각 정보 제거 시 성능 급락.
4. **Upstage Document Parse API 사용 시**: 더 정밀한 OCR로 특히 ChartQA, DocVQA 결과가 개선될 가능성 있음.

### 제안하는 후속 실험

- **문서 유형 분류기 추가**: 자동으로 hybrid/visual-only 경로를 선택하는 routing 모듈
- **Upstage Document Parse API**: API 키 확보 후 동일 실험 재실행 (정밀 파싱 효과 검증)
- **OCR 선택성**: 모든 페이지가 아닌 낮은 시각 품질 페이지에만 OCR 적용

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-28 KST | result_002.md v1.0 — 실험 완료, 전체 결과 기록 |

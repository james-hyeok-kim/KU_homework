# Novelty Validation Log

## Round 1 — Query-Conditioned Visual Token Pruning for Multi-Page Context Compression

**판정**: ACCEPT
**판정 근거**: 핵심 contribution이 90% 이상 겹치는 단일 published 논문은 없음. 유사 연구(QG-VTC, FlashVLM, AVIR 등)는 각각 일부 요소만 공유하며, VisRAG retriever + 멀티페이지 context compression + dynamic token budget의 결합은 기존 논문에서 동일하게 다루어진 사례를 발견하지 못함.

---

### 검색 쿼리 목록

1. `visual token pruning RAG multimodal document retrieval augmented generation`
2. `query-conditioned token compression VLM document understanding visual patch selection`
3. `patch token selection visual question answering cross-attention token pruning VLM`
4. `multi-page visual context compression VQA dynamic token budget retrieval augmented generation`
5. `dynamic token budget vision language model retrieval document multi-page`
6. `visual patch pruning cross-attention document understanding SlideVQA DocVQA token selection`
7. `QG-VTC question-guided visual token compression RAG retrieval augmented generation multipage`
8. `AVIR adaptive visual in-document retrieval multi-page document question answering token selection`

---

### 유사 논문 목록

| 논문 | 연도 | 유사점 | 차이점 |
|------|------|--------|--------|
| QG-VTC (arxiv 2504.00654) | 2025 | query-guided 시각 토큰 압축, 단일 페이지 VQA | RAG context compression 없음, 단일 이미지 대상, VisRAG 구조와 통합 없음, 동적 budget 없음 |
| FlashVLM (arxiv 2512.20561) | 2024 | text-guided visual token selection, LLM 입력 직전 단계 | RAG/멀티페이지 문서 맥락 없음, 단일 이미지 대상, retrieval pipeline과 통합 없음 |
| AVIR (arxiv 2601.11976) | 2026 | 멀티페이지 문서 QA, 관련 페이지 선택 | page-level 선택 (patch-level 아님), query-conditioned patch pruning 없음, token budget 동적 조정 없음 |
| RegionRAG (arxiv 2510.27261) | 2025 | RAG + 시각 문서에서 region-level 검색 | region-level 검색 단위 변경이 핵심, patch token pruning 후 multi-page assembly 구조 아님 |
| VimRAG (arxiv 2602.12735) | 2026 | 대규모 시각 context에서 multimodal memory 기반 RAG | memory graph 구조 기반, query-conditioned cross-attention selector 방식 아님 |
| MI-Pruner (arxiv 2604.03072) | 2026 | crossmodal mutual information 기반 시각 토큰 pruning | 단일 이미지 대상, RAG pipeline 통합 없음, dynamic budget 없음 |
| AdaptInfer (arxiv 2508.06084) | 2025 | 동적 텍스트 guidance 기반 시각 토큰 pruning | 단일 이미지 추론 최적화가 목적, retrieval-augmented 멀티페이지 압축 아님 |
| Index-Preserving Token Pruning (arxiv 2509.06415) | 2025 | 문서 이해에서 patch-level token pruning | 문서 내 텍스트/배경 이진 분류 방식, query-conditioned cross-attention scoring 아님 |

---

### 결론

**판정: ACCEPT — novel한 아이디어로 판정**

제안 아이디어의 핵심 contribution은 세 가지 요소의 특정 결합이다:

1. **VisRAG retrieval pipeline에 통합된** query-conditioned patch-level token pruning
2. **멀티페이지 문서에 걸친** dynamic token budget (K 페이지 수에 따라 r이 결정)
3. **경량 cross-attention selector 모듈**을 retriever-generator 사이에 삽입하는 구조

개별 요소별로는:
- query-guided token compression: QG-VTC, FlashVLM 등에서 연구됨 (단, 단일 이미지, RAG 구조 없음)
- 멀티페이지 page-level 선택: AVIR에서 다루어짐 (단, patch-level 아님)
- RAG에서의 시각 token 압축: RegionRAG, VimRAG 등이 연관 (단, 방법론이 다름)

**세 가지 요소를 동시에 조합하여 VisRAG에 적용하는 논문은 발견되지 않았다.** 특히 retriever가 반환한 멀티페이지 결과에서 patch-level로 query-conditioned pruning을 수행하고, 남은 token budget을 K에 따라 동적으로 배분하는 구체적인 파이프라인은 기존 논문에 존재하지 않는 것으로 판단된다.

검색 날짜: 2026-05-26 KST

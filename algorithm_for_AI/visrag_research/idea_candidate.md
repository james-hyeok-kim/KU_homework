# VisRAG 개선 아이디어: Query-Conditioned Visual Token Pruning for Multi-Page Context Compression

## 핵심 아이디어
검색된 여러 페이지 이미지를 VLM generator에 그대로 넣는 대신, 쿼리와의 관련성에 따라 각 페이지의 visual patch token을 선택적으로 pruning하여 context window를 효율적으로 활용한다. 이를 통해 동일한 context 예산(token budget) 내에서 더 많은 페이지를 포함시키고, 동시에 노이즈 patch를 제거하여 generator의 답변 품질을 향상시킨다. Retriever와 Generator 사이에 경량 cross-attention 기반 token selector 모듈을 삽입하는 방식으로 기존 VisRAG 구조를 최소한으로 변경한다.

## 해결하는 VisRAG 한계점
- **한계 #2 (멀티페이지 이해)**: 동일 token budget 내에 더 많은 페이지를 포함시켜 cross-page 정보를 통합
- **한계 #6 (Re-ranking 없음)**: 페이지 내 patch 수준에서 query-relevant 영역을 강조하는 효과
- **한계 #7 (Context window 제한)**: patch pruning으로 per-page token 수를 줄여 동일 window에서 더 많은 페이지 처리 가능

## 알고리즘 설명

### 전체 파이프라인
1. **Retrieval (기존 VisRAG와 동일)**: 쿼리 q에 대해 top-K 페이지 이미지 {P_1, ..., P_K}를 VIS Retriever로 검색
2. **Visual Token Pruning (신규)**: 각 페이지 P_k를 VLM의 vision encoder로 인코딩하여 patch token sequence H_k = [h_1, ..., h_N] 획득 (N = total patches per page)
3. **Query-Patch Relevance Scoring**: 쿼리 텍스트 임베딩 e_q와 각 patch token h_i 사이의 relevance score 계산:

   ```
   s_i = softmax(W_q · e_q)^T · (W_v · h_i)   for i = 1..N
   ```

   여기서 W_q, W_v는 경량 projection matrix (학습 가능, ~2M params)

4. **Top-r Patch Selection**: 각 페이지에서 relevance score 상위 r%의 patch만 선택:

   ```
   H_k^pruned = {h_i | s_i ∈ top-r(s_1..s_N)}
   ```

   pruning ratio r은 페이지 수 K에 따라 동적으로 결정:
   ```
   r = min(1.0, budget_total / (K × N))
   ```

5. **Multi-Page Context Assembly**: 선택된 patch token들을 [SEP] 토큰으로 구분하여 VLM generator에 입력:

   ```
   context = [H_1^pruned | SEP | H_2^pruned | SEP | ... | H_K^pruned]
   ```

6. **Generation**: VLM이 압축된 multi-page visual context로 답변 생성

### 학습 전략
- **Stage 1**: 기존 VisRAG retriever 동결, token selector만 VQA loss로 학습
- **Stage 2**: selector + generator end-to-end fine-tuning (선택적)
- **Supervision**: 정답이 있는 페이지의 patch는 유지되도록 page-level hinge loss 추가:
  ```
  L_hinge = max(0, margin - mean(s_i for i in answer_page) + mean(s_j for j in noise_page))
  ```

### 계산 복잡도
- Pruning 연산: O(K × N × d) — retrieval에 비해 무시할 수준
- VLM 입력 token 수: K × N → K × (r × N), context window 절약율 = (1 - r)

## 예상 개선 효과

| Benchmark | VisRAG 기존 | 예상 개선 | 근거 |
|-----------|------------|----------|------|
| SlideVQA | ~65% | +3~5%p | 멀티페이지 슬라이드에서 cross-page 정보 통합 효과 |
| DocVQA | ~81% | +1~2%p | 관련 영역 patch 집중으로 hallucination 감소 |
| ChartQA | ~74% | +2~3%p | 차트의 핵심 부분(x/y축, 데이터 포인트) patch 보존 |
| InfoVQA | ~48% | +2~4%p | 복잡한 레이아웃에서 query-relevant 구조 추출 |

- **Context 효율성**: K=3 페이지를 처리할 때 동일 token budget으로 K=5~7 페이지까지 확장 가능
- **Latency**: pruning 오버헤드 <5ms (retrieval 대비 무시할 수준)

## 구현 복잡도
**Medium**

- 기존 VisRAG 코드베이스에 token selector 모듈 추가 (~200 lines)
- Retriever와 Generator 사이에 모듈 삽입 형태 → 기존 구조 변경 최소화
- 학습 데이터: 기존 VisRAG 학습 데이터 재사용 가능
- 추가 필요: patch-level supervision signal 생성 코드 (answer localization)
- GPU 메모리 추가 요구: 경량 selector weights (projection matrices) ~100MB

## 핵심 키워드 (novelty 검증용)
1. `visual token pruning VLM RAG`
2. `query-conditioned patch selection document retrieval`
3. `token compression multimodal context window`
4. `visual token merging question answering`
5. `patch-level relevance scoring retrieval augmented generation`
6. `multi-page visual context compression VQA`
7. `dynamic token budget vision language model`

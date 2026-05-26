# Plan 001 — VisRAG Improvement: Multi-Agent Research Pipeline

**Created**: 2026-05-26 KST  
**Mode**: Research Design (아이디어 + 실험 계획 + 스켈레톤 코드 생성; 실제 실행은 사용자가 직접)  
**Goal**: VisRAG에 접목 가능한 novel한 알고리즘 아이디어 1개 발굴 → 실험 계획 수립 → 문서화

---

## 목표

VisRAG(Vision-based RAG) 논문의 한계점을 극복하거나 성능을 향상시킬 수 있는 새로운 아이디어를 발굴하고,  
해당 아이디어의 novelty를 검증한 뒤, 실험 계획(코드 스켈레톤 포함)을 완성하는 것.

---

## Agent 구성

| Agent | 역할 | 주요 산출물 |
|-------|------|------------|
| **A1: Idea Generator** | VisRAG 분석 → 한계점 파악 → 개선 아이디어 생성 | `state/idea_candidate.md` |
| **A2: Novelty Validator** | arXiv/Semantic Scholar 검색 → novelty 판정 → feedback | `state/validation_log.md` |
| **A3: Experiment Planner** | 실험 설계 → 평가 지표 → 검증 체크리스트 | `experiment_001.md` |
| **A4: Experiment Executor** | 스켈레톤 코드 + 의사(mock) 실행 계획 작성 | skeleton code in `src/` |
| **A5: Orchestrator** | GPU/CPU 리소스 계획, 병렬 실험 전략 수립 | `state/orchestration_plan.md` |
| **A6: Documenter** | 최종 아이디어 + 실험 계획 통합 문서화 | `result_001.md` |

---

## Phase 순서 (순차, 피드백 루프 포함)

```
Phase 1: Idea Generation (A1)
    ↓
Phase 2: Novelty Validation (A2) 
    ↓ novel 판정 → Phase 3
    ↓ 기존 논문 존재 → Phase 1 재실행 (feedback loop)
Phase 3: Experiment Planning (A3)
    ↓
Phase 4+5: Skeleton Code + Orchestration Plan (A4 + A5, 병렬)
    ↓
Phase 6: Documentation (A6)
```

---

## 성공 기준 (Success Criteria)

- [ ] Novel 아이디어 1개 확보 (arXiv best-effort 검색 통과)
- [ ] 아이디어가 VisRAG 논문의 구체적 한계점과 연결됨
- [ ] 실험 계획에 명확한 baseline/metric/비교군 포함
- [ ] 스켈레톤 코드가 실제 실행 가능한 구조
- [ ] 최종 문서(result_001.md)가 논문 수준의 구성 갖춤

---

## 공유 상태 디렉토리

```
experiments/visrag/
├── state/
│   ├── idea_candidate.md       # A1 → A2 전달 아이디어
│   ├── validation_log.md       # A2 검증 결과 누적
│   ├── orchestration_plan.md   # A5 리소스 계획
│   └── round.txt               # 현재 아이디어 라운드 번호
├── experiment_001.md           # A3 실험 설계
├── result_001.md               # A6 최종 문서
└── src/                        # A4 스켈레톤 코드
```

---

## 위험 요소

| 위험 | 대응 |
|------|------|
| 아이디어가 이미 존재 | A2 → A1 feedback, 최대 10 라운드 |
| 아이디어가 너무 추상적 | A2가 "구체성 부족"으로 reject, A1 재생성 |
| 검색 결과 모호 | A2가 "아마 novel" 판정 후 유사 논문 목록 첨부 |
| 실험 계획 구현 불가 | A3가 복잡도 평가 포함, A4가 현실적 범위로 조정 |

---

## 예상 소요 시간 (연구 설계 모드)

- Phase 1-2 루프: ~15-30분
- Phase 3: ~10분  
- Phase 4+5: ~15분
- Phase 6: ~10분
- **총합: ~1시간 내외**

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|---------|
| 2026-05-26 KST | 최초 작성 |

# 5-Way Comparison (comparison_v2_tf451)

Generator: Qwen2-VL-7B-Instruct (NF4 4-bit) | Metric: Relaxed Exact Match
Samples: Chaemin first100 (oracle qrels, topk=1) | DocVQA == MP-DocVQA

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|---|---:|---:|---:|---:|---:|
| Baseline (Image) | 0.1600 | 0.3492 | 0.1600 | 0.1600 | — |
| Text — OCR(Open-영혁-Hybrid) | 0.0400 | 0.2857 | 0.1100 | 0.0900 | -7.59%p |
| Text — OCR(Closed-VDU) | 0.5400 | 0.5238 | 0.7600 | 0.5200 | +37.87%p |
| Text — OCR(Open-영혁-Hybrid) + Image | 0.1000 | 0.3333 | 0.1300 | 0.1500 | -2.90%p |
| Selective (LLM) | 0.0400 | 0.3333 | 0.1100 | 0.0900 | -6.40%p |

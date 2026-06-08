# 5-Way Comparison (comparison_v2)

Generator: Qwen2-VL-7B-Instruct (NF4 4-bit) | Metric: Relaxed Exact Match
Samples: Chaemin first100 (oracle qrels, topk=1) | DocVQA == MP-DocVQA

| Method | InfoVQA | ChartQA | MP-DocVQA | SlideVQA | 평균 Δ vs Baseline |
|---|---:|---:|---:|---:|---:|
| Baseline (Image) | 0.7200 | 0.5556 | 0.8800 | 0.5400 | — |
| Text — OCR(Open-영혁-Hybrid) | 0.3400 | 0.3175 | 0.6700 | 0.4300 | -23.45%p |
| Text — OCR(Closed-VDU) | 0.5400 | 0.5238 | 0.7600 | 0.5200 | -8.79%p |
| Text — OCR(Open-영혁-Hybrid) + Image | 0.7300 | 0.6667 | 0.8200 | 0.5800 | +2.53%p |
| Selective (LLM) | 0.3400 | 0.5238 | 0.6700 | 0.4400 | -18.04%p |

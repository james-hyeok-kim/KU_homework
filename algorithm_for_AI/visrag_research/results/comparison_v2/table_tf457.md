# Main result (tf457)

Generator: Qwen2-VL-7B-Instruct (NF4 4-bit) | EM: Relaxed Exact Match | Faith: Faithfulness (Qwen2.5-VL-72B Judge)
Samples: Chaemin first100 (oracle qrels, topk=1) | DocVQA == MP-DocVQA

| Method | InfoVQA (EM/Faith) | ChartQA (EM/Faith) | MP-DocVQA (EM/Faith) | SlideVQA (EM/Faith) | 평균 Δ vs Baseline (EM/Faith) |
|---|---:|---:|---:|---:|---:|
| Baseline (Image_only) | 0.7200 / 0.7800 | 0.5556 / 0.7302 | 0.8800 / 0.9000 | 0.5400 / 0.8500 | — / 0.8151 |
| Text_only (Qwen OCR) | 0.3400 / 0.3600 | 0.3175 / 0.2222 | 0.6700 / 0.7900 | 0.4300 / 0.5450 | -23.45%p / 0.4793 |
| Text_only (Upstage Document Parser) | 0.5400 / 0.6200 | 0.5238 / 0.5873 | 0.7600 / 0.9394 | 0.5200 / 0.6950 | -8.79%p / 0.7104 |
| Selective Text_only (Qwen OCR) | 0.3400 / 0.3500 | 0.5238 / 0.5397 | 0.6700 / 0.7800 | 0.4400 / 0.5650 | -18.04%p / 0.5587 |
| Selective Text_only (Upstage Document Parser) | 0.5400 / 0.6300 | 0.4603 / 0.6190 | 0.7700 / 0.9000 | 0.4800 / 0.6800 | -11.13%p / 0.7073 |
| Image+Text (Qwen OCR) | 0.7300 / 0.7700 | 0.6667 / 0.7778 | 0.8200 / 0.8800 | 0.5800 / 0.8500 | +2.53%p / 0.8195 |
| Image+Text (Upstage Document Parser) | 0.5400 / 0.6500 | 0.5079 / 0.5238 | 0.8000 / 0.9000 | 0.5200 / 0.7300 | -8.19%p / 0.7010 |

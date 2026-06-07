"""Render the 5-way comparison table (Chaemin slide-10 layout) from result JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Optional argv[1]: results dir name under results/ (default comparison_v2).
RESULTS = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "comparison_v2")

DATASETS = ["InfoVQA", "ChartQA", "MP-DocVQA", "SlideVQA"]

# (mode file token, display label)
METHODS = [
    ("image_only", "Baseline (Image)"),
    ("ocr_text_only", "Text — OCR(Open-영혁-Hybrid)"),
    ("closed_text_only", "Text — OCR(Closed-VDU)"),
    ("ocr_text_image", "Text — OCR(Open-영혁-Hybrid) + Image"),
    ("selective_llm", "Selective (LLM)"),
]


def load_acc(dataset: str, mode: str) -> float | None:
    path = RESULTS / f"{dataset}_{mode}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]["accuracy"]


def fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def main() -> None:
    table: dict[str, dict[str, float | None]] = {
        mode: {ds: load_acc(ds, mode) for ds in DATASETS} for mode, _ in METHODS
    }
    base = table["image_only"]

    lines = [
        f"# 5-Way Comparison ({RESULTS.name})",
        "",
        "Generator: Qwen2-VL-7B-Instruct (NF4 4-bit) | Metric: Relaxed Exact Match",
        "Samples: Chaemin first100 (oracle qrels, topk=1) | DocVQA == MP-DocVQA",
        "",
        "| Method | " + " | ".join(DATASETS) + " | 평균 Δ vs Baseline |",
        "|---|" + "---:|" * (len(DATASETS) + 1),
    ]
    for mode, label in METHODS:
        accs = table[mode]
        deltas = [
            accs[ds] - base[ds]
            for ds in DATASETS
            if accs[ds] is not None and base[ds] is not None
        ]
        if mode == "image_only":
            delta_str = "—"
        elif len(deltas) == len(DATASETS):
            delta_str = f"{sum(deltas) / len(deltas) * 100:+.2f}%p"
        else:
            delta_str = "—"
        lines.append(
            f"| {label} | " + " | ".join(fmt(accs[ds]) for ds in DATASETS) + f" | {delta_str} |"
        )

    md = "\n".join(lines) + "\n"
    out_md = RESULTS / "table.md"
    out_md.write_text(md, encoding="utf-8")
    (RESULTS / "table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(md)
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

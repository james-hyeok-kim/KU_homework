"""
Evaluation script for QCVTP across DocVQA, SlideVQA, and ChartQA.

Metrics:
  - ANLS  (Average Normalized Levenshtein Similarity) — DocVQA, SlideVQA
  - Relaxed Accuracy — ChartQA
  - Exact Match

Outputs results to JSON and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# ANLS implementation (no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def normalized_levenshtein(pred: str, gt: str) -> float:
    """1 - edit_distance / max(len(pred), len(gt)); returns 1.0 if both empty."""
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if len(pred) == 0 and len(gt) == 0:
        return 1.0
    denom = max(len(pred), len(gt))
    return 1.0 - _edit_distance(pred, gt) / denom


def anls_score(pred: str, gt_answers: list[str], threshold: float = 0.5) -> float:
    """
    ANLS for a single question.
    Any gt answer scoring above `threshold` gives its similarity score; below → 0.
    Returns max over all gt answers.
    """
    best = 0.0
    for gt in gt_answers:
        sim = normalized_levenshtein(pred, gt)
        best = max(best, sim if sim >= threshold else 0.0)
    return best


def compute_anls(
    predictions: list[str],
    ground_truths: list[list[str]],
    threshold: float = 0.5,
) -> float:
    """Corpus-level ANLS: mean of per-sample ANLS scores."""
    assert len(predictions) == len(ground_truths), "Length mismatch"
    scores = [anls_score(p, g, threshold) for p, g in zip(predictions, ground_truths)]
    return float(np.mean(scores)) if scores else 0.0


def compute_exact_match(
    predictions: list[str], ground_truths: list[list[str]]
) -> float:
    """EM: prediction matches any normalised gt answer exactly."""
    hits = 0
    for pred, gts in zip(predictions, ground_truths):
        pred_norm = pred.strip().lower()
        if any(pred_norm == g.strip().lower() for g in gts):
            hits += 1
    return hits / len(predictions) if predictions else 0.0


def relaxed_accuracy(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """
    ChartQA relaxed accuracy: numeric answers within ±5% of gt are also correct.
    Falls back to exact string match for non-numeric answers.
    """
    hits = 0
    for pred, gts in zip(predictions, ground_truths):
        pred_norm = pred.strip().lower()
        correct = False
        for gt in gts:
            gt_norm = gt.strip().lower()
            if pred_norm == gt_norm:
                correct = True
                break
            # Numeric relaxation
            try:
                p_val, g_val = float(pred_norm.replace(",", "")), float(gt_norm.replace(",", ""))
                if g_val != 0 and abs(p_val - g_val) / abs(g_val) <= 0.05:
                    correct = True
                    break
                if g_val == 0 and p_val == 0:
                    correct = True
                    break
            except ValueError:
                pass
        if correct:
            hits += 1
    return hits / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    predictions: list[str],
    ground_truths: list[list[str]],
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI for any scalar metric function."""
    rng = np.random.default_rng(seed)
    n = len(predictions)
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        sample_preds = [predictions[i] for i in idx]
        sample_gts = [ground_truths[i] for i in idx]
        scores.append(metric_fn(sample_preds, sample_gts))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(
    pipeline,  # VisRAGWithPruning (or any object with .forward())
    dataloader: DataLoader,
    dataset_name: str,
    token_budget: int = 2048,
    device: str = "cuda",
) -> dict:
    """
    Run inference on a single dataset split and compute all metrics.

    Returns a result dict with keys: dataset, anls, exact_match,
    relaxed_accuracy, n_samples, time_sec, per_sample (list of dicts).
    """
    pipeline.eval()
    all_preds: list[str] = []
    all_gts: list[list[str]] = []
    timings: list[float] = []
    token_counts_before: list[int] = []
    token_counts_after: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            queries = batch["queries"]
            pages = batch["pages"]
            gts = batch["ground_truths"]  # list[list[str]]

            t0 = time.perf_counter()

            # TODO: replace with actual pipeline forward
            # preds = pipeline.forward(queries, pages, token_budget=token_budget)
            # For now, raise to make the TODO explicit
            raise NotImplementedError(
                "Connect pipeline.forward() — see visrag_pipeline.VisRAGWithPruning.forward()"
            )

            elapsed = time.perf_counter() - t0
            timings.append(elapsed / len(queries))

            # TODO: collect token counts from pipeline internals
            # token_counts_before.extend(pipeline.last_token_counts_before)
            # token_counts_after.extend(pipeline.last_token_counts_after)

            all_preds.extend(preds)
            all_gts.extend(gts)

    # Metrics
    anls = compute_anls(all_preds, all_gts)
    em = compute_exact_match(all_preds, all_gts)
    rel_acc = relaxed_accuracy(all_preds, all_gts)

    anls_lo, anls_hi = bootstrap_ci(all_preds, all_gts, compute_anls)
    em_lo, em_hi = bootstrap_ci(all_preds, all_gts, compute_exact_match)

    result = {
        "dataset": dataset_name,
        "n_samples": len(all_preds),
        "anls": anls,
        "anls_ci_95": [anls_lo, anls_hi],
        "exact_match": em,
        "exact_match_ci_95": [em_lo, em_hi],
        "relaxed_accuracy": rel_acc,
        "avg_inference_ms": float(np.mean(timings) * 1000) if timings else None,
        "avg_tokens_before": float(np.mean(token_counts_before)) if token_counts_before else None,
        "avg_tokens_after": float(np.mean(token_counts_after)) if token_counts_after else None,
        "per_sample": [
            {"query_idx": i, "pred": p, "gts": g}
            for i, (p, g) in enumerate(zip(all_preds, all_gts))
        ],
    }
    return result


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Dataset':<15} {'ANLS':>8} {'95% CI':>18} {'EM':>8} {'RelAcc':>8}")
    print("-" * 70)
    for r in results:
        lo, hi = r["anls_ci_95"]
        print(
            f"{r['dataset']:<15} "
            f"{r['anls']:>8.4f} "
            f"[{lo:.4f}, {hi:.4f}]  "
            f"{r['exact_match']:>8.4f} "
            f"{r['relaxed_accuracy']:>8.4f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate QCVTP pipeline")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to selector checkpoint .pt")
    p.add_argument("--datasets", nargs="+", default=["slidevqa", "docvqa", "chartqa"])
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--token_budget", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_json", type=str, default="results/exp001_results.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: load pipeline
    # retriever = load_visrag_retriever(...)
    # vlm, tokenizer, processor = load_minicpm_v(...)
    # pipeline = VisRAGWithPruning(retriever, vlm, tokenizer, processor, ...)
    # pipeline.token_selector.load_state_dict(torch.load(args.checkpoint))
    # pipeline.to(args.device)

    all_results: list[dict] = []

    for ds_name in args.datasets:
        print(f"\nEvaluating {ds_name} ({args.split})...")

        # TODO: build dataset and loader for each benchmark
        # dataset = load_benchmark(ds_name, args.data_path, split=args.split)
        # loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        result = evaluate_dataset(
            pipeline=None,   # TODO: pass actual pipeline
            dataloader=None, # TODO: pass actual loader
            dataset_name=ds_name,
            token_budget=args.token_budget,
            device=args.device,
        )
        all_results.append(result)

    print_summary(all_results)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

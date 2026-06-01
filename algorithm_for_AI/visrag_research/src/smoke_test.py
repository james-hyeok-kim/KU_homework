"""
Smoke test for QCVTP pipeline.
Tests all components with synthetic tensors — no model weights required.

Run:
    cd algorithm_for_AI/visrag_research/src
    python smoke_test.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

from token_selector import QueryConditionedTokenSelector, hinge_loss
from evaluate import compute_anls, compute_exact_match, relaxed_accuracy

# ── Smoke-test config (tiny, CPU-only) ──────────────────────────────────────
B          = 2    # batch size
K          = 3    # pages per sample
N          = 64   # patches per page  (e.g., 8×8 SigLIP grid)
PATCH_DIM  = 128  # vision hidden dim
QUERY_DIM  = 128  # retriever embedding dim
PROJ_DIM   = 64   # selector projection dim
BUDGET     = 100  # token budget  (< K*N=192 → forces pruning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[smoke_test] device={DEVICE}")


# ── Test 1: TokenSelector forward & dynamic ratio ───────────────────────────
def test_token_selector():
    print("\n[T1] QueryConditionedTokenSelector forward pass")
    sel = QueryConditionedTokenSelector(QUERY_DIM, PATCH_DIM, PROJ_DIM).to(DEVICE)

    q   = torch.randn(B, QUERY_DIM, device=DEVICE)
    pts = torch.randn(B, K, N, PATCH_DIM, device=DEVICE)

    ratio = sel.compute_dynamic_ratio(K, N, BUDGET)
    k_keep = max(1, int(ratio * N))
    print(f"    keep_ratio={ratio:.3f}  →  k_keep={k_keep}/{N}")
    assert 0.2 <= ratio <= 1.0

    pruned, scores = sel(q, pts, ratio)
    assert pruned.shape == (B, K, k_keep, PATCH_DIM), \
        f"pruned shape mismatch: {pruned.shape}"
    assert scores.shape == (B, K, N), \
        f"scores shape mismatch: {scores.shape}"
    print(f"    input  {tuple(pts.shape)}  →  pruned {tuple(pruned.shape)}  scores {tuple(scores.shape)}")
    print("    PASS")


# ── Test 2: hinge_loss gradient flow ────────────────────────────────────────
def test_hinge_loss():
    print("\n[T2] hinge_loss")
    scores = torch.randn(B, K, N, requires_grad=True, device=DEVICE)
    # page 0 contains the answer, pages 1-2 are noise
    ans_mask = torch.zeros(B, K, dtype=torch.bool, device=DEVICE)
    ans_mask[:, 0] = True

    loss = hinge_loss(scores, ans_mask, margin=0.5)
    print(f"    hinge_loss={loss.item():.4f}")
    assert loss.item() >= 0.0
    loss.backward()
    assert scores.grad is not None
    print(f"    grad norm={scores.grad.norm().item():.4f}")
    print("    PASS")


# ── Test 3: K=1 full-keep bypass ────────────────────────────────────────────
def test_k1_bypass():
    print("\n[T3] K=1 bypass (keep_ratio forced to 1.0)")
    sel = QueryConditionedTokenSelector(QUERY_DIM, PATCH_DIM, PROJ_DIM).to(DEVICE)
    q   = torch.randn(B, QUERY_DIM, device=DEVICE)
    pts = torch.randn(B, 1, N, PATCH_DIM, device=DEVICE)  # K=1

    pruned, _ = sel(q, pts, keep_ratio=1.0)
    assert pruned.shape == pts.shape, f"K=1 should keep all: {pruned.shape}"
    print(f"    shape {tuple(pts.shape)} unchanged  PASS")


# ── Test 4: evaluation metrics ───────────────────────────────────────────────
def test_metrics():
    print("\n[T4] Evaluation metrics")
    preds = ["the answer is 42", "paris", "blue", "12.5"]
    gts   = [
        ["42", "the answer is 42"],   # exact match
        ["Paris", "paris"],            # case-insensitive match
        ["red", "green"],              # wrong
        ["12.6", "12.4"],              # numeric ±5% → relaxed match
    ]

    anls    = compute_anls(preds, gts)
    em      = compute_exact_match(preds, gts)
    rel_acc = relaxed_accuracy(preds, gts)

    print(f"    ANLS={anls:.4f}  EM={em:.4f}  RelAcc={rel_acc:.4f}")
    assert 0.0 <= anls    <= 1.0
    assert 0.0 <= em      <= 1.0
    assert 0.0 <= rel_acc <= 1.0
    # "12.5" should be within 5% of "12.6" → relaxed_accuracy >= 0.5
    assert rel_acc >= 0.5, f"RelAcc should be ≥0.5 (numeric match), got {rel_acc}"
    print("    PASS")


# ── Test 5: mock end-to-end pipeline ────────────────────────────────────────
class MockRetriever(nn.Module):
    """Returns random embeddings; shape matches real VisRAG retriever API."""
    def encode_queries(self, queries: list[str]) -> torch.Tensor:
        return torch.randn(len(queries), QUERY_DIM)

    def encode_pages(self, pages) -> torch.Tensor:
        B_   = len(pages)
        maxp = max(len(p) for p in pages)
        return torch.randn(B_, maxp, QUERY_DIM)


class MockVisionEncoder(nn.Module):
    """Mimics SigLIP / InternViT: image → (N, D) patch tokens."""
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz = pixel_values.shape[0]
        return torch.randn(bsz, N, PATCH_DIM, device=pixel_values.device)


def test_mock_pipeline():
    print("\n[T5] Mock end-to-end pipeline (retrieve → encode → prune → generate)")
    retriever    = MockRetriever().to(DEVICE)
    vis_encoder  = MockVisionEncoder().to(DEVICE)
    selector     = QueryConditionedTokenSelector(QUERY_DIM, PATCH_DIM, PROJ_DIM).to(DEVICE)

    queries = ["What is shown in the chart?", "What year was this published?"]

    # Stage 1: encode queries
    q_emb = retriever.encode_queries(queries).to(DEVICE)   # (B, QUERY_DIM)
    assert q_emb.shape == (len(queries), QUERY_DIM)
    print(f"    [1] query embeddings : {tuple(q_emb.shape)}")

    # Stage 2: vision encode K pages each
    dummy_pixels = torch.randn(len(queries) * K, 3, 224, 224, device=DEVICE)
    patch_flat   = vis_encoder(dummy_pixels)               # (B*K, N, D)
    patch_tokens = patch_flat.view(len(queries), K, N, PATCH_DIM)
    print(f"    [2] patch tokens     : {tuple(patch_tokens.shape)}")

    # Stage 3: token pruning
    ratio   = selector.compute_dynamic_ratio(K, N, BUDGET)
    pruned, scores = selector(q_emb, patch_tokens, ratio)
    k_keep  = pruned.shape[2]

    tokens_before = len(queries) * K * N
    tokens_after  = len(queries) * K * k_keep
    reduction     = (1 - tokens_after / tokens_before) * 100
    print(f"    [3] pruned tokens    : {tuple(pruned.shape)}  "
          f"({tokens_before}→{tokens_after} tokens, {reduction:.1f}% saved)")
    assert tokens_after < tokens_before

    # Stage 4: mock generation (placeholder for VLM.generate)
    answers = ["chart shows quarterly sales", "published in 2024"]
    print(f"    [4] generated answers: {answers}")
    print("    PASS")


# ── Runner ───────────────────────────────────────────────────────────────────
TESTS = [
    test_token_selector,
    test_hinge_loss,
    test_k1_bypass,
    test_metrics,
    test_mock_pipeline,
]

if __name__ == "__main__":
    print("=" * 60)
    print("QCVTP Smoke Test  —  2026-05-26 KST")
    print("=" * 60)
    failed = []
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"    FAILED: {exc}")
            traceback.print_exc()
            failed.append(fn.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(TESTS)}): {failed}")
        sys.exit(1)
    else:
        print(f"ALL {len(TESTS)} TESTS PASSED")

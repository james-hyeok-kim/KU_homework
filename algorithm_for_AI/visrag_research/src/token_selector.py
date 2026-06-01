"""
Query-Conditioned Visual Token Selector.

Lightweight bilinear dot-product scoring module inserted between VisRAG retriever and VLM generator.
Projects query and patch tokens into a shared 256-dim space, computes per-patch relevance scores
via dot product, then selects the top-r fraction. No cross-attention, no softmax weighting.
Prunes patch tokens per page based on query relevance, enabling more pages within a fixed token budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QueryConditionedTokenSelector(nn.Module):
    """Scores each visual patch token against a query embedding and keeps the top-r fraction."""

    def __init__(self, query_dim: int, patch_dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.W_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_dim = proj_dim

    def forward(
        self,
        query_embedding: torch.Tensor,  # (B, query_dim)
        patch_tokens: torch.Tensor,     # (B, K, N, patch_dim)
        keep_ratio: float,              # r = budget / (K * N), clamped to [0.2, 1.0]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pruned_tokens : (B, K, k_keep, patch_dim)  selected patch tokens
            scores        : (B, K, N)                   raw relevance scores for hinge loss
        """
        B, K, N, D = patch_tokens.shape

        # Project query once, then broadcast over pages
        q_proj = self.W_q(query_embedding)  # (B, proj_dim)
        q_proj = F.normalize(q_proj, dim=-1)

        # Project all patch tokens
        v_flat = patch_tokens.reshape(B * K, N, D)
        v_proj = self.W_v(v_flat)                  # (B*K, N, proj_dim)
        v_proj = F.normalize(v_proj, dim=-1)

        # Relevance score: dot product between query projection and each patch projection
        q_expanded = q_proj.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)  # (B*K, proj_dim)
        scores_flat = torch.bmm(
            v_proj, q_expanded.unsqueeze(-1)
        ).squeeze(-1)              # (B*K, N)
        scores = scores_flat.view(B, K, N)

        # Top-r patch selection
        keep_ratio = max(0.2, min(1.0, keep_ratio))  # enforce [0.2, 1.0] floor
        k_keep = max(1, int(keep_ratio * N))
        topk_indices = scores.topk(k_keep, dim=-1).indices  # (B, K, k_keep)

        topk_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        pruned_tokens = patch_tokens.gather(2, topk_exp)    # (B, K, k_keep, D)

        return pruned_tokens, scores

    def compute_dynamic_ratio(
        self, num_pages: int, tokens_per_page: int, token_budget: int
    ) -> float:
        """r = min(1.0, budget / (K * N)), floor at 0.2 to prevent over-pruning."""
        raw = token_budget / max(1, num_pages * tokens_per_page)
        return max(0.2, min(1.0, raw))


def hinge_loss(
    scores: torch.Tensor,            # (B, K, N)
    answer_page_mask: torch.Tensor,  # (B, K) bool, True = page contains the answer
    noise_page_mask: Optional[torch.Tensor] = None,  # (B, K) bool; complement of answer_page_mask if None
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Ensures answer-page patches score higher than noise-page patches by `margin`.

    Skips samples where supervision is unavailable (all-positive or all-negative pages).
    Returns scalar loss (0.0 if no valid samples in batch).
    """
    B, K, N = scores.shape
    page_mean = scores.mean(dim=-1)  # (B, K)

    if noise_page_mask is None:
        noise_page_mask = ~answer_page_mask

    # Anchor to scores so zero is differentiable when no valid pairs exist
    loss = scores.sum() * 0.0
    count = 0

    for b in range(B):
        pos_mask = answer_page_mask[b]  # (K,)
        neg_mask = noise_page_mask[b]   # (K,)

        if not pos_mask.any() or not neg_mask.any():
            # Cannot apply hinge without both positive and negative pages
            continue

        pos_score = page_mean[b][pos_mask].mean()
        neg_score = page_mean[b][neg_mask].mean()
        loss = loss + torch.clamp(margin - pos_score + neg_score, min=0.0)
        count += 1

    return loss / max(count, 1)

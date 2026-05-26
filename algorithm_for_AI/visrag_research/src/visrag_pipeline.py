"""
VisRAG pipeline extended with Query-Conditioned Visual Token Pruning (QCVTP).

Wraps a frozen VisRAG retriever + MiniCPM-V (or InternVL) generator and inserts
the token selector between retrieval and generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from token_selector import QueryConditionedTokenSelector


class VisRAGWithPruning(nn.Module):
    """End-to-end VisRAG pipeline with patch-token pruning."""

    SEP_TOKEN_ID: int = 2  # TODO: confirm separator token id for target VLM

    def __init__(
        self,
        retriever: nn.Module,
        vlm: nn.Module,
        vlm_tokenizer,
        vlm_processor,
        query_dim: int,
        patch_dim: int,
        proj_dim: int = 256,
        token_budget: int = 2048,
        min_keep_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.vlm = vlm
        self.tokenizer = vlm_tokenizer
        self.processor = vlm_processor
        self.token_budget = token_budget
        self.min_keep_ratio = min_keep_ratio

        self.token_selector = QueryConditionedTokenSelector(
            query_dim=query_dim,
            patch_dim=patch_dim,
            proj_dim=proj_dim,
        )

    # ------------------------------------------------------------------
    # Retrieval stage (unchanged from vanilla VisRAG)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: list[str],
        document_pages: list[list],  # list[list[PIL.Image]] per sample
        top_k: int = 5,
    ) -> tuple[list[list], torch.Tensor]:
        """
        Returns:
            selected_pages : top-K PIL images per sample
            query_embeddings : (B, query_dim) float tensor
        """
        # TODO: replace with actual VisRAG retriever call
        # query_embeddings = self.retriever.encode_queries(query)         # (B, query_dim)
        # page_embeddings  = self.retriever.encode_pages(document_pages)  # (B, total_pages, query_dim)
        # scores = torch.matmul(query_embeddings, page_embeddings.transpose(-1, -2))
        # topk_idx = scores.topk(top_k, dim=-1).indices
        # selected_pages = [[document_pages[b][i] for i in topk_idx[b]] for b in range(len(query))]
        raise NotImplementedError("Connect VisRAG retriever here")

    # ------------------------------------------------------------------
    # Vision encoding stage
    # ------------------------------------------------------------------

    def encode_pages_to_patch_tokens(
        self, selected_pages: list[list]
    ) -> torch.Tensor:
        """
        Encode a batch of page image lists into patch token tensors.

        Returns:
            patch_tokens : (B, K, N, patch_dim)
        """
        # TODO: extract intermediate hidden states from VLM vision encoder
        # For MiniCPM-V, hook into `self.vlm.vpm` (SigLIP visual encoder).
        # For InternVL, use `self.vlm.vision_model`.
        #
        # Example with forward hook:
        #   hidden_states = []
        #   hook = self.vlm.vpm.register_forward_hook(
        #       lambda m, i, o: hidden_states.append(o.last_hidden_state)
        #   )
        #   self.vlm.vpm(pixel_values)
        #   hook.remove()
        #   return hidden_states[-1]  # (B*K, N, patch_dim)
        raise NotImplementedError("Implement vision encoder patch extraction")

    # ------------------------------------------------------------------
    # Token pruning stage
    # ------------------------------------------------------------------

    def prune_tokens(
        self,
        query_embeddings: torch.Tensor,  # (B, query_dim)
        patch_tokens: torch.Tensor,      # (B, K, N, patch_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            pruned_tokens : (B, K, k_keep, patch_dim)
            scores        : (B, K, N)  relevance scores for loss computation
            keep_ratio    : actual r used
        """
        B, K, N, _ = patch_tokens.shape
        keep_ratio = self.token_selector.compute_dynamic_ratio(K, N, self.token_budget)
        keep_ratio = max(self.min_keep_ratio, keep_ratio)

        # Bypass pruning when only one page (r would be 1.0 anyway)
        if K == 1:
            scores = torch.ones(B, K, N, device=patch_tokens.device)
            return patch_tokens, scores, 1.0

        pruned_tokens, scores = self.token_selector(query_embeddings, patch_tokens, keep_ratio)
        return pruned_tokens, scores, keep_ratio

    # ------------------------------------------------------------------
    # Generation stage
    # ------------------------------------------------------------------

    def assemble_and_generate(
        self,
        queries: list[str],
        pruned_tokens: torch.Tensor,  # (B, K, k_keep, patch_dim)
    ) -> list[str]:
        """
        Concatenate pruned patch tokens across pages with SEP, then run VLM generation.

        Returns:
            answers : list of generated answer strings
        """
        # TODO: assemble pruned tokens into VLM input format
        # The assembly strategy depends on the VLM API:
        #   - Flatten: (B, K*k_keep, patch_dim) with SEP tokens interspersed
        #   - Then pass to VLM.generate() with custom visual token injection
        #
        # For MiniCPM-V the entry point is vlm.chat() or vlm.generate();
        # patch tokens may need to be injected via model.get_vllm_embedding().
        raise NotImplementedError("Implement VLM generation with injected pruned tokens")

    def compute_loss(
        self,
        queries: list[str],
        pruned_tokens: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,                            # token ids of target answers
        answer_page_mask: Optional[torch.Tensor] = None, # (B, K) bool
        lambda_hinge: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss : scalar
            metrics    : dict with vqa_loss and hinge_loss values
        """
        from token_selector import hinge_loss
        import torch.nn.functional as F

        # TODO: get VLM logits over vocabulary from pruned_tokens + queries
        # logits = self.assemble_and_forward_logits(queries, pruned_tokens)  # (B, seq_len, vocab)
        # vqa_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        raise NotImplementedError("Implement logit extraction for VQA loss")

        if answer_page_mask is not None:
            h_loss = hinge_loss(scores, answer_page_mask)
        else:
            h_loss = scores.new_zeros(1).squeeze()

        total_loss = vqa_loss + lambda_hinge * h_loss
        metrics = {
            "vqa_loss": vqa_loss.item(),
            "hinge_loss": h_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: list[str],
        document_pages: list[list],  # list[list[PIL.Image]] per sample
        token_budget: Optional[int] = None,
        top_k: int = 5,
    ) -> list[str]:
        """
        Full pipeline: retrieve -> encode -> prune -> generate.

        Args:
            query          : batch of question strings
            document_pages : for each sample, all candidate page images
            token_budget   : override self.token_budget if provided
            top_k          : number of pages to retrieve

        Returns:
            answers : generated answer strings
        """
        if token_budget is not None:
            self.token_budget = token_budget

        # Stage 1: Retrieval
        selected_pages, query_embeddings = self.retrieve(query, document_pages, top_k)

        # Stage 2: Vision encoding
        patch_tokens = self.encode_pages_to_patch_tokens(selected_pages)  # (B, K, N, D)

        # Stage 3: Token pruning
        pruned_tokens, _, _ = self.prune_tokens(query_embeddings, patch_tokens)

        # Stage 4: Generation
        answers = self.assemble_and_generate(query, pruned_tokens)

        return answers

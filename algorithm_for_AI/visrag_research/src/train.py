"""
Training script for Query-Conditioned Visual Token Pruning (QCVTP).

Stage 1: retriever + VLM frozen, selector only.
Stage 2: end-to-end fine-tuning (optional, high VRAM).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from token_selector import QueryConditionedTokenSelector, hinge_loss
from visrag_pipeline import VisRAGWithPruning


# ---------------------------------------------------------------------------
# Dataset placeholder — replace with actual SlideVQA / DocVQA loaders
# ---------------------------------------------------------------------------

class VisRAGDataset(torch.utils.data.Dataset):
    """
    Minimal dataset interface for multi-page VQA.

    Each item: {
        "query"           : str,
        "pages"           : list[PIL.Image],          # top-K retrieved pages
        "answer_ids"      : list[int],                # tokenised target answer
        "answer_page_mask": list[bool],               # True if page has answer
    }
    """

    def __init__(self, data_path: str, split: str = "train", max_samples: Optional[int] = None):
        # TODO: load actual dataset (HuggingFace datasets, local JSON, etc.)
        self.data_path = data_path
        self.split = split
        self.samples: list[dict] = []
        # TODO: self.samples = load_slidevqa(data_path, split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    # TODO: pad answer_ids to same length, stack masks
    queries = [x["query"] for x in batch]
    pages = [x["pages"] for x in batch]
    answer_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x["answer_ids"]) for x in batch],
        batch_first=True,
        padding_value=-100,
    )
    answer_page_mask = torch.stack(
        [torch.tensor(x["answer_page_mask"]) for x in batch]
    )
    return {
        "queries": queries,
        "pages": pages,
        "answer_ids": answer_ids,
        "answer_page_mask": answer_page_mask,
    }


# ---------------------------------------------------------------------------
# Stage 1: selector-only training
# ---------------------------------------------------------------------------

def train_stage1(
    pipeline: VisRAGWithPruning,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    max_steps: int = 2000,
    lambda_hinge: float = 0.1,
    grad_accum: int = 4,
    checkpoint_dir: str = "checkpoints/stage1",
    val_every: int = 500,
    device: str = "cuda",
) -> None:
    """Train only the token selector; retriever and VLM are frozen."""

    # Freeze retriever and VLM
    for p in pipeline.retriever.parameters():
        p.requires_grad_(False)
    for p in pipeline.vlm.parameters():
        p.requires_grad_(False)
    pipeline.token_selector.train()

    optimizer = AdamW(pipeline.token_selector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad()
    log_entries: list[dict] = []

    pipeline.to(device)

    for batch in train_loader:
        if step >= max_steps:
            break

        queries = batch["queries"]
        pages = batch["pages"]
        answer_ids = batch["answer_ids"].to(device)
        answer_page_mask = batch["answer_page_mask"].to(device)

        with torch.no_grad():
            # Retriever: query embeddings
            # TODO: replace with actual retriever call
            q_emb = pipeline.retriever.encode_queries(queries)     # (B, query_dim)
            # Vision encoder: patch tokens per page
            patch_tokens = pipeline.encode_pages_to_patch_tokens(pages)  # (B, K, N, D)

        K, N = patch_tokens.shape[1], patch_tokens.shape[2]
        keep_ratio = pipeline.token_selector.compute_dynamic_ratio(
            K, N, pipeline.token_budget
        )

        pruned_tokens, scores = pipeline.token_selector(q_emb, patch_tokens, keep_ratio)

        # VQA loss: compute logits from pruned tokens
        # TODO: pipeline.compute_loss returns (loss, metrics)
        loss, metrics = pipeline.compute_loss(
            queries,
            pruned_tokens,
            scores,
            answer_ids,
            answer_page_mask=answer_page_mask,
            lambda_hinge=lambda_hinge,
        )

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(pipeline.token_selector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            entry = {"step": step, "keep_ratio": keep_ratio, **metrics}
            log_entries.append(entry)
            print(
                f"[Stage1] step={step:4d} | "
                f"vqa={metrics['vqa_loss']:.4f} | "
                f"hinge={metrics['hinge_loss']:.4f} | "
                f"r={keep_ratio:.3f}"
            )

        if val_loader is not None and (step + 1) % val_every == 0:
            # TODO: run evaluate() on val_loader and log ANLS
            pass

        if (step + 1) % 1000 == 0:
            ckpt_path = ckpt_dir / f"step{step + 1}.pt"
            torch.save(pipeline.token_selector.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        step += 1

    # Final checkpoint
    torch.save(pipeline.token_selector.state_dict(), ckpt_dir / "final.pt")
    (ckpt_dir / "train_log.json").write_text(json.dumps(log_entries, indent=2))
    print("Stage 1 training complete.")


# ---------------------------------------------------------------------------
# Stage 2: end-to-end fine-tuning (optional)
# ---------------------------------------------------------------------------

def train_stage2(
    pipeline: VisRAGWithPruning,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    lr: float = 5e-6,            # lower LR for end-to-end
    warmup_steps: int = 200,
    max_steps: int = 5000,
    lambda_hinge: float = 0.05,
    grad_accum: int = 16,        # large accumulation to handle OOM with K=5
    checkpoint_dir: str = "checkpoints/stage2",
    device: str = "cuda",
    use_gradient_checkpointing: bool = True,
) -> None:
    """Unfreeze selector + VLM generator for end-to-end fine-tuning."""

    # Unfreeze selector and VLM generator (keep retriever frozen)
    for p in pipeline.retriever.parameters():
        p.requires_grad_(False)
    for p in pipeline.vlm.parameters():
        p.requires_grad_(True)
    pipeline.token_selector.train()

    if use_gradient_checkpointing:
        # TODO: enable gradient checkpointing on VLM
        # pipeline.vlm.gradient_checkpointing_enable()
        pass

    trainable = list(pipeline.token_selector.parameters()) + list(pipeline.vlm.parameters())
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad()

    pipeline.to(device)

    for batch in train_loader:
        if step >= max_steps:
            break

        queries = batch["queries"]
        pages = batch["pages"]
        answer_ids = batch["answer_ids"].to(device)
        answer_page_mask = batch["answer_page_mask"].to(device)

        # Full forward with gradients through VLM
        with torch.no_grad():
            q_emb = pipeline.retriever.encode_queries(queries)

        patch_tokens = pipeline.encode_pages_to_patch_tokens(pages)
        K, N = patch_tokens.shape[1], patch_tokens.shape[2]
        keep_ratio = pipeline.token_selector.compute_dynamic_ratio(K, N, pipeline.token_budget)
        pruned_tokens, scores = pipeline.token_selector(q_emb, patch_tokens, keep_ratio)

        loss, metrics = pipeline.compute_loss(
            queries, pruned_tokens, scores, answer_ids,
            answer_page_mask=answer_page_mask,
            lambda_hinge=lambda_hinge,
        )

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            print(
                f"[Stage2] step={step:4d} | "
                f"vqa={metrics['vqa_loss']:.4f} | "
                f"hinge={metrics['hinge_loss']:.4f}"
            )

        if (step + 1) % 1000 == 0:
            ckpt_path = ckpt_dir / f"step{step + 1}.pt"
            torch.save(
                {
                    "selector": pipeline.token_selector.state_dict(),
                    "vlm": pipeline.vlm.state_dict(),
                },
                ckpt_path,
            )

        step += 1

    torch.save(
        {
            "selector": pipeline.token_selector.state_dict(),
            "vlm": pipeline.vlm.state_dict(),
        },
        ckpt_dir / "final.pt",
    )
    print("Stage 2 training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QCVTP token selector")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="slidevqa", choices=["slidevqa", "docvqa", "chartqa"])
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_hinge", type=float, default=0.1)
    p.add_argument("--token_budget", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--sanity_check", action="store_true", help="Run on 100 samples only")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    max_samples = 100 if args.sanity_check else None

    # TODO: instantiate retriever and VLM
    # retriever = load_visrag_retriever(...)
    # vlm, tokenizer, processor = load_minicpm_v(...)
    # pipeline = VisRAGWithPruning(retriever, vlm, tokenizer, processor, ...)

    train_dataset = VisRAGDataset(args.data_path, split="train", max_samples=max_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # TODO: val_loader = DataLoader(VisRAGDataset(args.data_path, split="val", ...), ...)

    ckpt_dir = os.path.join(args.checkpoint_dir, f"exp001_seed{args.seed}")

    if args.stage == 1:
        train_stage1(
            pipeline=None,  # TODO: pass actual pipeline
            train_loader=train_loader,
            val_loader=None,
            lr=args.lr,
            max_steps=args.max_steps,
            lambda_hinge=args.lambda_hinge,
            grad_accum=args.grad_accum,
            checkpoint_dir=os.path.join(ckpt_dir, "stage1"),
            device=args.device,
        )
    else:
        train_stage2(
            pipeline=None,  # TODO: pass actual pipeline with Stage 1 weights loaded
            train_loader=train_loader,
            val_loader=None,
            lr=1e-5,
            max_steps=args.max_steps,
            lambda_hinge=args.lambda_hinge,
            grad_accum=args.grad_accum,
            checkpoint_dir=os.path.join(ckpt_dir, "stage2"),
            device=args.device,
        )


if __name__ == "__main__":
    main()

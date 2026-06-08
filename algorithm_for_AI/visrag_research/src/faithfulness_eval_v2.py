"""
Faithfulness evaluation for comparison_v2 results using Qwen2.5-VL-72B as judge.

Adapts kjmin's faithfulness.py logic to our comparison_v2 data format.
Uses transformers directly (no vLLM server needed).

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python3 faithfulness_eval_v2.py \
        --result-dir results/comparison_v2

    CUDA_VISIBLE_DEVICES=2,3 python3 faithfulness_eval_v2.py \
        --result-dir results/comparison_v2_tf451

Output:
    {result_dir}/faithfulness/{method}/{dataset}.jsonl
    {result_dir}/faithfulness_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # visrag_research/
CHAEMIN = ROOT.parent / "AI-Final-project_chaemin"

JUDGE_MODEL_PATH = Path("/data/jameskimh/visrag_experiment/models/Qwen2.5-VL-72B-Instruct")
LOCAL_DATA = Path("/data/jameskimh/visrag_experiment/data/test")
LOCAL_DIR = {
    "InfoVQA": "infovqa",
    "ChartQA": "chartqa",
    "MP-DocVQA": "docvqa",
    "SlideVQA": "slidevqa",
}

DATASETS = ["InfoVQA", "ChartQA", "MP-DocVQA", "SlideVQA"]
# file token → display label
METHODS = [
    ("image_only",         "Baseline (Image)"),
    ("ocr_text_only",      "Text — OCR(Open-영혁-Hybrid)"),
    ("closed_text_only",   "Text — OCR(Closed-VDU)"),
    ("selective_llm",      "Selective (LLM)"),
    ("selective_upstage",  "Selective (Upstage VDU)"),
    ("ocr_text_image",     "Text — OCR(Open-영혁-Hybrid) + Image"),
    ("closed_text_image",  "Text — OCR(Closed-VDU) + Image"),
]
METHOD_TOKENS = [m for m, _ in METHODS]

# Evidence shared between tf457 and tf451 (same corpus, same OCR)
SHARED_CACHE_DIR = ROOT / "results" / "comparison_v2" / "cache"

MAX_TEXT_CHARS = 12000
MAX_NEW_TOKENS = 256
VRAM_THRESHOLD_PCT = 98
SAMPLE_TIMEOUT_SEC = 240  # per-sample SIGALRM timeout (safety net for CUDA hangs)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S KST")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# VRAM watchdog
# ---------------------------------------------------------------------------

def spawn_watchdog(physical_gpu_idx: int, target_pid: int, flag_dir: Path) -> subprocess.Popen:
    flag_file = flag_dir / f"watchdog_gpu{physical_gpu_idx}.flag"
    watchdog_sh = ROOT / "src" / "vram_watchdog.sh"
    proc = subprocess.Popen(
        ["bash", str(watchdog_sh), str(physical_gpu_idx), str(VRAM_THRESHOLD_PCT),
         str(target_pid), str(flag_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log(f"Spawned VRAM watchdog for GPU {physical_gpu_idx} (threshold {VRAM_THRESHOLD_PCT}%, pid={proc.pid})")
    return proc


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def get_max_memory(headroom_gb: float = 15.0) -> dict:
    mem = {}
    for i in range(torch.cuda.device_count()):
        free_bytes = torch.cuda.mem_get_info(i)[0]
        usable_gb = max(0, free_bytes / (1024 ** 3) - headroom_gb)
        mem[i] = f"{int(usable_gb)}GiB"
    return mem


def load_judge_model():
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor"]

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    max_mem = get_max_memory(headroom_gb=15)
    log(f"Loading Qwen2.5-VL-72B, max_memory={max_mem}")

    processor = AutoProcessor.from_pretrained(str(JUDGE_MODEL_PATH))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(JUDGE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        attn_implementation="flash_attention_2",
    ).eval()

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"Loaded {params_b:.1f}B params")

    processor.image_processor.max_pixels = 1280 * 1024
    processor.image_processor.min_pixels = 256 * 256

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["processor"] = processor
    return model, processor


# ---------------------------------------------------------------------------
# Evidence loaders
# ---------------------------------------------------------------------------

def _load_jsonl_as_dict(path: Path) -> dict[str, str]:
    """Load a {key, value} JSONL file into a dict."""
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row["key"]] = row["value"]
    return result


def load_ocr_cache(dataset: str) -> dict[str, str]:
    path = SHARED_CACHE_DIR / f"ocr_{dataset}.jsonl"
    return _load_jsonl_as_dict(path)


def load_route_cache(dataset: str) -> dict[str, str]:
    path = SHARED_CACHE_DIR / f"route_{dataset}.jsonl"
    return _load_jsonl_as_dict(path)


def load_upstage_cache(dataset: str) -> dict[str, str]:
    path = CHAEMIN / "data" / "parsed" / f"{dataset}_first100_qrels_upstage.jsonl"
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row["corpus_id"]] = row.get("text", "")
    return result


def load_corpus_images(dataset: str, needed: set[str]) -> dict[str, Image.Image]:
    corpus_dir = LOCAL_DATA / LOCAL_DIR[dataset] / "corpus"
    images: dict[str, Image.Image] = {}
    for parquet in sorted(corpus_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet, columns=["corpus-id", "image"])
        for _, row in df.iterrows():
            cid = row["corpus-id"]
            if cid in needed and cid not in images:
                img_data = row["image"]
                raw = img_data["bytes"] if isinstance(img_data, dict) else img_data
                images[cid] = Image.open(BytesIO(raw)).convert("RGB")
        if len(images) == len(needed):
            break
    missing = needed - set(images)
    if missing:
        log(f"WARNING: {dataset}: {len(missing)} images missing: {sorted(missing)[:3]}")
    return images


# ---------------------------------------------------------------------------
# Judge prompt (adapted from kjmin's _build_judge_prompt)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating answer faithfulness for a multimodal RAG system.

Assume the retrieved context is the only source of truth.
Do not use outside knowledge.
Do not judge whether the retrieved context is the correct gold context.
Judge only whether the model answer is supported by the provided context.

If the answer is a multiple-choice letter, map it to the option text in the original prompt before judging support.
If the provided context is insufficient to support the answer, mark it unsupported.
Return only valid JSON with this schema:
{{
  "supported": true or false,
  "score": 1.0, 0.5, or 0.0,
  "evidence_used": "image", "text", "both", or "none",
  "reason": "brief explanation"
}}
Do not use LaTeX or backslash characters in the JSON string values.

Dataset: {dataset}
Method: {method}
Question:
{query}

Model answer:
{prediction}

Retrieved image evidence: {image_count} image(s) attached after this text.
Retrieved OCR text evidence:
{text_context}
"""


def build_judge_prompt(row: dict, text_context: str, image_count: int) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        dataset=row["dataset"],
        method=row["method"],
        query=row["query"],
        prediction=row.get("prediction", ""),
        image_count=image_count,
        text_context=text_context if text_context else "[none]",
    )


def format_text_context(texts: list[tuple[str, str]], max_chars: int = MAX_TEXT_CHARS) -> str:
    """texts: list of (doc_id, text)"""
    blocks = []
    total = 0
    for doc_id, text in texts:
        block = f"[{doc_id}]\n{text}".strip()
        if not block:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total += len(blocks[-1])
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Judge inference
# ---------------------------------------------------------------------------

def _parse_judge_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    for candidate in (cleaned, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)):
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {"_parse_error": cleaned[:200]}


def _coerce_score(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


@torch.no_grad()
def judge_one(
    model,
    processor,
    row: dict,
    images: list[Image.Image],
    texts: list[tuple[str, str]],
) -> dict:
    from qwen_vl_utils import process_vision_info

    text_context = format_text_context(texts)
    prompt = build_judge_prompt(row, text_context, len(images))

    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)

    if image_inputs is not None:
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
    else:
        inputs = processor(
            text=[text_input],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    raw_response = processor.batch_decode(
        out[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    parsed = _parse_judge_json(raw_response)
    return {
        "dataset": row["dataset"],
        "method": row["method"],
        "query_id": row["qid"],
        "judge_model": "Qwen2.5-VL-72B-Instruct",
        "faithfulness_score": _coerce_score(parsed.get("score")),
        "supported": parsed.get("supported"),
        "evidence_used": parsed.get("evidence_used"),
        "reason": parsed.get("reason", ""),
        "raw_response": raw_response,
        "parse_error": parsed.get("_parse_error"),
    }


# ---------------------------------------------------------------------------
# Evidence resolution per method
# ---------------------------------------------------------------------------

def get_evidence(
    row: dict,
    method: str,
    corpus_images: dict[str, Image.Image],
    ocr_cache: dict[str, str],
    upstage_cache: dict[str, str],
    route_cache: dict[str, str],
) -> tuple[list[Image.Image], list[tuple[str, str]]]:
    docids = row.get("docids", [])
    images: list[Image.Image] = []
    texts: list[tuple[str, str]] = []

    if method == "image_only":
        for did in docids:
            if did in corpus_images:
                images.append(corpus_images[did])

    elif method == "ocr_text_only":
        for did in docids:
            if did in ocr_cache:
                texts.append((did, ocr_cache[did]))

    elif method == "closed_text_only":
        for did in docids:
            if did in upstage_cache:
                texts.append((did, upstage_cache[did]))

    elif method == "ocr_text_image":
        for did in docids:
            if did in corpus_images:
                images.append(corpus_images[did])
            if did in ocr_cache:
                texts.append((did, ocr_cache[did]))

    elif method == "selective_llm":
        for did in docids:
            route = route_cache.get(did, "mixed")
            if route == "chart":
                if did in corpus_images:
                    images.append(corpus_images[did])
            else:  # text or mixed
                if did in ocr_cache:
                    texts.append((did, ocr_cache[did]))

    elif method == "selective_upstage":
        for did in docids:
            route = route_cache.get(did, "mixed")
            if route == "chart":
                if did in corpus_images:
                    images.append(corpus_images[did])
            else:  # text or mixed → Upstage VDU text
                if did in upstage_cache:
                    texts.append((did, upstage_cache[did]))

    elif method == "closed_text_image":
        for did in docids:
            if did in corpus_images:
                images.append(corpus_images[did])
            if did in upstage_cache:
                texts.append((did, upstage_cache[did]))

    return images, texts


# ---------------------------------------------------------------------------
# Per-dataset-method evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset_method(
    result_dir: Path,
    dataset: str,
    method_token: str,
    model,
    processor,
    limit: int | None = None,
) -> dict:
    # Load predictions
    pred_file = result_dir / f"{dataset}_{method_token}.json"
    if not pred_file.exists():
        log(f"  SKIP: {pred_file.name} not found")
        return {}
    with open(pred_file, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if limit is not None:
        rows = rows[:limit]

    # Output path
    faith_dir = result_dir / "faithfulness" / method_token
    faith_dir.mkdir(parents=True, exist_ok=True)
    out_path = faith_dir / f"{dataset}.jsonl"

    # Load already-done query_ids
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_ids.add(r["query_id"])
                except Exception:
                    pass
    log(f"  {dataset}/{method_token}: {len(rows)} preds, {len(done_ids)} already done")

    # Load evidence caches
    ocr_cache = load_ocr_cache(dataset)
    upstage_cache = load_upstage_cache(dataset)
    route_cache = load_route_cache(dataset)

    # Coverage check: warn early so we don't discover empty evidence mid-loop
    all_docids = {did for r in rows for did in r.get("docids", [])}
    if method_token in ("ocr_text_only", "ocr_text_image", "selective_llm"):
        missing_ocr = all_docids - set(ocr_cache.keys())
        if missing_ocr:
            log(f"  WARN: {len(missing_ocr)} docids missing from OCR cache for {dataset}: {sorted(missing_ocr)[:3]}")
    if method_token in ("closed_text_only", "closed_text_image", "selective_upstage"):
        missing_up = all_docids - set(upstage_cache.keys())
        if missing_up:
            log(f"  WARN: {len(missing_up)} docids missing from Upstage cache for {dataset}: {sorted(missing_up)[:3]}")

    # Collect needed images (only for image-using methods)
    needs_images = method_token in ("image_only", "ocr_text_image", "selective_llm", "selective_upstage", "closed_text_image")
    corpus_images: dict[str, Image.Image] = {}
    if needs_images:
        needed_docids = {
            did
            for row in rows
            if row["qid"] not in done_ids
            for did in row.get("docids", [])
        }
        if needed_docids:
            log(f"  Loading {len(needed_docids)} corpus images for {dataset} ...")
            corpus_images = load_corpus_images(dataset, needed_docids)

    def _sigalrm_handler(signum, frame):
        raise TimeoutError(f"Sample timed out after {SAMPLE_TIMEOUT_SEC}s")

    todo = [r for r in rows if r["qid"] not in done_ids]
    scores: list[float] = []
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(todo):
            t0 = time.time()
            try:
                signal.signal(signal.SIGALRM, _sigalrm_handler)
                signal.alarm(SAMPLE_TIMEOUT_SEC)
                try:
                    images, texts = get_evidence(
                        row, method_token, corpus_images, ocr_cache, upstage_cache, route_cache
                    )
                    result = judge_one(model, processor, row, images, texts)
                finally:
                    signal.alarm(0)
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"  ERROR at {row['qid']}: {e}")
                torch.cuda.empty_cache()
                result = {
                    "dataset": row["dataset"],
                    "method": method_token,
                    "query_id": row["qid"],
                    "judge_model": "Qwen2.5-VL-72B-Instruct",
                    "faithfulness_score": None,
                    "supported": None,
                    "evidence_used": None,
                    "reason": f"ERROR: {e}",
                    "raw_response": "",
                    "parse_error": str(e),
                }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            if result.get("faithfulness_score") is not None:
                scores.append(result["faithfulness_score"])
            elapsed = time.time() - t0
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  [{i+1}/{len(todo)}] {row['qid']} | score={result.get('faithfulness_score')} | {elapsed:.1f}s")

    # Compute summary from ALL done rows
    all_scores: list[float] = []
    all_supported: list[float] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = r.get("faithfulness_score")
            if s is not None:
                all_scores.append(float(s))
            sup = r.get("supported")
            if isinstance(sup, bool):
                all_supported.append(float(sup))

    summary = {
        "dataset": dataset,
        "method": method_token,
        "num_predictions": len(rows),
        "num_judged": len(all_scores),
        "faithfulness_macro": sum(all_scores) / len(all_scores) if all_scores else None,
        "supported_rate": sum(all_supported) / len(all_supported) if all_supported else None,
    }
    log(f"  -> {dataset}/{method_token}: faithfulness_macro={summary['faithfulness_macro']:.4f}" if summary['faithfulness_macro'] else f"  -> {dataset}/{method_token}: no scores")
    return summary


# ---------------------------------------------------------------------------
# Update table.md with faithfulness
# ---------------------------------------------------------------------------

def update_table_with_faithfulness(result_dir: Path, summaries: list[dict]) -> None:
    # Build lookup: method_token -> dataset -> faithfulness_macro
    faith: dict[str, dict[str, float | None]] = {}
    for s in summaries:
        m = s.get("method", "")
        d = s.get("dataset", "")
        if m and d:
            if m not in faith:
                faith[m] = {}
            faith[m][d] = s.get("faithfulness_macro")

    # Determine which table file to update
    if "tf451" in result_dir.name:
        table_file = result_dir / "table_tf451.md"
    else:
        table_file = result_dir / "table_tf457.md"

    if not table_file.exists():
        log(f"WARNING: {table_file} not found, skipping table update")
        return

    # Build faithfulness table section
    lines = [
        "",
        "## Faithfulness (Qwen2.5-VL-72B Judge)",
        "",
        "| Method | " + " | ".join(DATASETS) + " | 평균 Faithfulness |",
        "|---|" + "---:|" * (len(DATASETS) + 1),
    ]
    for method_token, label in METHODS:
        md = faith.get(method_token, {})
        vals = [md.get(ds) for ds in DATASETS]
        valid = [v for v in vals if v is not None]
        avg = sum(valid) / len(valid) if valid else None

        def fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"

        lines.append(
            f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(avg)} |"
        )

    faith_section = "\n".join(lines) + "\n"
    existing = table_file.read_text(encoding="utf-8")

    # Remove old faithfulness section if present
    marker = "\n## Faithfulness"
    if marker in existing:
        existing = existing[:existing.index(marker)]

    table_file.write_text(existing.rstrip("\n") + "\n" + faith_section, encoding="utf-8")
    log(f"Updated {table_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithfulness evaluation with Qwen2.5-VL-72B judge.")
    parser.add_argument("--result-dir", required=True, help="Path to result dir (e.g. results/comparison_v2)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHOD_TOKENS,
                        choices=["image_only", "ocr_text_only", "closed_text_only",
                                 "ocr_text_image", "selective_llm",
                                 "selective_upstage", "closed_text_image"])
    parser.add_argument("--physical-gpus", nargs="+", type=int, default=[2, 3],
                        help="Physical GPU indices to watch for VRAM watchdog")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit predictions per method/dataset (smoke test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    if not result_dir.exists():
        raise SystemExit(f"result_dir not found: {result_dir}")

    log(f"Faithfulness eval: {result_dir.name}")
    log(f"Datasets: {args.datasets}")
    log(f"Methods:  {args.methods}")

    # Spawn VRAM watchdogs for physical GPUs
    flag_dir = result_dir / "faithfulness"
    flag_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    watchdogs = [spawn_watchdog(g, pid, flag_dir) for g in args.physical_gpus]

    # Load judge model
    model, processor = load_judge_model()

    all_summaries: list[dict] = []
    for method_token in args.methods:
        for dataset in args.datasets:
            log(f"=== {dataset} / {method_token} ===")
            summary = evaluate_dataset_method(result_dir, dataset, method_token, model, processor, limit=args.limit)
            if summary:
                all_summaries.append(summary)

    # Save summary JSON
    summary_path = result_dir / "faithfulness_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    log(f"Saved summary to {summary_path}")

    # Update table.md
    update_table_with_faithfulness(result_dir, all_summaries)

    # Clean up watchdogs
    for w in watchdogs:
        w.terminate()

    log("Done.")


if __name__ == "__main__":
    main()

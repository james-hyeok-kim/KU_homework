"""
5-Way Comparison aligned to Chaemin's AI-Final-project conditions.

Methods (--mode):
  image_only      : Baseline — page images only                     (Method 1)
  ocr_text_only   : Qwen self-OCR text only, no images              (Method 2)
  ocr_text_image  : Qwen self-OCR text + page images                (Method 4)
  selective_llm   : LLM page classify -> chart: image / text,mixed: OCR text  (Method 5)

Method 3 (Closed-VDU / Upstage) reuses Chaemin's parsed_text_only result JSONs
directly — no GPU run needed (see run_comparison_v2.sh).

Alignment with Chaemin:
  - Same samples: qid/query/answer/docids taken from Chaemin's image_only
    result JSON (first100, oracle qrels, topk=1).
  - Same generator: Qwen2-VL-7B-Instruct, bitsandbytes NF4 4-bit.
  - Same metric: relaxed_exact_match (imported from Chaemin's benchmark.metrics).
  - Same answer prompt: ported verbatim from benchmark/v12_on_visrag.py.
  - Same image cap: max_pixels=1280*1024, min_pixels=256*256.
  - Same max_new_tokens=20 for answers.

OCR / classification prompts are ported verbatim from my previous experiment
(/data/jameskimh/visrag_experiment/src/evaluate_docparse_v2.py) so that
"OCR(Open-영혁-Hybrid)" means the same OCR engine as Exp 002/003.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]                    # visrag_research/
CHAEMIN = ROOT.parent / "AI-Final-project_chaemin"            # algorithm_for_AI/AI-Final-project_chaemin
sys.path.insert(0, str(CHAEMIN))
from benchmark.metrics import accuracy, write_json  # noqa: E402  (Chaemin's relaxed EM)

LOCAL_DATA = Path("/data/jameskimh/visrag_experiment/data/test")
LOCAL_MODEL = "/data/jameskimh/visrag_experiment/models/Qwen2-VL-7B-Instruct"

# Dataset name -> local parquet dir
LOCAL_DIR = {"InfoVQA": "infovqa", "ChartQA": "chartqa", "MP-DocVQA": "docvqa", "SlideVQA": "slidevqa"}

MODES = ("image_only", "ocr_text_only", "ocr_text_image", "selective_llm", "selective_upstage")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Ported verbatim from previous experiment (evaluate_docparse_v2.py).
OCR_PROMPT = (
    "Extract all text content from this document image. "
    "Preserve the structure: use headings, bullet points, and paragraphs as in the original. "
    "Output only the extracted text — no explanations or commentary."
)

ROUTE_PROMPT = (
    "Look at this document page image. "
    "Classify it into one of these types:\n"
    "  chart  — the page is primarily a chart, graph, or data visualization\n"
    "  text   — the page is primarily plain readable text (no heavy visual encoding)\n"
    "  mixed  — the page has both significant text and visual/infographic elements\n"
    "Output exactly one word: chart, text, or mixed. No explanation."
)


def build_answer_prompt(query: str, parsed_context: str, with_images: bool, with_text: bool) -> str:
    """Chaemin's build_prompt (v12_on_visrag.py) generalised over evidence flags."""
    if with_images and with_text:
        evidence_instruction = "Use parsed text/table evidence for exact labels and values, and use images to verify visual/layout evidence."
        context_block = parsed_context or "(not provided)"
    elif with_text:
        evidence_instruction = "Use only the parsed text/table evidence. No images are provided."
        context_block = parsed_context or "(not provided)"
    else:
        evidence_instruction = "Use only the provided document page image(s)."
        context_block = "(not provided)"

    return f"""You are evaluating a document QA method on the VisRAG benchmark.
{evidence_instruction}
Answer only the final answer. For numeric answers, return the exact visible value when possible.
If the evidence is insufficient, answer: insufficient to answer.

[Parsed text/table evidence]
{context_block}

[Question]
{query}
"""


# ---------------------------------------------------------------------------
# Model (NF4) loading — mirrors Chaemin's call_qwen2vl(use_bnb4=True)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def load_model(model_path: str = LOCAL_MODEL):
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    # SKIP_VISUAL_QUANT=1 keeps the vision tower in fp16 (ablation; default off
    # to stay aligned with Chaemin's config which quantises everything).
    quant_kwargs: dict[str, Any] = {}
    if os.environ.get("SKIP_VISUAL_QUANT", "0") == "1":
        quant_kwargs["llm_int8_skip_modules"] = ["visual"]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        **quant_kwargs,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    # Same image-token cap as Chaemin's script.
    processor.image_processor.max_pixels = 1280 * 1024
    processor.image_processor.min_pixels = 256 * 256
    if model.generation_config is not None:
        for k in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, k):
                setattr(model.generation_config, k, None)
    _MODEL_CACHE["model"] = (model, processor)
    return model, processor


def vlm_generate(prompt: str, images: list[Image.Image], max_new_tokens: int) -> dict[str, Any]:
    """One Qwen2-VL inference call: prompt (+optional images) -> text."""
    from qwen_vl_utils import process_vision_info

    model, processor = load_model()
    parts: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    parts.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": parts}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    start = time.time()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = {
        "text": out_text.strip(),
        "elapsed_sec": round(time.time() - start, 3),
        "input_tokens": int(inputs.input_ids.numel()),
        "output_tokens": int(sum(t.numel() for t in trimmed)),
    }
    del inputs, generated, trimmed
    return result


# ---------------------------------------------------------------------------
# Caches (OCR text / page route) — keyed by corpus_id, shared across modes
# ---------------------------------------------------------------------------

class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.data[row["key"]] = row["value"]

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def load_upstage_cache(dataset: str) -> dict[str, str]:
    """Load Chaemin's pre-parsed Upstage VDU text, keyed by corpus_id."""
    path = CHAEMIN / "data" / "parsed" / f"{dataset}_first100_qrels_upstage.jsonl"
    cache: dict[str, str] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["corpus_id"]] = row["text"]
    return cache


def get_ocr_text(corpus_id: str, image: Image.Image, cache: JsonlCache) -> tuple[str, float]:
    cached = cache.get(corpus_id)
    if cached is not None:
        return cached, 0.0
    out = vlm_generate(OCR_PROMPT, [image], max_new_tokens=512)
    cache.put(corpus_id, out["text"])
    return out["text"], out["elapsed_sec"]


def get_route(corpus_id: str, image: Image.Image, cache: JsonlCache) -> tuple[str, float]:
    cached = cache.get(corpus_id)
    if cached is not None:
        return cached, 0.0
    out = vlm_generate(ROUTE_PROMPT, [image], max_new_tokens=8)
    label = out["text"].strip().lower()
    # normalise to one of the three labels (fallback: mixed)
    for cand in ("chart", "text", "mixed"):
        if cand in label:
            label = cand
            break
    else:
        label = "mixed"
    cache.put(corpus_id, label)
    return label, out["elapsed_sec"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chaemin_samples(dataset: str) -> list[dict[str, Any]]:
    """Sample list (qid/query/answer/docids) from Chaemin's image_only result JSON."""
    path = CHAEMIN / "results" / "v12_on_visrag" / "today" / f"{dataset}_image_only_top1_first100.json"
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        {"qid": r["qid"], "query": r["query"], "answer": r["answer"], "docids": r["docids"]}
        for r in payload["rows"]
    ]


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
        raise RuntimeError(f"{dataset}: {len(missing)} corpus images missing, e.g. {sorted(missing)[:3]}")
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="5-way comparison run (Chaemin-aligned).")
    parser.add_argument("--dataset", required=True, choices=list(LOCAL_DIR))
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on #samples (smoke test).")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_chaemin_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    needed = {d for s in samples for d in s["docids"]}
    print(f"[{args.dataset}/{args.mode}] {len(samples)} queries, {len(needed)} corpus images")
    corpus = load_corpus_images(args.dataset, needed)

    # COMPARISON_RESULTS_DIR lets a parallel suite (e.g. transformers 4.51 venv)
    # keep its own results + OCR/route caches, so environments never mix.
    results_root = Path(os.environ.get("COMPARISON_RESULTS_DIR", ROOT / "results" / "comparison_v2"))
    cache_dir = results_root / "cache"
    ocr_cache = JsonlCache(cache_dir / f"ocr_{args.dataset}.jsonl")
    route_cache = JsonlCache(cache_dir / f"route_{args.dataset}.jsonl")

    upstage_text = load_upstage_cache(args.dataset) if args.mode == "selective_upstage" else {}

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        images = [corpus[d] for d in s["docids"]]
        row: dict[str, Any] = {
            "dataset": args.dataset,
            "qid": s["qid"],
            "query": s["query"],
            "answer": s["answer"],
            "docids": s["docids"],
            "method": args.mode,
            "generator_backend": "qwen2vl7b_bnb4",
            "generator_model": "Qwen/Qwen2-VL-7B-Instruct",
            "retrieval_source": "chaemin_first100_oracle",
        }
        ocr_sec = 0.0

        if args.mode == "image_only":
            prompt = build_answer_prompt(s["query"], "", with_images=True, with_text=False)
            gen_images = images

        elif args.mode in ("ocr_text_only", "ocr_text_image"):
            chunks = []
            for docid, img in zip(s["docids"], images):
                text, sec = get_ocr_text(docid, img, ocr_cache)
                ocr_sec += sec
                chunks.append(f"[Document {docid}]\n{text}")
            context = "\n\n".join(chunks)
            with_imgs = args.mode == "ocr_text_image"
            prompt = build_answer_prompt(s["query"], context, with_images=with_imgs, with_text=True)
            gen_images = images if with_imgs else []
            row["ocr_context_available"] = bool(context.strip())

        elif args.mode == "selective_llm":  # chart -> image, text/mixed -> OCR text
            routes, chunks, gen_images = [], [], []
            for docid, img in zip(s["docids"], images):
                label, sec = get_route(docid, img, route_cache)
                ocr_sec += sec
                routes.append(label)
                if label == "chart":
                    gen_images.append(img)
                else:
                    text, sec2 = get_ocr_text(docid, img, ocr_cache)
                    ocr_sec += sec2
                    chunks.append(f"[Document {docid}]\n{text}")
            context = "\n\n".join(chunks)
            prompt = build_answer_prompt(
                s["query"], context,
                with_images=bool(gen_images), with_text=bool(chunks),
            )
            row["routes"] = routes
            row["ocr_context_available"] = bool(chunks)

        else:  # selective_upstage: chart -> image, text/mixed -> Upstage VDU text
            routes, chunks, gen_images = [], [], []
            for docid, img in zip(s["docids"], images):
                label, sec = get_route(docid, img, route_cache)
                ocr_sec += sec
                routes.append(label)
                if label == "chart":
                    gen_images.append(img)
                else:
                    text = upstage_text.get(docid, "")
                    if text:
                        chunks.append(f"[Document {docid}]\n{text}")
            context = "\n\n".join(chunks)
            prompt = build_answer_prompt(
                s["query"], context,
                with_images=bool(gen_images), with_text=bool(chunks),
            )
            row["routes"] = routes
            row["upstage_context_available"] = bool(chunks)

        out = vlm_generate(prompt, gen_images, max_new_tokens=args.max_new_tokens)
        row.update({
            "prediction": out["text"],
            "elapsed_sec": out["elapsed_sec"],
            "ocr_elapsed_sec": round(ocr_sec, 3),
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "n_images": len(gen_images),
        })
        rows.append(row)
        extra = f" routes={row.get('routes')}" if "routes" in row else ""
        print(f"[{i + 1}/{len(samples)}] {s['qid']}: imgs={len(gen_images)}{extra} pred={out['text'][:60]!r}")

    summary = accuracy(rows, pred_key="prediction", answer_key="answer")
    payload = {"summary": summary, "rows": rows}
    out_path = Path(args.output or results_root / f"{args.dataset}_{args.mode}.json")
    write_json(out_path, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

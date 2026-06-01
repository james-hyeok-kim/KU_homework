"""
FLUX sampling script for FFHQ-like face generation
Uses FLUX.1-schnell (4 NFE) and FLUX.1-dev (28 NFE) for comparison
Saves to /data/jameskimh/final_project/samples/flux_{model}/
"""

import os
import argparse
import time
import torch
from diffusers import FluxPipeline
from PIL import Image

FLUX_SCHNELL = "/data/jameskimh/flux_pretrained/FLUX.1-schnell"
FLUX_DEV = "/data/jameskimh/flux_pretrained/FLUX.1-dev"
SAMPLE_BASE = "/data/jameskimh/final_project/samples"

FACE_PROMPTS = [
    "a high quality portrait photo of a person, photorealistic, 64x64",
    "portrait of a young woman, professional photography, neutral background",
    "portrait of a man with natural lighting, photorealistic",
    "headshot photo of a person, clear face, high resolution",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["schnell", "dev"], default="schnell")
    p.add_argument("--nfe", type=int, default=None,
                   help="Override NFE (default: 4 for schnell, 28 for dev)")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--skip_fid", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = FLUX_SCHNELL if args.model == "schnell" else FLUX_DEV
    default_nfe = 4 if args.model == "schnell" else 28
    n_steps = args.nfe if args.nfe is not None else default_nfe
    out_dir = os.path.join(SAMPLE_BASE, f"flux_{args.model}_nfe{n_steps}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading FLUX.1-{args.model} from {model_path}...")
    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)

    print(f"Generating {args.n_samples} samples (NFE={n_steps})...")
    t0 = time.time()
    idx = 0
    prompt_cycle = 0

    while idx < args.n_samples:
        bs = min(args.batch_size, args.n_samples - idx)
        prompt = FACE_PROMPTS[prompt_cycle % len(FACE_PROMPTS)]
        prompts = [prompt] * bs

        with torch.no_grad():
            images = pipe(
                prompts,
                num_inference_steps=n_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                output_type="pil",
            ).images

        for img in images:
            img.save(os.path.join(out_dir, f"sample_{idx:06d}.png"))
            idx += 1
        prompt_cycle += 1

        if idx % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {idx}/{args.n_samples} | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"Done: {args.n_samples} samples in {elapsed:.1f}s")
    print(f"NFE per sample: {n_steps}")
    print(f"Samples saved → {out_dir}")

    if not args.skip_fid:
        print("Computing FID...")
        try:
            from cleanfid import fid
            import json
            data_root = "/data/jameskimh/final_project/data/ffhq64"
            score = fid.compute_fid(out_dir, fdir2=data_root, num_gen=args.n_samples)
            print(f"FID = {score:.4f}")
            result = {"fid": score, "nfe": n_steps, "time_s": round(elapsed, 1)}
            log_path = f"/data/jameskimh/final_project/logs/flux_{args.model}_nfe{n_steps}_fid.json"
            with open(log_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"FID saved → {log_path}")
        except Exception as e:
            print(f"FID computation failed: {e}")


if __name__ == "__main__":
    main()

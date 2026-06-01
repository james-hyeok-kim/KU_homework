"""
DDIM / DDPM sampling script.
Uses google/ddpm-celebahq-256 (113M params), outputs downsampled to 64x64.
Generates 5,000 samples per variant and computes FID vs FFHQ-64.

Usage:
  python sample_ddim.py                          # all variants
  python sample_ddim.py --variants ddim_10 ddim_50
  python sample_ddim.py --skip_fid               # sampling only
"""

import os, argparse, time, json
import torch
from PIL import Image
from diffusers import DDPMPipeline, DDIMScheduler

DATA_ROOT   = "/data/jameskimh/final_project/data/ffhq64"
SAMPLE_BASE = "/data/jameskimh/final_project/samples"
LOG_DIR     = "/data/jameskimh/final_project/logs"
MODEL_ID    = "google/ddpm-celebahq-256"
TARGET_SIZE = 64

VARIANTS = {
    "ddim_10":  {"nfe": 10,  "stochastic": False},
    "ddim_20":  {"nfe": 20,  "stochastic": False},
    "ddim_50":  {"nfe": 50,  "stochastic": False},
    "ddpm_100": {"nfe": 100, "stochastic": True},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--skip_fid", action="store_true")
    return p.parse_args()


def generate(pipe, scheduler, nfe, stochastic, n_samples, batch_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    idx = existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if existing >= n_samples:
        print(f"  Already have {existing} samples, skipping generation.")
        return

    t0 = time.time()
    while idx < n_samples:
        bs = min(batch_size, n_samples - idx)
        with torch.no_grad():
            out = pipe(
                batch_size=bs,
                num_inference_steps=nfe,
                output_type="pil",
            )
        for img in out.images:
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
            img.save(os.path.join(out_dir, f"sample_{idx:06d}.png"))
            idx += 1
        if idx % 500 == 0:
            print(f"  {idx}/{n_samples} | {time.time()-t0:.0f}s")

    print(f"  Generated {n_samples} samples in {time.time()-t0:.1f}s")
    return time.time() - t0


def compute_fid(gen_dir, data_root, n_samples):
    from cleanfid import fid
    return fid.compute_fid(gen_dir, fdir2=data_root, num_gen=n_samples)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {MODEL_ID}...")
    base_pipe = DDPMPipeline.from_pretrained(MODEL_ID).to(device)

    results = {}
    for name in args.variants:
        cfg = VARIANTS[name]
        nfe, stochastic = cfg["nfe"], cfg["stochastic"]
        print(f"\n=== {name.upper()} (NFE={nfe}, {'DDPM' if stochastic else 'DDIM'}) ===")

        if not stochastic:
            base_pipe.scheduler = DDIMScheduler.from_config(
                base_pipe.scheduler.config
            )

        out_dir = os.path.join(SAMPLE_BASE, f"{name.replace('_', '_nfe')}")
        t_elapsed = generate(base_pipe, base_pipe.scheduler, nfe,
                             stochastic, args.n_samples, args.batch_size, out_dir)

        fid_score = None
        if not args.skip_fid:
            print(f"  Computing FID...")
            fid_score = compute_fid(out_dir, DATA_ROOT, args.n_samples)
            print(f"  FID = {fid_score:.4f}")

        results[name] = {"fid": fid_score, "nfe": nfe, "time_s": round(t_elapsed, 1) if t_elapsed else None}

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, "ddim_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

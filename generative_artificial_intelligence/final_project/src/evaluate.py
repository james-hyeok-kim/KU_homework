"""
Evaluation: NLL (bits/dim) + FID for GLOW and FLUX
Results saved to /data/jameskimh/final_project/experiments/results/
"""

import os
import argparse
import time
import torch
import torchvision.utils as vutils
import numpy as np
from model import Glow
from dataset import get_loaders

RESULT_DIR = "/data/jameskimh/final_project/experiments/results"
FLUX_SAMPLE_DIR = "/data/jameskimh/final_project/samples/flux"
GLOW_SAMPLE_DIR = "/data/jameskimh/final_project/samples/glow"
DATA_ROOT = "/data/jameskimh/final_project/data/ffhq64"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="/data/jameskimh/final_project/glow_pretrained/glow_v2_ffhq64_030000.pt")
    p.add_argument("--n_fid_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--skip_fid", action="store_true")
    return p.parse_args()


def compute_nll(model, loader, device, max_batches=50):
    model.eval()
    nlls = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            nll = model.nll_loss(x)
            nlls.append(nll.item())
    return np.mean(nlls)


def generate_glow_samples(model, n, batch_size, temperature, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    generated = 0
    idx = 0
    t0 = time.time()
    with torch.no_grad():
        while generated < n:
            bs = min(batch_size, n - generated)
            samples = model.sample(bs, temperature=temperature, device=device)
            samples = (samples + 0.5).clamp(0, 1)
            for s in samples:
                path = os.path.join(out_dir, f"sample_{idx:06d}.png")
                vutils.save_image(s, path)
                idx += 1
            generated += bs
    elapsed = time.time() - t0
    nfe = 1  # GLOW: single forward pass
    print(f"  GLOW generated {n} samples in {elapsed:.1f}s (NFE=1 per sample)")
    return elapsed, nfe


def compute_fid_from_dir(gen_dir, data_root, n_samples):
    try:
        from cleanfid import fid
        score = fid.compute_fid(gen_dir, dataset_name=None, dataset_res=64,
                                dataset_split="custom", num_gen=n_samples,
                                fdir2=data_root)
        return score
    except Exception as e:
        print(f"  FID computation failed: {e}")
        return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading GLOW model...")
    model = Glow(in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded: iter={ckpt['iteration']}, train_loss={ckpt['loss']:.4f}")

    print("\n[1] NLL evaluation (validation set)...")
    _, val_loader = get_loaders(DATA_ROOT, batch_size=args.batch_size)
    nll = compute_nll(model, val_loader, device)
    print(f"  GLOW NLL: {nll:.4f} bits/dim")

    print("\n[2] Generating GLOW samples for FID...")
    t_glow, nfe_glow = generate_glow_samples(
        model, args.n_fid_samples, args.batch_size, args.temperature, device, GLOW_SAMPLE_DIR
    )

    fid_glow = None
    if not args.skip_fid:
        print("\n[3] Computing FID (GLOW)...")
        fid_glow = compute_fid_from_dir(GLOW_SAMPLE_DIR, DATA_ROOT, args.n_fid_samples)
        if fid_glow is not None:
            print(f"  GLOW FID: {fid_glow:.2f}")

    # NFE comparison summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'NLL (bits/dim)':<18} {'FID':<10} {'NFE':<8} {'Time(s)'}")
    print(f"{'GLOW':<12} {nll:<18.4f} {str(fid_glow) if fid_glow else 'N/A':<10} {nfe_glow:<8} {t_glow:.1f}")
    print("Note: FLUX NFE = number of denoising steps (e.g. 28 for dev, 4 for schnell)")

    # Save results
    result_path = os.path.join(RESULT_DIR, "eval_results.txt")
    with open(result_path, "w") as f:
        f.write(f"GLOW checkpoint: {args.ckpt}\n")
        f.write(f"GLOW NLL: {nll:.4f} bits/dim\n")
        f.write(f"GLOW FID: {fid_glow}\n")
        f.write(f"GLOW NFE: {nfe_glow}\n")
        f.write(f"GLOW sampling time ({args.n_fid_samples} samples): {t_glow:.1f}s\n")
    print(f"\nResults saved → {result_path}")


if __name__ == "__main__":
    main()

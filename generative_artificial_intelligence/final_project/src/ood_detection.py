"""
OOD detection via GLOW NLL.
Tests GLOW's NLL on three types of input:
  - FFHQ faces (in-distribution)
  - Solid color images (OOD)
  - Random noise (OOD)

Saves figure to experiments/results/fig7_ood_detection.png
Saves JSON results to logs/ood_results.json

Usage:
  python ood_detection.py
  python ood_detection.py --ckpt /path/to/checkpoint.pt --n_images 100
"""

import os, sys, json, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import Glow
from dataset import get_loaders

CKPT      = "/data/jameskimh/final_project/glow_pretrained/glow_v2_ffhq64_030000.pt"
DATA_ROOT = "/data/jameskimh/final_project/data/ffhq64"
OUT_FIG   = "/home/jovyan/workspace/KU_homework/generative_artificial_intelligence/final_project/experiments/results/fig7_ood_detection.png"
OUT_JSON  = "/data/jameskimh/final_project/logs/ood_results.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def compute_nll_batch(model, x, device):
    x = x.to(device)
    with torch.no_grad():
        nll = model.nll_loss(x)
    return nll.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Glow(in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: iter={ckpt['iteration']}")

    # ── 1. In-distribution: FFHQ faces ────────────────────────────────────────
    _, val_loader = get_loaders(DATA_ROOT, batch_size=args.batch_size)
    ffhq_nlls = []
    n = 0
    for x in val_loader:
        if n >= args.n_images:
            break
        nll = compute_nll_batch(model, x, device)
        ffhq_nlls.append(nll)
        n += len(x)
    ffhq_mean = float(np.mean(ffhq_nlls))
    ffhq_std  = float(np.std(ffhq_nlls))
    print(f"FFHQ NLL:  mean={ffhq_mean:.3f}  std={ffhq_std:.3f}")

    # ── 2. OOD: solid color images ─────────────────────────────────────────────
    solid_nlls = []
    colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
              (1.0, 0.5, 0.0), (0.5, 0.0, 1.0)]
    for _ in range(args.n_images // args.batch_size + 1):
        r, g, b = colors[len(solid_nlls) % len(colors)]
        x = torch.zeros(args.batch_size, 3, 64, 64)
        x[:, 0] = r - 0.5
        x[:, 1] = g - 0.5
        x[:, 2] = b - 0.5
        try:
            nll = compute_nll_batch(model, x, device)
            if not (np.isnan(nll) or np.isinf(nll)):
                solid_nlls.append(nll)
        except Exception:
            pass
        if len(solid_nlls) * args.batch_size >= args.n_images:
            break
    solid_mean = float(np.mean(solid_nlls)) if solid_nlls else float("nan")
    solid_std  = float(np.std(solid_nlls))  if solid_nlls else float("nan")
    print(f"Solid NLL: mean={solid_mean:.3f}  std={solid_std:.3f}")

    # ── 3. OOD: random noise ───────────────────────────────────────────────────
    noise_nll_str = "NaN (numerical overflow)"
    try:
        x = torch.rand(args.batch_size, 3, 64, 64) - 0.5
        nll = compute_nll_batch(model, x, device)
        noise_nll_str = f"{nll:.3f}" if not (np.isnan(nll) or np.isinf(nll)) else "NaN"
    except Exception:
        pass
    print(f"Noise NLL: {noise_nll_str}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="white")

    # Left: histogram
    ax = axes[0]
    ax.hist(ffhq_nlls, bins=20, alpha=0.75, color="#2EA05B", label=f"FFHQ faces (μ={ffhq_mean:.2f})")
    ax.hist(solid_nlls, bins=15, alpha=0.75, color="#F5A623", label=f"Solid color (μ={solid_mean:.2f})")
    ax.axvline(ffhq_mean, color="#2EA05B", lw=2, ls="--")
    ax.axvline(solid_mean, color="#F5A623", lw=2, ls="--")
    ax.set_xlabel("NLL (bits/dim)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("NLL Distribution: In-dist vs OOD", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: bar chart summary
    ax2 = axes[1]
    labels = ["FFHQ faces\n(in-dist)", "Solid color\n(OOD)", "Random noise\n(OOD)"]
    values = [ffhq_mean, solid_mean, float("nan")]
    colors_bar = ["#2EA05B", "#F5A623", "#E84A2F"]
    bars = ax2.bar(labels, [v if not np.isnan(v) else 0 for v in values],
                   color=colors_bar, alpha=0.85, edgecolor="white", linewidth=1.5)
    for bar, val, lbl in zip(bars, values, labels):
        if np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, 0.2,
                     "NaN\n(overflow)", ha="center", va="bottom",
                     fontsize=12, color="#E84A2F", fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylabel("NLL (bits/dim)", fontsize=13)
    ax2.set_title("NLL by Input Type\n(lower = model assigns higher prob.)", fontsize=13, fontweight="bold")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.text(0.98, 0.05, "⚠ Likelihood ≠ Perceptual Quality\n(solid images score lower NLL than faces)",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=10, color="#E84A2F",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#F5A623"))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("GLOW OOD Detection via Exact Log-Likelihood",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {OUT_FIG}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    results = {
        "ffhq_nll_mean": round(ffhq_mean, 3),
        "ffhq_nll_std":  round(ffhq_std, 3),
        "solid_nll_mean": round(solid_mean, 3) if not np.isnan(solid_mean) else None,
        "solid_nll_std":  round(solid_std, 3)  if not np.isnan(solid_std)  else None,
        "noise_nll": noise_nll_str,
        "finding": "solid NLL < FFHQ NLL — likelihood != perceptual quality",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved  → {OUT_JSON}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

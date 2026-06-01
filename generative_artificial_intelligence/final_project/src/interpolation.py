"""
GLOW latent space interpolation.
Encodes two real images A and B, linearly interpolates their latent vectors,
and decodes back to image space.

Saves figure to experiments/results/fig6_interpolation.png

Usage:
  python interpolation.py
  python interpolation.py --n_pairs 4 --alphas 0.0 0.25 0.5 0.75 1.0
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import Glow, unsqueeze
from dataset import get_loaders

CKPT      = "/data/jameskimh/final_project/glow_pretrained/glow_v2_ffhq64_030000.pt"
DATA_ROOT = "/data/jameskimh/final_project/data/ffhq64"
OUT_FIG   = "/home/jovyan/workspace/KU_homework/generative_artificial_intelligence/final_project/experiments/results/fig6_interpolation.png"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return p.parse_args()


def encode(model, x):
    with torch.no_grad():
        z_list, _, _ = model(x)
    return z_list


def decode(model, z_list):
    with torch.no_grad():
        x, _ = model.blocks[-1]._reverse(z_list[-1])
        for i in range(len(model.blocks) - 2, -1, -1):
            x = unsqueeze(x)
            x, _ = model.blocks[i]._reverse((x, z_list[i]))
        x = unsqueeze(x)
    return x.clamp(-0.5, 0.5)


def to_rgb(tensor):
    return (tensor.cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Glow(in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (iter={ckpt['iteration']})")

    # Load image pairs
    _, val_loader = get_loaders(DATA_ROOT, batch_size=args.n_pairs * 2)
    batch = next(iter(val_loader))[: args.n_pairs * 2].to(device)
    imgs_a = batch[: args.n_pairs]
    imgs_b = batch[args.n_pairs :]

    z_a = encode(model, imgs_a)
    z_b = encode(model, imgs_b)
    print(f"Encoded {args.n_pairs} pairs")

    # Build rows: pair x alpha
    rows = []
    for pi in range(args.n_pairs):
        row = []
        for alpha in args.alphas:
            z_interp = [
                za[pi : pi + 1] * (1 - alpha) + zb[pi : pi + 1] * alpha
                for za, zb in zip(z_a, z_b)
            ]
            img_t = decode(model, z_interp)[0]
            row.append(to_rgb(img_t))
        rows.append(row)

    # Plot
    n_cols = len(args.alphas)
    fig, axes = plt.subplots(
        args.n_pairs, n_cols,
        figsize=(n_cols * 1.5, args.n_pairs * 1.65),
        facecolor="#111111",
    )
    fig.suptitle(
        "GLOW Latent Space Interpolation\n"
        "Real images A & B encoded to z, linearly interpolated, decoded back",
        color="white", fontsize=11, y=0.99,
    )

    for pi, row in enumerate(rows):
        for ai, (alpha, img) in enumerate(zip(args.alphas, row)):
            ax = axes[pi][ai]
            ax.imshow(img)
            ax.axis("off")
            if pi == 0:
                ax.set_title(f"α={alpha:.1f}", color="white", fontsize=9, pad=3)
            if ai == 0:
                ax.set_ylabel("Face A", color="#4CAF50", fontsize=9,
                              rotation=0, labelpad=28, va="center")
            if ai == n_cols - 1:
                ax.text(1.05, 0.5, "Face B", transform=ax.transAxes,
                        color="#FF7043", fontsize=9, va="center", ha="left")

    fig.text(
        0.5, 0.01,
        "← Face A (α=0.0)          Interpolation direction →          Face B (α=1.0) →",
        ha="center", color="#AAAAAA", fontsize=9,
    )
    plt.subplots_adjust(wspace=0.04, hspace=0.06, top=0.91, bottom=0.04,
                        left=0.08, right=0.93)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, facecolor="#111111", bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()

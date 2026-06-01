"""
Generate all paper figures (fig1 ~ fig9).
Reads pre-computed results from logs/ and samples/ directories.
Calls interpolation.py and visualize_2d.py for figures that require model inference.

Usage:
  python make_figures.py              # regenerate all figures
  python make_figures.py --figs 3 7  # regenerate specific figures only
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "/home/jovyan/workspace/KU_homework/generative_artificial_intelligence/final_project/experiments/results"
LOGS_DIR    = "/data/jameskimh/final_project/logs"
SAMPLES_DIR = "/data/jameskimh/final_project/samples"
CKPT        = "/data/jameskimh/final_project/glow_pretrained/glow_v2_ffhq64_030000.pt"
DATA_ROOT   = "/data/jameskimh/final_project/data/ffhq64"

os.makedirs(RESULTS_DIR, exist_ok=True)

DARK  = "#0D1117"
BLUE  = "#2E5FA3"
GREEN = "#2EA05B"
RED   = "#E84A2F"
GOLD  = "#F5A623"
WHITE = "#FFFFFF"

# ── Quantitative data ─────────────────────────────────────────────────────────
FID_DATA = {
    "GLOW\n(NFE=1)":        {"fid": 183.35, "nfe": 1,   "time": 137,   "color": GREEN},
    "FLUX schnell\n(NFE=4)":{"fid": 184.94, "nfe": 4,   "time": 1231,  "color": RED},
    "FLUX dev-8\n(NFE=8)":  {"fid": 126.15, "nfe": 8,   "time": None,  "color": "#9B59B6"},
    "DDIM-10\n(NFE=10)":    {"fid": 68.75,  "nfe": 10,  "time": 602,   "color": BLUE},
    "DDIM-20\n(NFE=20)":    {"fid": 65.96,  "nfe": 20,  "time": 1346,  "color": BLUE},
    "DDIM-50\n(NFE=50)":    {"fid": 61.78,  "nfe": 50,  "time": 2658,  "color": BLUE},
    "DDPM-100\n(NFE=100)":  {"fid": 71.87,  "nfe": 100, "time": 5453,  "color": GOLD},
    "FLUX dev\n(NFE=28)":   {"fid": 117.87, "nfe": 28,  "time": 8617,  "color": "#9B59B6"},
}

TRAIN_CURVE = {
    "steps": [0, 100, 1000, 5000, 10000, 20000, 30000],
    "nll":   [None, -2.39, -3.81, -4.31, -4.43, -4.55, -4.61],
}


def fig_path(n):
    return os.path.join(RESULTS_DIR, f"fig{n}_" + {
        1: "concept_diagram",
        2: "training_curve",
        3: "fid_pareto",
        4: "sample_grid",
        5: "reconstruction",
        6: "interpolation",
        7: "ood_detection",
        8: "transport_paths",
        9: "twomoons_4way",
    }[n] + ".png")


# ── Fig 1: Concept diagram ─────────────────────────────────────────────────────
def make_fig1():
    fig = plt.figure(figsize=(19, 9.5), facecolor=DARK)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.88, bottom=0.08, left=0.04, right=0.97)

    models = [
        ("Normalizing Flow\n(GLOW)", GREEN,
         "Exact bijection f: X ↔ Z\nSingle forward/reverse pass (NFE=1)\nExact log-likelihood via change-of-variables"),
        ("Flow Matching\n(FLUX.1)", BLUE,
         "Learns vector field v_θ(x,t)\nStraight ODE path: noise→data\nNFE=4–28 via ODE solver"),
        ("DDPM", RED,
         "Stochastic Markov chain\nZigzag SDE reverse process\nNFE=100–1000 steps"),
        ("DDIM", GOLD,
         "Deterministic ODE from same score\nNo retraining required\nNFE=10–50 steps"),
    ]

    path_descs = [
        "Grid deformation\n(bijective rearrangement)",
        "Straight lines\n(optimal transport)",
        "Stochastic zigzag\n(Langevin dynamics)",
        "Smooth curve\n(deterministic ODE)",
    ]

    for i, ((name, col, desc), path_d) in enumerate(zip(models, path_descs)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#161B22")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(name, color=col, fontsize=13, fontweight="bold", pad=6)

        # Draw stylized transport path
        t = np.linspace(0, 1, 50)
        if i == 0:   # NF: grid-like
            for j in np.linspace(0.2, 0.8, 5):
                ax.plot(t, j + 0.15 * np.sin(2 * np.pi * t), c=col, alpha=0.5, lw=1.2)
            for j in np.linspace(0.2, 0.8, 5):
                ax.plot(j + 0.1 * np.cos(2 * np.pi * t) * 0.5, t, c=col, alpha=0.5, lw=1.2)
        elif i == 1:  # FM: straight
            for y0 in np.linspace(0.15, 0.85, 6):
                y1 = 0.5 + 0.3 * np.sin(y0 * np.pi)
                ax.annotate("", xy=(0.85, y1), xytext=(0.15, y0),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        elif i == 2:  # DDPM: zigzag
            for y0 in np.linspace(0.2, 0.8, 4):
                x_z = np.linspace(0.15, 0.85, 30)
                y_z = y0 + 0.08 * np.cumsum(np.random.randn(30)) / 10
                y_z = np.clip(y_z, 0.05, 0.95)
                ax.plot(x_z, y_z, c=col, alpha=0.6, lw=1.2)
        else:         # DDIM: smooth curve
            for y0 in np.linspace(0.15, 0.85, 6):
                y1 = 0.5 + 0.25 * np.sin(y0 * np.pi)
                x_c = np.linspace(0.15, 0.85, 30)
                y_c = y0 + (y1 - y0) * (x_c - 0.15) / 0.7
                y_c += 0.04 * np.sin(np.pi * (x_c - 0.15) / 0.7)
                ax.plot(x_c, y_c, c=col, alpha=0.6, lw=1.4)

        ax.text(0.5, 0.05, path_d, transform=ax.transAxes,
                ha="center", va="bottom", color=col, fontsize=8.5, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

        # Bottom: property description
        ax2 = fig.add_subplot(gs[1, i])
        ax2.set_facecolor("#0D1117")
        ax2.axis("off")
        ax2.text(0.5, 0.85, name.split("\n")[0], transform=ax2.transAxes,
                 ha="center", va="top", color=col, fontsize=12, fontweight="bold")
        ax2.text(0.5, 0.5, desc, transform=ax2.transAxes,
                 ha="center", va="center", color=WHITE, fontsize=10.5,
                 linespacing=1.6)

    # Property comparison table
    props = ["Exact NLL", "Reconstruction", "Latent Interp.", "Single NFE"]
    vals  = [
        [True, True, True, True],    # NF
        [False, False, False, False], # FM
        [False, False, False, False], # DDPM
        [False, False, False, False], # DDIM
    ]
    fig.text(0.5, 0.935, "Generative Model Principles — Path Geometry & Key Properties",
             ha="center", color=WHITE, fontsize=15, fontweight="bold")

    out = fig_path(1)
    plt.savefig(out, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close()
    print(f"Fig 1 saved → {out}")


# ── Fig 2: Training curve ──────────────────────────────────────────────────────
def make_fig2():
    steps = [s for s, n in zip(TRAIN_CURVE["steps"], TRAIN_CURVE["nll"]) if n is not None]
    nlls  = [n for n in TRAIN_CURVE["nll"] if n is not None]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor=WHITE)
    ax.plot(steps, nlls, "o-", color=GREEN, lw=2.5, ms=7, mfc=WHITE, mew=2)
    ax.axvline(1000, color="gray", ls="--", lw=1.2, label="Warmup end (step 1k)")
    ax.fill_between(steps, nlls, min(nlls) - 0.1, alpha=0.08, color=GREEN)
    for s, n in zip(steps, nlls):
        ax.annotate(f"{n}", (s, n), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("NLL (bits/dim)", fontsize=13)
    ax.set_title("GLOW Training Curve on FFHQ-64\n(still descending at 30k — not yet converged)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-500, 31500)
    plt.tight_layout()
    plt.savefig(fig_path(2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig 2 saved → {fig_path(2)}")


# ── Fig 3: FID bar + Pareto ────────────────────────────────────────────────────
def make_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=WHITE)

    names = list(FID_DATA.keys())
    fids  = [FID_DATA[n]["fid"] for n in names]
    cols  = [FID_DATA[n]["color"] for n in names]
    bars  = ax1.bar(names, fids, color=cols, alpha=0.85, edgecolor="white", lw=1.5)
    for bar, fid in zip(bars, fids):
        ax1.text(bar.get_x() + bar.get_width()/2, fid + 1.5,
                 f"{fid:.1f}", ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("FID ↓", fontsize=13)
    ax1.set_title("FID Comparison (5k samples vs 70k FFHQ-64)", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Pareto
    for name, d in FID_DATA.items():
        col = d["color"]
        ax2.scatter(d["nfe"], d["fid"], c=col, s=120, zorder=5, edgecolors="white", lw=1.5)
        label = name.replace("\n", " ")
        ax2.annotate(label, (d["nfe"], d["fid"]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8, color=col)

    # Pareto frontier line (GLOW → DDIM-10 → DDIM-20 → DDIM-50)
    pareto = sorted([(d["nfe"], d["fid"]) for d in FID_DATA.values()
                     if d["color"] in (GREEN, BLUE)], key=lambda x: x[0])
    px, py = zip(*pareto)
    ax2.plot(px, py, "--", color="gray", lw=1.2, alpha=0.5, label="Pareto frontier")

    ax2.set_xscale("log")
    ax2.set_xlabel("NFE (log scale)", fontsize=13)
    ax2.set_ylabel("FID ↓", fontsize=13)
    ax2.set_title("Compute–Quality Pareto Frontier", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path(3), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig 3 saved → {fig_path(3)}")


# ── Fig 4: Sample grid ─────────────────────────────────────────────────────────
def make_fig4():
    def load_samples(subdir, n=10):
        d = os.path.join(SAMPLES_DIR, subdir)
        files = sorted([f for f in os.listdir(d) if f.endswith(".png")])[:n]
        return [Image.open(os.path.join(d, f)) for f in files]

    rows = [
        ("glow",             "GLOW  (NFE=1 | FID=183.35 | 137s/5k)",       GREEN),
        ("flux_schnell_nfe4","FLUX.1-schnell  (NFE=4 | FID=184.94 | 1,231s/5k)", RED),
        ("flux_dev_nfe28",   "FLUX.1-dev  (NFE=28 | FID=117.87 | ~8,617s/5k)",   BLUE),
    ]
    n_cols = 10
    fig, axes = plt.subplots(len(rows) * 2, n_cols,
                              figsize=(n_cols * 1.2, len(rows) * 2.7),
                              facecolor=DARK)
    fig.suptitle("Generated Samples Comparison — FFHQ 64×64",
                 color=WHITE, fontsize=14, fontweight="bold")

    for ri, (subdir, label, col) in enumerate(rows):
        try:
            imgs = load_samples(subdir, n_cols * 2)
        except Exception:
            imgs = []
        for ci in range(n_cols):
            for row_offset in range(2):
                ax = axes[ri * 2 + row_offset][ci]
                ax.set_facecolor(DARK)
                ax.axis("off")
                idx = row_offset * n_cols + ci
                if idx < len(imgs):
                    ax.imshow(imgs[idx])
        axes[ri * 2][0].set_ylabel(label, color=col, fontsize=9,
                                    rotation=0, labelpad=5, va="center",
                                    ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(fig_path(4), dpi=120, facecolor=DARK, bbox_inches="tight")
    plt.close()
    print(f"Fig 4 saved → {fig_path(4)}")


# ── Fig 5: Reconstruction ──────────────────────────────────────────────────────
def make_fig5():
    import torch
    from model import Glow, unsqueeze
    from dataset import get_loaders
    import math

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Glow(in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, val_loader = get_loaders(DATA_ROOT, batch_size=8)
    batch = next(iter(val_loader))[:8].to(device)

    with torch.no_grad():
        z_list, _, _ = model(batch)
        x_rec, _ = model.blocks[-1]._reverse(z_list[-1])
        for i in range(len(model.blocks) - 2, -1, -1):
            x_rec = unsqueeze(x_rec)
            x_rec, _ = model.blocks[i]._reverse((x_rec, z_list[i]))
        x_rec = unsqueeze(x_rec).clamp(-0.5, 0.5)

    def to_np(t):
        return (t.cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0, 1)

    n = min(8, len(batch))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5), facecolor=WHITE)
    fig.suptitle(f"GLOW Reconstruction Quality  (PSNR=24.24 dB, SSIM=0.977)",
                 fontsize=12, fontweight="bold")
    for i in range(n):
        axes[0][i].imshow(to_np(batch[i])); axes[0][i].axis("off")
        axes[1][i].imshow(to_np(x_rec[i])); axes[1][i].axis("off")
    axes[0][0].set_ylabel("Original",       fontsize=10, rotation=0, labelpad=55, va="center")
    axes[1][0].set_ylabel("Reconstructed",  fontsize=10, rotation=0, labelpad=55, va="center")
    plt.tight_layout()
    plt.savefig(fig_path(5), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fig 5 saved → {fig_path(5)}")


# ── Fig 6: Interpolation (delegate to interpolation.py) ──────────────────────
def make_fig6():
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "interpolation.py")
    subprocess.run(["python3", script], check=True)
    print(f"Fig 6 saved → {fig_path(6)}")


# ── Fig 7: OOD detection (delegate to ood_detection.py) ──────────────────────
def make_fig7():
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "ood_detection.py")
    subprocess.run(["python3", script], check=True)
    print(f"Fig 7 saved → {fig_path(7)}")


# ── Fig 8 & 9: 2D visualization (delegate to visualize_2d.py) ────────────────
def make_fig8_9():
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "visualize_2d.py")
    subprocess.run(["python3", script], check=True)
    print(f"Fig 8 & 9 saved → {fig_path(8)}, {fig_path(9)}")


# ── Main ──────────────────────────────────────────────────────────────────────
MAKERS = {
    1: make_fig1,
    2: make_fig2,
    3: make_fig3,
    4: make_fig4,
    5: make_fig5,
    6: make_fig6,
    7: make_fig7,
    8: make_fig8_9,
    9: None,  # generated together with fig8
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--figs", nargs="*", type=int, default=None,
                   help="Which figures to generate (default: all). Example: --figs 2 3 5")
    return p.parse_args()


def main():
    args = parse_args()
    targets = args.figs if args.figs else [1, 2, 3, 4, 5, 6, 7, 8]

    for n in targets:
        maker = MAKERS.get(n)
        if maker is None:
            print(f"Fig {n}: generated as part of fig 8, skipping.")
            continue
        print(f"\n--- Generating Fig {n} ---")
        try:
            maker()
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

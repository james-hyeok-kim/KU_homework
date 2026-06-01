"""
GLOW reconstruction quality: encode real images → decode → PSNR/SSIM
Tests exact invertibility of the flow model.
"""
import os, sys, math
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import Glow
from dataset import get_loaders

CKPT = "/data/jameskimh/final_project/glow_pretrained/glow_v2_ffhq64_030000.pt"
DATA_ROOT = "/data/jameskimh/final_project/data/ffhq64"
OUT_DIR = "/data/jameskimh/final_project/experiments/results"
N_IMAGES = 200

def psnr(x, y):
    mse = ((x - y) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def ssim_simple(x, y):
    """Per-channel mean SSIM, simplified (no windowing)."""
    c1, c2 = (0.01 * 1.0)**2, (0.03 * 1.0)**2
    mu_x, mu_y = x.mean(), y.mean()
    sig_x = ((x - mu_x)**2).mean()
    sig_y = ((y - mu_y)**2).mean()
    sig_xy = ((x - mu_x)*(y - mu_y)).mean()
    num = (2*mu_x*mu_y + c1) * (2*sig_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)
    return (num / den).item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Glow(in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {CKPT}  (iter={ckpt['iteration']})")

    _, val_loader = get_loaders(DATA_ROOT, batch_size=8)
    psnr_list, ssim_list, n = [], [], 0

    with torch.no_grad():
        for x_orig in val_loader:
            if n >= N_IMAGES:
                break
            x = x_orig.to(device)
            x_dq = x + torch.zeros_like(x)   # no dequantization for reconstruction

            # Encode: forward pass → z_list
            x_sq = __import__('model').squeeze(x_dq)
            z_list, log_det, log_pz = model(x_dq)

            # Decode: reverse pass from z_list
            # Reconstruct via model's reverse
            from model import squeeze, unsqueeze
            # last block reverse
            z_last = z_list[-1]
            x_rec, _ = model.blocks[-1]._reverse(z_last)
            for i in range(len(model.blocks) - 2, -1, -1):
                x_rec = unsqueeze(x_rec)
                z_split = z_list[i]
                x_rec, _ = model.blocks[i]._reverse((x_rec, z_split))
            x_rec = unsqueeze(x_rec)
            x_rec = x_rec.clamp(-0.5, 0.5)

            # Metrics in [0,1] range
            x0 = (x_dq + 0.5).clamp(0, 1)
            xr = (x_rec + 0.5).clamp(0, 1)
            for xi, xri in zip(x0, xr):
                psnr_list.append(psnr(xi, xri))
                ssim_list.append(ssim_simple(xi, xri))
            n += len(x0)
            if n % 40 == 0:
                print(f"  {n}/{N_IMAGES}  PSNR={np.mean(psnr_list):.2f}  SSIM={np.mean(ssim_list):.4f}")

    print("\n=== RECONSTRUCTION RESULTS ===")
    print(f"N = {len(psnr_list)} images")
    print(f"PSNR  mean={np.mean(psnr_list):.2f} dB  std={np.std(psnr_list):.2f}")
    print(f"SSIM  mean={np.mean(ssim_list):.4f}   std={np.std(ssim_list):.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "reconstruction_results.txt"), "w") as f:
        f.write(f"N={len(psnr_list)}\n")
        f.write(f"PSNR mean={np.mean(psnr_list):.4f} std={np.std(psnr_list):.4f}\n")
        f.write(f"SSIM mean={np.mean(ssim_list):.4f} std={np.std(ssim_list):.4f}\n")
    print("Saved → reconstruction_results.txt")

if __name__ == "__main__":
    main()

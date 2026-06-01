"""
GLOW training script — FFHQ-64x64
Checkpoints saved to /data/jameskimh/final_project/glow_pretrained/
Samples saved to /data/jameskimh/final_project/samples/glow_train/
"""

import os
import argparse
import torch
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast
from model import Glow
from dataset import get_loaders

CKPT_DIR = "/data/jameskimh/final_project/glow_pretrained"
SAMPLE_DIR = "/data/jameskimh/final_project/samples/glow_train"
DATA_ROOT = "/data/jameskimh/final_project/data/ffhq64"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--n_flows", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=1000,
                   help="linear warmup steps from lr/100 to lr")
    p.add_argument("--n_iter", type=int, default=200000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--n_sample", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume from")
    p.add_argument("--resume_latest", action="store_true",
                   help="auto-resume from latest checkpoint in CKPT_DIR")
    p.add_argument("--no_amp", action="store_true",
                   help="disable automatic mixed precision")
    return p.parse_args()


def save_samples(model, iteration, n, temperature, device):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    samples = model.sample(n, temperature=temperature, device=device)
    samples = samples + 0.5  # [-0.5,0.5] → [0,1]
    path = os.path.join(SAMPLE_DIR, f"sample_{iteration:06d}.png")
    vutils.save_image(samples, path, nrow=4, normalize=False)
    print(f"  Saved samples → {path}")


def find_latest_ckpt(prefix="glow_v2_ffhq64"):
    """Find latest checkpoint with the given prefix (avoids broken old checkpoints)."""
    if not os.path.isdir(CKPT_DIR):
        return None
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and f.startswith(prefix)]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f.split("_")[-1].replace(".pt", "")))
    return os.path.join(CKPT_DIR, ckpts[-1])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  AMP: {use_amp}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    train_loader, val_loader = get_loaders(DATA_ROOT, batch_size=args.batch_size)

    model = Glow(
        in_channels=3,
        n_blocks=args.n_blocks,
        n_flows=args.n_flows,
        hidden_channels=args.hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=use_amp)

    start_iter = 0
    resume_path = args.resume
    if args.resume_latest and resume_path is None:
        resume_path = find_latest_ckpt()

    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"]
        print(f"  Resumed at iter {start_iter}, loss={ckpt['loss']:.4f}")
    else:
        print("Training from scratch (no checkpoint found/specified)")

    model.train()
    data_iter = iter(train_loader)
    running_loss = 0.0

    def get_lr(iteration):
        if iteration <= args.warmup:
            return args.lr * max(iteration / args.warmup, 0.01)
        return args.lr

    for iteration in range(start_iter + 1, args.n_iter + 1):
        # linear warmup
        current_lr = get_lr(iteration)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x = next(data_iter)

        x = x.to(device, non_blocking=True)
        # dequantize: add uniform noise (8-bit → continuous)
        x = x + torch.rand_like(x) / 256.0

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            loss = model.nll_loss(x)

        if not torch.isfinite(loss):
            print(f"[{iteration:6d}] WARNING: loss={loss.item()}, skipping update")
            optimizer.zero_grad()
            running_loss += 0.0
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # skip update if gradients contain inf/nan after clipping
            grads_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if grads_ok:
                scaler.step(optimizer)
                running_loss += loss.item()
            else:
                print(f"[{iteration:6d}] WARNING: inf/nan grad, skipping update")
                optimizer.zero_grad()
                running_loss += 0.0
            scaler.update()

        if iteration % 100 == 0:
            avg = running_loss / 100
            running_loss = 0.0
            print(f"[{iteration:6d}/{args.n_iter}] loss={avg:.4f} bits/dim  lr={current_lr:.2e}", flush=True)

        if iteration % args.sample_every == 0:
            model.eval()
            save_samples(model, iteration, args.n_sample, args.temperature, device)
            model.train()

        if iteration % args.save_every == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"glow_v2_ffhq64_{iteration:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "loss": loss.item(),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()

# Experiment 001: GLOW vs FLUX.1-schnell on FFHQ-64×64

## Hypothesis

A normalizing flow model (GLOW) trained from scratch on a task-specific 64×64 dataset can
achieve competitive sample quality (FID) compared to a large pretrained diffusion model
(FLUX.1-schnell) when both are evaluated at the same resolution, while offering strict
advantages in likelihood tractability (NLL) and sampling efficiency (NFE).

## Model Configurations

### GLOW
- Architecture: 4 blocks × 32 flows, hidden_channels=512, AffineCoupling with clamp(−3,3)
- Parameters: ~76M
- Training data: FFHQ-64×64 (68,000 train / 2,000 val)
- Optimizer: Adam, lr=5e-5, 1,000-step linear warmup from lr/100
- Gradient clipping: norm ≤ 0.5 + NaN/Inf grad guard
- Training steps: 30,000 (batch=128, single GPU)
- Checkpoints: every 5,000 steps; samples every 2,000 steps (temp=0.7)

### FLUX.1-schnell
- Architecture: Flow-matching diffusion transformer (~12B params)
- Pretrained weights: `/data/jameskimh/flux_pretrained/FLUX.1-schnell`
- Inference: 4 NFE, guidance_scale=0.0, resolution forced to 64×64
- Prompts: 4 portrait/face prompts cycled across 5,000 samples

## Dataset

| Split    | Images | Path |
|----------|--------|------|
| Train    | 68,000 | `/data/jameskimh/final_project/data/ffhq64` |
| Val      |  2,000 | First 2,000 sorted files |
| FID ref  | 70,000 | Full FFHQ-64×64 directory |

## Metrics

| Metric | Description |
|--------|-------------|
| NLL (bits/dim) | Exact log-likelihood on val set; lower is better; GLOW only |
| FID | Fréchet Inception Distance vs full FFHQ-64×64 (cleanfid, 5,000 gen samples) |
| NFE | Forward evaluations per sample |
| Sample time | Wall-clock seconds to generate 5,000 samples on single GPU |

## Training Stability Issues Encountered

1. **Inf loss at step ~300–500** — `clamp(-5,5)` on coupling log_s allowed compound
   amplification over 128 sequential flow steps → fixed: `clamp(-3,3)`
2. **Gradient explosion with lr=1e-4** — Adam overshooting at early training →
   fixed: lr=5e-5 + 1,000-step linear warmup
3. **Old checkpoint (5k steps, tanh coupling) unusable** — all temperatures produced
   NaN samples; training from scratch required

## Source Files

| File | Description |
|------|-------------|
| `src/model.py` | GLOW: ActNorm, LU-InvConv, AffineCoupling, GlowBlock |
| `src/train.py` | Training loop with warmup, NaN/Inf guard, GradScaler scaffold |
| `src/dataset.py` | FFHQ DataLoader (num_workers=16, persistent_workers) |
| `src/evaluate.py` | Val NLL + cleanfid FID |
| `src/sample_flux.py` | FLUX.1-schnell/dev sampling via diffusers |

# Result 001: GLOW vs FLUX.1 vs DDIM/DDPM on FFHQ-64×64

## Summary

Full NFE--quality Pareto frontier across 8 operating points (NFE ∈ {1, 4, 8, 10, 20, 28, 50, 100}).

Key finding: **DDIM-50 achieves the best overall FID=61.78** at 50 NFE, outperforming all FLUX.1 variants.
GLOW (NFE=1, FID=183.35) remains the only single-pass option with exclusive likelihood tractability.
FLUX.1-schnell (NFE=4, FID=184.94) is Pareto-suboptimal: dominated by GLOW on both FID and speed.

## Quantitative Results

| Model | NLL (bits/dim) ↓ | FID ↓ | NFE | Sample time (5k imgs) | Params |
|-------|-----------------|-------|-----|-----------------------|--------|
| **GLOW** (30k steps) | **−4.61** | 183.35 | **1** | **137s** | 76M |
| FLUX.1-schnell | N/A | 184.94 | 4 | 1,231s | ~12B |
| FLUX.1-dev-8 | N/A | 126.15 | 8 | — | ~12B |
| DDIM-10† | N/A | 68.75 | 10 | 602s | 113M |
| DDIM-20† | N/A | 65.96 | 20 | 1,346s | 113M |
| DDIM-50† | N/A | **61.78** | 50 | 2,658s | 113M |
| DDPM-100† | N/A | 71.87 | 100 | 5,453s | 113M |
| FLUX.1-dev | N/A | 117.87 | 28 | ~8,617s | ~12B |

† CelebA-HQ-256 pretrained weights (`google/ddpm-celebahq-256`), outputs bicubically downsampled to 64×64.

## Training Curve (GLOW)

GLOW NLL (bits/dim) over 30,000 steps:

| Step | NLL | LR |
|------|-----|----|
| 100 | −2.39 | 5e-6 (warmup) |
| 1,000 | −3.81 | 5e-5 (full) |
| 5,000 | −4.31 | 5e-5 |
| 10,000 | −4.43 | 5e-5 |
| 20,000 | −4.55 | 5e-5 |
| 30,000 | **−4.61** | 5e-5 |

Curve is still descending at step 30k — continued training would further reduce NLL.

## Key Findings

### 1. DDIM dominates the middle Pareto region
DDIM-10/20/50 achieve FID=68.75/65.96/61.78, all substantially better than any FLUX.1 variant.
This is because `google/ddpm-celebahq-256` (trained on face images) transfers well to 64×64 via
bicubic downsampling — unlike FLUX.1, which suffers a resolution mismatch (design ≥512×512).

### 2. DDIM vs DDPM: deterministic ODE wins
DDIM-50 (FID=61.78) outperforms DDPM-100 (FID=71.87) with half the NFE.
The deterministic DDIM trajectory is more sample-efficient than the stochastic DDPM chain.

### 3. FLUX.1-schnell is Pareto-suboptimal
FLUX.1-schnell (NFE=4, FID=184.94) is dominated by GLOW on both FID and speed.
At NFE=8, FLUX.1-dev improves to FID=126.15 but still cannot match DDIM-10 (FID=68.75 at 10 NFE).

### 4. GLOW's tractable likelihood is a unique advantage
GLOW reports exact NLL = −4.61 bits/dim, plus encode-decode reconstruction (PSNR=24.24 dB, SSIM=0.977).
Neither FLUX.1 nor DDPM/DDIM can compute exact NLL or perform encode-decode reconstruction.

### 5. OOD detection (GLOW only)
GLOW NLL separates in-distribution faces from OOD inputs:
- FFHQ (in-dist): NLL mean = −4.609 bits/dim (std=0.405)
- Solid color images: NLL mean = −6.854 bits/dim
- Random noise: NaN (numerical overflow in 128 sequential flow layers)
Note: solid images score *lower* NLL than faces — the known "likelihood ≠ quality" failure mode in NFs.

## Pareto Analysis

Three Pareto regimes:
- **Latency-critical (NFE=1)**: GLOW — only option with single-pass inference + exact NLL
- **Efficiency-optimal (NFE=10–50)**: DDIM-10/20/50 — best FID per NFE in this range
- **Suboptimal**: FLUX.1-schnell — dominated by GLOW (slower, higher FID)

FLUX.1-dev (NFE=8,28) occupies neither extreme well at 64×64: worse FID than DDIM at same or higher NFE.

## GLOW-Exclusive Capabilities

| Capability | Value | Notes |
|---|---|---|
| Exact NLL | −4.61 bits/dim | Decreasing at 30k steps |
| Reconstruction PSNR | 24.24 dB (std=0.31) | 200 val images |
| Reconstruction SSIM | 0.977 (std=0.008) | Near-perfect structure |
| Latent interpolation | Qualitative ✓ | Smooth semantic transitions |
| OOD detection | FFHQ=−4.61, solid=−6.85 | Likelihood ≠ perceptual quality |

## Verdict

| Criterion | Winner |
|-----------|--------|
| FID at NFE=1 | GLOW (183.35 vs schnell 184.94) |
| FID at NFE=4–28 | **DDIM-50** (61.78 best overall) |
| Sampling speed | **GLOW** (9× vs schnell; up to 63× vs dev) |
| Exact NLL | **GLOW** (others: N/A) |
| Reconstruction | **GLOW** (PSNR 24.24 dB, SSIM 0.977) |
| High-resolution | FLUX (not tested here) |
| Text conditioning | FLUX |
| Training cost | FLUX / DDIM (pretrained) |

## Artifacts

| Path | Description |
|------|-------------|
| `glow_pretrained/glow_v2_ffhq64_030000.pt` | Final GLOW checkpoint |
| `samples/glow/` | 5,000 GLOW samples (FID eval) |
| `samples/flux_schnell_nfe4/` | 5,000 FLUX schnell samples |
| `samples/flux_dev_nfe8/` | 5,000 FLUX dev-8 samples |
| `samples/flux_dev_nfe28/` | 5,000 FLUX dev-28 samples |
| `samples/ddim_nfe10/` | 5,000 DDIM-10 samples |
| `samples/ddim_nfe20/` | 5,000 DDIM-20 samples |
| `samples/ddim_nfe50/` | 5,000 DDIM-50 samples |
| `samples/ddpm_nfe100/` | 5,000 DDPM-100 samples |
| `logs/ddim_results.json` | DDIM/DDPM FID results |
| `logs/flux_dev_fid.json` | FLUX dev-28 FID result |
| `logs/flux_nfe8_fid.json` | FLUX dev-8 FID result |
| `logs/ood_results.json` | OOD detection NLL values |
| `experiments/results/` | All figures (fig1–fig9) |

## Next Steps (Future Work)

- Train GLOW to 100k–200k steps to see further NLL/FID improvement
- Evaluate GLOW at higher temperatures (temp > 0.7) to trade NLL for diversity
- Evaluate FLUX.1-dev at NFE=12, 16 to map the intermediate Pareto curve
- Test GLOW at 256×256 resolution to probe the resolution scaling limit

---

## Metric Glossary

### NLL (Negative Log-Likelihood, bits/dim)
NLL = -log₂ p(x) / (H×W×C). Lower is better. Random model = 8 bits/dim.
Only computable for normalizing flows (GLOW). Diffusion models require expensive ELBO approximation.

### FID (Fréchet Inception Distance)
Distributional distance between generated and real images (InceptionV3 features). Lower is better.

### NFE (Number of Function Evaluations)
Model forward passes per sample. GLOW: 1; FLUX schnell: 4; DDIM: 10/20/50; FLUX dev: 8/28; DDPM: 100.

### PSNR / SSIM (Reconstruction metrics)
Encode→decode roundtrip quality. GLOW-only. PSNR ∞ = perfect. SSIM 1.0 = identical.

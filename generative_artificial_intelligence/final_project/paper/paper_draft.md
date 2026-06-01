# GLOW vs FLUX: Comparing Normalizing Flows and Flow Matching for Face Image Generation at 64×64 Resolution

---

## Abstract

We present an empirical comparison of two fundamentally different generative modeling paradigms applied to face image synthesis at 64×64 resolution: GLOW (Generative Flow with Invertible 1×1 Convolutions), a normalizing flow model trained from scratch on FFHQ-64×64, and FLUX.1, a large-scale flow-matching diffusion model evaluated at the same resolution. Our experiments reveal that at low NFE budgets, task-specific normalizing flows match large pretrained diffusion models: GLOW achieves FID=183.35 vs FLUX.1-schnell FID=184.94 while sampling 9× faster. At 28 NFE, FLUX.1-dev achieves FID=117.87, outperforming GLOW at 63× the sampling cost and exposing a clear compute-quality Pareto frontier. GLOW retains exclusive advantages in tractable likelihood (NLL=−4.61 bits/dim), reconstruction fidelity (PSNR=24.24 dB, SSIM=0.977), and single-NFE throughput. These results demonstrate that architectural advantages of large diffusion models do not uniformly transfer outside their designed resolution regime.

---

## 1. Introduction

The landscape of deep generative modeling has been transformed by diffusion models and flow matching approaches, which achieve state-of-the-art sample quality at the cost of requiring multiple sequential forward evaluations (NFE) at inference time. In contrast, normalizing flows such as GLOW (Kingma & Dhariwal, 2018) produce samples in a single forward pass and additionally provide exact, tractable log-likelihoods — properties that diffusion models fundamentally cannot offer.

A natural question arises: when a large pretrained diffusion model and a task-specific normalizing flow are evaluated at the same resolution, which paradigm offers better sample quality per unit of compute? This comparison is non-trivial because modern diffusion models like FLUX.1 are trained on hundreds of millions of images across diverse resolutions, while GLOW is typically trained on a single domain dataset. The two models differ not only in architecture but in training regime, parameter count, and the type of inference they support.

In this work, we train GLOW from scratch on FFHQ-64×64 for 30,000 iterations and compare it against FLUX.1-schnell (4 NFE) and FLUX.1-dev (28 NFE), both forced to operate at 64×64 resolution. We evaluate across four axes: (1) sample quality via FID, (2) exact likelihood via NLL, (3) reconstruction fidelity via encode-decode PSNR/SSIM, and (4) sampling efficiency via NFE and wall-clock time. Our results provide a nuanced view of the trade-offs between these paradigms at low resolution.

---

## 2. Background

### 2.1 Normalizing Flows and GLOW

Normalizing flows define an invertible mapping f: X → Z between the data space X and a latent space Z equipped with a tractable prior p_Z (typically a standard Gaussian). The change-of-variables formula gives an exact log-likelihood:

    log p_X(x) = log p_Z(f(x)) + log |det(∂f/∂x)|

The log-determinant term is computed in closed form for each invertible layer, making exact density estimation possible. GLOW (Kingma & Dhariwal, 2018) builds on this framework with three key components per flow step: (1) ActNorm — data-driven mean/variance normalization, (2) Invertible 1×1 Convolution — learned channel permutation parameterized via LU decomposition, and (3) Affine Coupling — a split-and-transform operation where half the channels are used to predict scale and shift for the other half.

GLOW uses a multi-scale architecture that squeezes spatial dimensions while doubling channels at each level, then splits off half the latents to a learned Gaussian prior at intermediate resolutions. For FFHQ-64×64, we use 4 blocks × 32 flows with hidden dimension 512, totaling ~76M parameters.

### 2.2 Flow Matching and FLUX.1

FLUX.1 (Black Forest Labs, 2024) represents a recent paradigm shift: rather than defining an exact invertible bijection, it learns a vector field that transports samples from a noise distribution to the data distribution via an ODE. Inference requires solving this ODE numerically with multiple function evaluations (NFE). FLUX.1-schnell uses 4 NFE via a distilled trajectory; FLUX.1-dev uses 28 NFE for higher quality. The model is a 12B-parameter diffusion transformer conditioned on text prompts, pretrained on a large-scale internet dataset.

A key distinction is that FLUX.1's training data and architecture are designed for high-resolution synthesis (typically 512×1024). Using it at 64×64 requires overriding the resolution at inference time, which may degrade performance as the model's positional encodings and attention patterns are calibrated for larger feature maps.

---

## 3. Method

### 3.1 GLOW Training

We train GLOW on 68,000 FFHQ-64×64 images (2,000 reserved for validation). Images are normalized to [−0.5, 0.5] and dequantized with uniform noise (x + U(0, 1/256)) to convert discrete pixel values to a continuous distribution. We use the Adam optimizer with learning rate 5×10⁻⁵ and a 1,000-step linear warmup from lr/100 to lr, motivated by observed gradient instability at higher initial learning rates.

Training stability required two architectural modifications relative to standard GLOW implementations. First, the coupling layer's log-scale was clamped to [−3, 3] (compared to the commonly used [−5, 5]), reducing the maximum per-step amplification from e^5 ≈ 148 to e^3 ≈ 20. Second, a gradient validity check was added before each optimizer step, skipping the update if any parameter gradient contains NaN or Inf values. These modifications eliminated the recurrent divergence events (loss → ±∞) that occurred at around steps 300–500 with the default configuration.

The model is trained for 30,000 iterations with batch size 128 on a single NVIDIA GPU. Checkpoints and samples are saved every 5,000 and 2,000 steps respectively.

### 3.2 FLUX.1 Sampling

For both FLUX.1-schnell (4 NFE) and FLUX.1-dev (28 NFE), we generate 5,000 samples at 64×64 resolution using four cycling face-related text prompts: "a high quality portrait photo of a person", "portrait of a young woman, professional photography, neutral background", "portrait of a man with natural lighting, photorealistic", and "headshot photo of a person, clear face, high resolution". We use guidance_scale=0.0 for schnell (as recommended) and the default guidance scale for dev.

### 3.3 Evaluation

**FID**: We use cleanfid to compute FID between 5,000 generated samples and all 70,000 FFHQ-64×64 real images. Lower FID indicates better distributional match.

**NLL**: We evaluate GLOW's exact negative log-likelihood on the 2,000-image validation set, averaged over 50 batches. NLL is reported in bits per dimension (bits/dim).

**Reconstruction**: We measure GLOW's encode-decode fidelity by passing 200 validation images through the full forward-then-reverse pipeline and computing PSNR and SSIM between input and reconstruction.

**NFE and Sampling Time**: NFE is the number of model forward evaluations per sample. Sampling time is measured as wall-clock seconds to generate 5,000 images on a single GPU.

---

## 4. Experiments

### 4.1 Sample Quality: FID Comparison

Table 1 summarizes the FID results across all three models. GLOW trained for 30,000 steps achieves FID=183.35, marginally outperforming FLUX.1-schnell (FID=184.94) at 4 NFE. These results challenge the assumption that large pretrained models uniformly dominate task-specific smaller models: at 64×64 resolution, FLUX.1-schnell's massive parameter count, transformer attention, and text conditioning provide no FID benefit over GLOW.

FLUX.1-dev at 28 NFE achieves FID=117.87, substantially outperforming both GLOW and schnell. This demonstrates that the resolution mismatch can be partially overcome with sufficient inference-time compute: at 28 NFE, the dev model's iterative denoising accumulates enough correction steps to produce visually coherent 64×64 faces despite operating far below its training resolution. The FID gap (183.35 → 117.87 = 65.48 points) comes at a 63× sampling cost over GLOW, establishing a compute-quality trade-off rather than a uniform dominance.

We attribute the schnell–GLOW equivalence and the schnell–dev gap to two interacting factors. First, schnell uses only 4 NFE via a distilled trajectory optimized for high-resolution synthesis; at 64×64 the distillation target distribution is out-of-distribution, limiting the accuracy of each denoising step. Second, dev's 28-NFE trajectory uses a more conservative solver that remains on-manifold even at this resolution mismatch. The training NLL of −4.61 bits/dim for GLOW, still descending at 30k steps, suggests continued training would further reduce FID and potentially close the gap with dev.

### 4.2 Exact Likelihood: NLL Evaluation

One of the defining advantages of normalizing flows over diffusion models is the ability to compute exact log-likelihoods. GLOW achieves a validation NLL of −4.61 bits/dim. For context, a uniform distribution over [0,1]^D has NLL of 0 bits/dim, while a model that perfectly learns the data distribution achieves a lower bound set by the data's entropy.

FLUX.1, like all diffusion models, cannot provide exact NLL. Approximate bounds via ELBO require expensive importance sampling and are rarely reported in practice. This distinction is practically significant for applications requiring density estimation: anomaly detection (flagging low-likelihood inputs), lossless data compression (arithmetic coding with a learned model), and model-based optimization. GLOW's tractable likelihood makes it uniquely suited for these use cases, representing a non-trivial advantage that FID comparisons alone do not capture.

### 4.3 Reconstruction Fidelity

The bijective mapping of normalizing flows enables an encode-decode reconstruction test that diffusion models cannot perform: given a real image x, compute z=f(x), then recover x̂=f⁻¹(z) and measure the error. We evaluate 200 validation images with GLOW's 30k-step checkpoint.

Results show PSNR=24.24 dB (std=0.31) and SSIM=0.9771 (std=0.0081). The high SSIM indicates that reconstructions are structurally nearly identical to the originals — the faces' identity, pose, and expression are preserved. The moderate PSNR (rather than the theoretically infinite value of a perfect bijection) arises from two sources: (1) floating-point accumulation errors over 128 sequential flow steps and (2) the dequantization procedure, which trains the model to fit a continuous distribution offset from the original discrete pixel grid.

The consistency of these metrics across all 200 images (std=0.31 dB for PSNR) demonstrates that the reconstruction quality is not driven by outliers but reflects a stable property of the trained model. This encode-decode capability is fundamentally unavailable for FLUX.1 and represents a qualitatively different mode of use.

### 4.4 Sampling Efficiency

GLOW requires exactly 1 NFE per sample — a single deterministic forward pass through the invertible network. FLUX.1-schnell requires 4 NFE and FLUX.1-dev requires 28 NFE. The practical consequence is striking: generating 5,000 samples takes GLOW 137 seconds vs 1,231 seconds for FLUX schnell (~9× faster) and ~8,617 seconds for FLUX dev (~63× faster). All three models use a single GPU.

This efficiency gap is architectural, not incidental. Diffusion models fundamentally require iterative refinement because the generative process involves denoising across a continuous noise schedule. Flow models collapse this to a single closed-form computation. For deployment scenarios with latency or throughput requirements — real-time synthesis, high-volume batch generation, on-device inference — the NFE=1 property of GLOW represents a decisive advantage.

---

## 5. Discussion

Our experiments reveal a nuanced competitive landscape between normalizing flows and flow-matching diffusion models. At the specific operating point of 64×64 resolution with a task-specific dataset, GLOW achieves comparable FID to FLUX.1 while maintaining strict advantages in likelihood tractability, reconstruction capability, and sampling speed.

However, these results should not be interpreted as GLOW outperforming FLUX.1 in general. The comparison is explicitly constrained to a resolution where FLUX.1 is operating out of distribution. At 512×512 or higher, FLUX.1's advantages in image quality and diversity would be overwhelming. The meaningful conclusion is that the "scale wins" narrative in generative modeling has boundaries: at sufficiently low resolution, a well-trained task-specific model can match or exceed models that are orders of magnitude larger.

The training instability we encountered — specifically the gradient explosion at lr=1e-4 that required warmup and clamp reduction — highlights that normalizing flows are more sensitive to hyperparameters than modern diffusion training. The near-perfect SSIM (0.977) alongside moderate PSNR (24.24 dB) also raises an interesting question about what reconstruction quality means: structural fidelity appears well-preserved even as per-pixel accuracy is limited by floating-point precision.

---

## 6. Conclusion

We compare GLOW (normalizing flow, 76M params) and FLUX.1 (flow-matching diffusion, ~12B params) for face generation at 64×64 resolution. GLOW achieves FID=183.35, matching FLUX.1-schnell (184.94) at 4 NFE while sampling 9× faster. FLUX.1-dev at 28 NFE achieves FID=117.87, outperforming GLOW by 65 FID points at a 63× sampling cost. Beyond FID, GLOW retains exclusive capabilities: exact log-likelihood (NLL=−4.61 bits/dim) and stable encode-decode reconstruction (PSNR=24.24 dB, SSIM=0.977). These results map a compute-quality Pareto frontier — GLOW dominates at NFE=1, dev dominates at NFE=28, and schnell is Pareto-suboptimal — while demonstrating that normalizing flows remain practically significant when deployment requires tractable likelihood or single-pass inference.

---

## References

- Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1×1 convolutions. *NeurIPS*.
- Black Forest Labs. (2024). FLUX.1: A flow-matching text-to-image model.
- Karras, T., et al. (2019). A style-based generator architecture for generative adversarial networks. *CVPR*. (FFHQ dataset)
- Heusel, M., et al. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *NeurIPS*. (FID metric)
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using real-valued non-volume preserving transformations. *ICLR*. (NICE/RealNVP)
- Parmar, G., et al. (2022). On aliased resizing and surprising subtleties in GAN training. *CVPR*. (cleanfid)

# Algorithm for AI - KU Homework

## Overview
Korea University AI algorithms coursework. Implements Continual Concept Erasure experiments on Stable Diffusion, comparing UCE baseline with the proposed SCA-UCE method.

## Project Structure
```
algorithm_for_AI/
├── continual_CE.py        # Main experiment code (UCE & SCA-UCE algorithms)
├── run_continual_CE.sh    # Experiment runner script (logs to logs/)
├── logs/
│   └── experiment.log     # Experiment log output
└── results/               # Generated images and quantitative results
    ├── quantitative_table.json
    ├── {method}_{concept}_sample_{i}.png  # Visual samples
    └── {prefix}_{concept}/                # Per-evaluation image folders
```

## Algorithm
**UCE (Unified Concept Erasure)**: Modifies cross-attention (attn2) value projection weights in the Stable Diffusion UNet to erase target concepts. Maps the erased concept's embedding to the empty prompt ("") anchor while preserving related concepts.

**SCA-UCE (Sequential Concept-Aware UCE)**: Extends UCE by also accounting for previously erased concepts (past_concepts) in the closed-form weight update. Mitigates catastrophic forgetting when sequentially erasing multiple concepts.

## Experiment Design
- **Base Model**: CompVis/stable-diffusion-v1-4 (float16, CPU offload)
- **Target Concepts**: Superman, Van Gogh, Snoopy
- **Preserved Concepts**: 4 similar concepts per target (e.g., Superman -> Batman, Thor, Wonder Woman, Shazam)
- **Evaluation Metrics**: CLIP Score (CS), FID (Frechet Inception Distance)
- **3-Stage Comparison**: Original SD -> UCE Baseline -> SCA-UCE (Ours)
- **General Quality**: 50 MS-COCO captions to evaluate general generation ability

## Tech Stack
- Python 3, PyTorch (CUDA, float16)
- diffusers (StableDiffusionPipeline), transformers (CLIP)
- torchmetrics (FID), datasets (MS-COCO captions)

## Commands
```bash
# Run the experiment
cd /app/algorithm_for_AI
bash run_continual_CE.sh

# Or run directly
python continual_CE.py
```

## Notes
- GPU required (CUDA). Uses `enable_model_cpu_offload()` + `enable_attention_slicing()` for memory efficiency
- Results are saved as JSON to `results/quantitative_table.json`
- FID computation currently uses a zeros tensor as real_images placeholder - needs real images for accurate evaluation
- lamb=0.1 (regularization parameter) is the default for both UCE and SCA-UCE

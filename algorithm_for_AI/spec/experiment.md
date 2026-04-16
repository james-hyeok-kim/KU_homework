# Recommended Algorithms for Continual Concept Erasure

## Problem Overview

The task is **Continual Concept Erasure (Continual CE)** on Stable Diffusion v1.4, where three concepts — **Superman → Van Gogh → Snoopy** — are sequentially erased. After the final erasure step, all three concepts must remain erased while preserving all other concepts. The reference paper is SPM (One-dimensional Adapter) from CVPR 2024.

---

## 1. UCE (Unified Concept Editing)

- **Paper**: Gandikota et al., *Unified Concept Editing in Diffusion Models*, 2024
- **Mechanism**: Modifies the key/value projection matrices in cross-attention layers using a closed-form solution to erase target concepts.
- **Role in Experiment**: Listed as the **baseline** in the assignment table; must be implemented.
- **Continual CE Concern**: When applied sequentially, later erasures may overwrite earlier ones, causing previously erased concepts to reappear. Analyzing this limitation is a key contribution point.

## 2. SPM (Semi-Permeable Membrane / One-dimensional Adapter)

- **Paper**: Lyu et al., *One-dimensional Adapter to Rule Them All*, CVPR 2024
- **Mechanism**: Learns an independent 1D adapter for each target concept. The adapter acts as a filter that blocks the generation of the specific concept while leaving the rest of the model untouched.
- **Role in Experiment**: Ideal candidate for the **"Ours"** method. Since each adapter is trained independently without modifying shared model weights, it is inherently suited for continual settings — new adapters can be added without disturbing existing ones.

## 3. ESD (Erased Stable Diffusion)

- **Paper**: Gandikota et al., *Erasing Concepts from Diffusion Models*, ICCV 2023
- **Mechanism**: Fine-tunes the model by reversing the noise prediction loss direction for the target concept, effectively steering the model away from generating it.
- **Role in Experiment**: Useful as a **comparison method**. Implementation is relatively straightforward, but the direct weight modification makes it highly susceptible to catastrophic forgetting in continual settings.

## 4. Concept Ablation (CA)

- **Paper**: Kumari et al., *Ablating Concepts in Text-to-Image Diffusion Models*, ICCV 2023
- **Mechanism**: Maps the distribution of the target concept to an anchor concept (e.g., "a person" for character erasure, "a painting" for style erasure).
- **Role in Experiment**: Another **comparison method** that directly modifies model weights. Similar to ESD, performance degradation under sequential erasure provides a useful contrast against adapter-based approaches.

## 5. Continual Learning Regularization (EWC / SI)

- **Technique**: Elastic Weight Consolidation (EWC) or Synaptic Intelligence (SI)
- **Mechanism**: Adds a regularization term that penalizes changes to parameters deemed important for previously learned (or erased) tasks. This protects earlier erasures while allowing new ones.
- **Role in Experiment**: Can be combined with weight-modification methods (UCE, ESD) to mitigate catastrophic forgetting. This combination can serve as a **novel contribution** — demonstrating that continual learning strategies improve sequential concept erasure.

---

## Recommended Experimental Setup

| Role | Method | Key Characteristic |
|------|--------|--------------------|
| **Baseline** | UCE (sequential) | Closed-form weight editing; prone to forgetting |
| **Comparison** | ESD (sequential) | Fine-tuning based; strong forgetting expected |
| **Comparison** | CA (sequential) | Distribution remapping; similar forgetting issues |
| **Ours** | SPM-based + CL strategy | Independent adapters + regularization |

## Evaluation Metrics

For each erased concept and its related concepts, measure:

- **CS (CLIP Score)**: Lower is better for erased concepts (↓), higher is better for MS-COCO general quality (↑).
- **FID (Fréchet Inception Distance)**: Lower is better across all categories (↓).

### Related Concepts for Evaluation

| Erased Concept | Related Concepts |
|----------------|-----------------|
| Superman_r | Batman, Thor, Wonder Woman, Shazam |
| Van Gogh_r | Picasso, Monet, Paul Gauguin, Caravaggio |
| Snoopy_r | Mickey, Spongebob, Pikachu, Hello Kitty |

---

## Narrative for the Report

The report should follow this logical structure:

1. **Motivation**: Weight-modification methods (UCE, ESD) suffer from catastrophic forgetting in continual erasure scenarios — erasing a new concept restores previously erased ones.
2. **Idea**: Leverage adapter-based approaches (SPM) that keep each erasure module independent, optionally combined with continual learning regularization for any shared parameters.
3. **Results**: Show quantitative (CS, FID table) and qualitative (generated image comparisons) evidence that adapter-based methods maintain all erasures simultaneously.
4. **Analysis/Ablation**: Study the effect of regularization strength, adapter capacity, and erasure order on final performance.

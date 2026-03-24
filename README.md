# Reproducing and Improving TypiClust

Machine Learning Coursework 2 — Lucas Lobo (k23075501)

## Overview

Reproduction and modification of the TypiClust algorithm from [Hacohen et al. (2022)](https://proceedings.mlr.press/v162/hacohen22a.html) on CIFAR-10. We implement TPC_RP (SimCLR + K-Means + typicality selection), evaluate across all three paper frameworks with all 8 baselines, and propose a hybrid modification combining typicality with diversity-aware selection and PCA dimensionality reduction.

## Key Results

**Original TypiClust (SS Embedding, B=10):**
- Random baseline: 42.07% ± 7.45%
- TypiClust (TPC_RP): **51.84%** (+9.77% over random)
- Paper reports ~48% vs ~38% — relative improvement matches closely

**Our Modification (PCA-10 + Hybrid α=0.55):**
- Accuracy: **73.91%** (+22.07% over baseline)
- Full class coverage (10/10)

**Phase Transition Reproduced:**
- TypiClust dominates B=10 through B=40 in SS embedding framework
- Margin overtakes at B=50, confirming the paper's phase transition finding

## Repository Structure

```
.
├── notebooks/
│   ├── SimCLR_implementation.ipynb   # SimCLR encoder training (ResNet-18, 500 epochs, SGD)
│   ├── original_algorithm.ipynb      # Original TypiClust (TPC_RP) reproduction
│   ├── modified_algorithm.ipynb      # Modified algorithm (PCA-10 + Hybrid selection)
│   ├── experiments.ipynb             # Full experiment suite (sweeps, ablations, stats)
│   ├── fw1_fully_supervised.ipynb    # Framework 1: Fully supervised (ResNet-18)
│   ├── fw2_ss_embedding.ipynb        # Framework 2: SS embedding (linear probe)
│   └── fw3_semi_supervised.ipynb     # Framework 3: Semi-supervised (FlexMatch-style)
├── src/                              # Modular pipeline components
│   ├── representations.py            # SimCLR encoder, L2-normalised embedding extraction
│   ├── preprocessing.py              # PCA and UMAP dimensionality reduction
│   ├── clustering.py                 # K-Means clustering
│   ├── typicality.py                 # Typicality computation
│   ├── selection.py                  # All 9 selection strategies
│   ├── evaluation.py                 # 3 evaluation frameworks + multi-round AL loop
│   └── pipeline.py                   # End-to-end pipeline runner
├── report/
│   ├── main.tex                      # LaTeX report source
│   └── figures/                      # Report figures
└── requirements.txt
```

## Implementation Details

### SimCLR Encoder (Appendix F.1)
- ResNet-18, 500 epochs, SGD (lr=0.4, momentum 0.9, cosine schedule)
- Batch size 512, weight decay 1e-4
- L2-normalised 512-d penultimate layer embeddings
- Augmentations: random resized crop, horizontal flip, color jitter, random grayscale

### AL Baselines (Section 4.1)
All 8 paper baselines implemented from scratch:
1. Random
2. Uncertainty (lowest max softmax)
3. Margin (smallest top-2 margin)
4. Entropy (highest entropy)
5. CoreSet (greedy k-center)
6. BADGE (gradient embeddings + k-means++)
7. BALD (MC dropout, mutual information)
8. DBAL (MC dropout, mean entropy)

### Evaluation Frameworks (Appendix F.2)
1. **Fully supervised**: ResNet-18, SGD/Nesterov, cosine LR, random crops + flips
2. **SS embedding**: Linear probe (LogisticRegression, C=100) on frozen embeddings
3. **Semi-supervised**: FlexMatch-style pseudo-labelling with ResNet-18

## Setup

```bash
pip install -r requirements.txt
```

## Running

1. **Train SimCLR encoder:** Run `notebooks/SimCLR_implementation.ipynb` (~5h on RTX 4070)
2. **Original algorithm:** Run `notebooks/original_algorithm.ipynb`
3. **Modified algorithm:** Run `notebooks/modified_algorithm.ipynb`
4. **Experiments:** Run `notebooks/experiments.ipynb`
5. **Framework comparisons:** Run `fw1_fully_supervised.ipynb`, `fw2_ss_embedding.ipynb`, `fw3_semi_supervised.ipynb`

## Results Summary (SS Embedding Framework)

| Strategy | B=10 | B=20 | B=30 | B=40 | B=50 |
|----------|------|------|------|------|------|
| Random | 39.7% | 56.4% | 64.6% | 72.0% | 74.9% |
| Uncertainty | 44.2% | 55.8% | 58.6% | 62.8% | 70.2% |
| Margin | 44.2% | 60.1% | 68.2% | 78.0% | **82.4%** |
| CoreSet | 50.6% | 65.1% | 73.0% | 76.4% | 76.3% |
| BADGE | 44.2% | 61.4% | 70.1% | 73.0% | 78.1% |
| **TPC_RP** | **51.9%** | **76.9%** | **78.2%** | **81.6%** | 80.8% |

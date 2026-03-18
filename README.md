# Reproducing and Improving TypiClust

Machine Learning Coursework 2 — Lucas Lobo (k23075501)

## Overview

Reproduction and modification of the TypiClust algorithm from [Hacohen et al. (2022)](https://proceedings.mlr.press/v162/hacohen22a.html) on CIFAR-10. We implement the TPC_RP variant (SimCLR + K-Means + typicality selection) and propose a hybrid modification combining typicality with diversity-aware selection and UMAP dimensionality reduction.

## Repository Structure

```
.
├── notebooks/
│   ├── SimCLR_implementation.ipynb   # SimCLR encoder training (ResNet-18, 200 epochs)
│   ├── original_algorithm.ipynb      # Original TypiClust (TPC_RP) reproduction
│   ├── modified_algorithm.ipynb      # Modified algorithm (UMAP + hybrid selection)
│   └── experiments.ipynb             # Full experiment suite (sweeps, ablations, stats)
├── src/                              # Modular pipeline components
│   ├── representations.py            # SimCLR encoder loading and embedding extraction
│   ├── preprocessing.py              # PCA and UMAP dimensionality reduction
│   ├── clustering.py                 # K-Means clustering
│   ├── typicality.py                 # Typicality computation
│   ├── selection.py                  # Selection strategies (max typicality, hybrid)
│   ├── evaluation.py                 # Linear probe evaluation and random baseline
│   └── pipeline.py                   # End-to-end pipeline runner
├── report/
│   ├── main.tex                      # LaTeX report source
│   └── figures/                      # Report figures
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

1. **Train SimCLR encoder:** Run `notebooks/SimCLR_implementation.ipynb`
2. **Original algorithm:** Run `notebooks/original_algorithm.ipynb`
3. **Modified algorithm:** Run `notebooks/modified_algorithm.ipynb`
4. **Full experiments:** Run `notebooks/experiments.ipynb`

## Results

| Method | Accuracy | Classes Covered |
|--------|----------|----------------|
| Random (30 seeds) | 36.92 +/- 6.75% | -- |
| TypiClust (baseline) | 44.42 +/- 0.42% | 7 |
| Hybrid (alpha=0.4) | 53.34% | 10 |
| PCA-10 + Hybrid | 57.75% | 8 |
| UMAP-5 + Hybrid | **65.78 +/- 1.71%** | 9.6 |

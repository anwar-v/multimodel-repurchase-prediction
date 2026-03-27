# Enhancing Customer Repurchase Prediction with Targeted CLIP Features: An RFM-Based Multimodal Approach

## Overview

This repository contains the complete implementation of a two-stage multimodal pipeline for predicting 30-day customer repurchase behaviour on the Amazon All Beauty 2023 dataset. The study combines RFM (Recency, Frequency, Monetary) behavioural features with targeted features extracted from OpenAI's CLIP vision-language model.

**Key Finding:** Raw 512-dimensional CLIP embeddings hurt prediction performance when concatenated with behavioural features. However, targeted CLIP-derived features (zero-shot classification scores, review sentiment, text-image similarity) improved 3 out of 4 classifiers, with XGBoost improving from F1=0.3988 to F1=0.4078 (+0.009).

## Repository Structure

```
├── Thesis_Complete_final.ipynb    # Full pipeline notebook (Google Colab, A100 GPU)
├── README.md                      # This file
└── amazon_thesis_data/            # Data directory (Google Drive, not included)
    ├── All_Beauty.jsonl           # Amazon Reviews 2023 - All Beauty (311.5 MB)
    ├── Meta_All_Beauty.jsonl      # Product metadata (203.1 MB)
    ├── text_embeddings.npy        # CLIP text embeddings (5423 × 512)
    └── image_embeddings.npy       # CLIP image embeddings (5423 × 512)
```

## Pipeline

### Stage 1: Baseline RFM Model

1. **Data Collection & Merging** — Load Amazon Reviews 2023 All Beauty dataset (701K reviews, 112K products), merge on `parent_asin`
2. **Data Cleaning** — Remove duplicates, apply temporal filter (2019–2022), handle missing prices, filter to repeat buyers (2+ purchases), winsorize price outliers
3. **EDA** — Rating distributions, review volume trends, purchase frequency analysis, feature correlations
4. **Label Engineering** — Binary 30-day repurchase label (any product, not same product), resulting in 79/21 class split
5. **RFM Feature Engineering** — Customer-level recency, frequency, monetary values with quintile-based scoring
6. **K-Means Clustering** — k=4 clusters on standardised RFM features (elbow method)
7. **Feature Engineering** — 11 handcrafted features (RFM values, scores, cluster, rating, review length, product-level stats)
8. **Seven Classifiers** — Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, AdaBoost, XGBoost with GridSearchCV hyperparameter tuning

**Best Stage 1 Result:** XGBoost — F1=0.3988, ROC-AUC=0.6627

### Stage 2: CLIP Multimodal Extension

9. **CLIP Embeddings** — Extract 512-dim text and image embeddings using `openai/clip-vit-base-patch32`
10. **PCA Dimensionality Reduction** — Reduce to 95% variance (text: 512→274, image: 512→252)
11. **Ablation Study** — Test Base, Base+Text, Base+Image, Base+Both → raw embeddings hurt performance across all conditions
12. **Improvement Strategies** — Aggressive PCA, per-condition retuning, feature rescaling, late fusion → all failed to beat baseline
13. **Targeted CLIP Features** — Extract 6 meaningful features instead of raw embeddings:
    - CLIP zero-shot satisfaction/dissatisfaction/quality scores
    - VADER/TextBlob sentiment polarity and subjectivity
    - CLIP text-image cosine similarity

**Best Stage 2 Result:** XGBoost with Base + CLIP scores only (15 features) — F1=0.4078, ROC-AUC=0.6645

## Dataset

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — All Beauty category

| Metric | Value |
|--------|-------|
| Raw reviews | 701,528 |
| Raw products | 112,590 |
| After preprocessing | 5,423 transactions |
| Unique repeat buyers | 2,358 |
| Unique products | 2,511 |
| Date range | 2019-01-02 to 2022-12-30 |
| Class split | 79.1% no return / 20.9% return |

## Results Summary

### Stage 1: Baseline (11 RFM features, tuned)

| Classifier | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|------------|----------|-----------|--------|----------|---------|
| **XGBoost** | **0.6415** | **0.3064** | **0.5708** | **0.3988** | **0.6627** |
| SVM | 0.6304 | 0.2998 | 0.5796 | 0.3952 | 0.6570 |
| Decision Tree | 0.6442 | 0.3039 | 0.5487 | 0.3912 | 0.6222 |
| Random Forest | 0.6553 | 0.3083 | 0.5265 | 0.3889 | 0.6616 |
| Logistic Regression | 0.6046 | 0.2808 | 0.5752 | 0.3774 | 0.6427 |
| k-NN | 0.7253 | 0.2632 | 0.1770 | 0.2116 | 0.5605 |
| AdaBoost | 0.7899 | 0.4375 | 0.0310 | 0.0579 | 0.6654 |

### Stage 2: Targeted CLIP Features (Best Condition)

| Classifier | Base F1 | + CLIP F1 | Change |
|------------|---------|-----------|--------|
| **XGBoost** | 0.3988 | **0.4078** | **+0.0090** |
| Logistic Regression | 0.3774 | 0.3843 | +0.0069 |
| k-NN | 0.2116 | 0.2135 | +0.0019 |
| Decision Tree | 0.3912 | 0.3912 | 0.0000 |
| Random Forest | 0.3889 | 0.3878 | −0.0011 |
| SVM | 0.3952 | 0.3941 | −0.0011 |

## Requirements

The notebook runs on **Google Colab with A100 GPU**. Key dependencies:

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost
- transformers, torch (for CLIP)
- Pillow, requests (for image downloading)
- matplotlib, seaborn
- vaderSentiment, textblob

All packages are installed in the first cell of the notebook.

## Reproducibility

- Random seed: `SEED = 42` used consistently across all splits, models, and PCA
- Train/test split: 80/20 stratified
- Cross-validation: 5-fold stratified
- Product-level features computed from training data only (no data leakage)
- All intermediate artifacts saved as `.pkl` and `.npy` files for session recovery

## How to Run

1. Clone this repository
2. Download the Amazon All Beauty 2023 dataset from [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
3. Upload `All_Beauty.jsonl` and `Meta_All_Beauty.jsonl` to Google Drive under `amazon_thesis_data/`
4. Open `Thesis_Complete_final.ipynb` in Google Colab
5. Set runtime to **GPU (A100 recommended)**
6. Run all cells sequentially

## Citation

If you use this work, please cite:

```
@mastersthesis{anwar2026repurchase,
  title={Enhancing Customer Repurchase Prediction with Targeted CLIP Features: An RFM-Based Multimodal Approach},
  author={Anwar, Vemula},
  year={2026},
  school={Gisma University of Applied Sciences}
}
```

## License

This project is for academic and research purposes. The Amazon Reviews 2023 dataset is subject to its own [terms of use](https://amazon-reviews-2023.github.io/).

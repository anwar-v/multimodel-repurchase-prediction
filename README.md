# Enhancing Customer Repurchase Prediction with Targeted CLIP Features
## An RFM-Based Multimodal Approach

MSc Artificial Intelligence and Data Science — Gisma University of Applied Sciences

## Overview
This repository contains the complete experimental pipeline for predicting 30-day customer repurchase behaviour using RFM behavioural features combined with targeted CLIP-derived features on the Amazon All Beauty 2023 dataset.

## Pipeline
- **Stage 1:** RFM feature engineering + K-Means clustering (k=4) + 7 ML classifiers with 3-phase hyperparameter tuning
- **Stage 2:** CLIP feature extension — raw embeddings (failed), then targeted features (zero-shot scores, sentiment, text-image similarity)

## Key Results
- Best baseline: XGBoost F1 = 0.401, AUC = 0.666
- Best with targeted CLIP: XGBoost F1 = 0.4078 (+0.009)
- Raw 512d CLIP embeddings reduced performance across all classifiers

## Dataset
Amazon Reviews 2023 — All Beauty category
- Source: https://amazon-reviews-2023.github.io/
- 5,423 transactions from 2,358 repeat buyers (2019–2022)

## Requirements
- Google Colab with A100 GPU
- Python 3.10+
- Key libraries: scikit-learn, xgboost, transformers, torch, clip

## How to Run
1. Upload the notebook to Google Colab
2. Mount Google Drive with the dataset files (All_Beauty.jsonl, Meta_All_Beauty.jsonl)
3. Run all cells sequentially

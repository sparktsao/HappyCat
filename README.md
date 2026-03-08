# HappyCat

HappyCat is an educational malware-classification demo originally created in 2016 during early company-wide machine learning adoption.

![HappyCat](happycat.jpg)

## Background

This project was built to help engineers understand practical classification workflows in a production-adjacent context at a time when formal MLOps practices were not yet common in the field.

The current codebase keeps that educational intent while presenting the workflow with cleaner structure and reproducible execution.

## Overview

HappyCat provides a concise end-to-end pipeline for binary malware classification (`normal` vs `malicious`) using `.vlog` feature files.

Core characteristics:

- deterministic data loading and shuffling,
- configurable K-fold cross-validation,
- train/test evaluation with precision, recall, F1, and confusion matrix,
- persisted outputs for model (`.pkl`) and metrics (`.json`).

## Project structure

- `learning/happycat.py`: CLI entrypoint for training and evaluation.
- `learning/learning_kernel.py`: dataset loading, model building, evaluation, cross-validation, and artifact persistence.
- `dataset/unittest/`: sample dataset used for local runs.

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

Run from repository root:

```bash
python learning/happycat.py \
  --dataset dataset/unittest \
  --folds 3 \
  --run-name unittest_demo
```

Expected outputs:

- `artifacts/unittest_demo_model.pkl`
- `artifacts/unittest_demo_metrics.json`

## Additional options

```bash
python learning/happycat.py \
  --dataset dataset/unittest \
  --folds 5 \
  --drop-columns 1,3,8 \
  --output-dir artifacts \
  --run-name my_experiment
```

## Notes for interview/review context

This project intentionally uses a **strong baseline model** (`LogisticRegression`) and emphasizes:

- readability,
- reproducibility,
- metrics quality,
- and clean CLI ergonomics.

That keeps the repository focused on software engineering quality while still demonstrating practical ML workflow fundamentals.

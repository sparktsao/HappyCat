# HappyCat

A compact, recruiter-friendly machine learning project that demonstrates a clean malware-classification workflow on tabular features.

![HappyCat](happycat.jpg)

## What this project shows

- Binary classification (`normal` vs `malicious`) on `.vlog` feature files.
- Reproducible training with deterministic shuffling and model configuration.
- K-fold cross-validation, train/test evaluation, and metrics serialization.
- Production-like outputs (`.pkl` model + `.json` metrics) in an artifacts directory.

## Project structure

- `learning/happycat.py`: training entrypoint and CLI.
- `learning/learning_kernel.py`: data loading, model building, CV, and artifact saving.
- `dataset/unittest/`: sample dataset.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

From repository root:

```bash
python learning/happycat.py \
  --dataset dataset/unittest \
  --folds 3 \
  --run-name unittest_demo
```

Outputs:

- `artifacts/unittest_demo_model.pkl`
- `artifacts/unittest_demo_metrics.json`

## Example command options

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

"""Core training utilities for the HappyCat malware-classification demo.

This module provides:
- dataset loading from `.vlog` files
- binary label generation (malicious vs normal)
- model training helpers
- evaluation and reporting helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

LABEL_MAP: Dict[str, int] = {"normal": 0, "malicious": 1}


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str


def _parse_vlog_file(path: Path) -> pd.DataFrame:
    """Parse a `.vlog` file and drop the leading sha1 column."""
    df = pd.read_csv(path, comment="#", header=None, engine="c")
    if df.empty:
        raise ValueError(f"Empty dataset file: {path}")

    # Standard file format stores sha1 in column 0.
    features_df = df.iloc[:, 1:].copy()
    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return features_df


def load_dataset(dataset_dir: str | Path, drop_columns: Sequence[int] | None = None) -> Dataset:
    """Load and merge all known class files from a folder.

    Args:
        dataset_dir: Directory containing `normal.vlog` and `malicious.vlog`.
        drop_columns: 1-based feature column indices to drop.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    matrices: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    feature_names: List[str] | None = None

    for name, label in LABEL_MAP.items():
        file_path = dataset_path / f"{name}.vlog"
        if not file_path.exists():
            continue

        df = _parse_vlog_file(file_path)
        if feature_names is None:
            feature_names = [f"f{i+1}" for i in range(df.shape[1])]

        matrices.append(df.to_numpy(dtype=np.float64))
        labels.append(np.full((df.shape[0],), label, dtype=np.uint8))

    if not matrices:
        expected = ", ".join(f"{name}.vlog" for name in LABEL_MAP)
        raise ValueError(f"No class files found in {dataset_path}. Expected one of: {expected}")

    X = np.vstack(matrices)
    y = np.concatenate(labels)

    if drop_columns:
        # Convert to 0-based indices.
        drop_ix = sorted({idx - 1 for idx in drop_columns if idx > 0})
        X = np.delete(X, drop_ix, axis=1)
        if feature_names is not None:
            feature_names = [name for i, name in enumerate(feature_names) if i not in set(drop_ix)]

    rng = np.random.default_rng(seed=1571)
    order = rng.permutation(len(X))
    X = X[order]
    y = y[order]

    return Dataset(features=X, labels=y, feature_names=feature_names or [])


def build_model() -> LogisticRegression:
    """Create a strong baseline model for binary classification."""
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=1571,
        max_iter=1000,
    )


def evaluate(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
    predictions = model.predict(X)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    confusion = confusion_matrix(y, predictions)
    report = classification_report(y, predictions, target_names=["normal", "malicious"], zero_division=0)
    return EvaluationResult(precision=precision, recall=recall, f1=f1, confusion=confusion, report=report)


def cross_validate(dataset: Dataset, folds: int) -> List[EvaluationResult]:
    if folds < 2:
        raise ValueError("folds must be >= 2 for cross-validation")

    cv = KFold(n_splits=folds, shuffle=True, random_state=1571)
    results: List[EvaluationResult] = []

    for train_idx, valid_idx in cv.split(dataset.features):
        model = build_model()
        model.fit(dataset.features[train_idx], dataset.labels[train_idx])
        results.append(evaluate(model, dataset.features[valid_idx], dataset.labels[valid_idx]))

    return results


def save_artifacts(
    model: LogisticRegression,
    metrics: dict,
    output_dir: str | Path,
    run_name: str,
) -> Tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f"{run_name}_model.pkl"
    metrics_path = output_path / f"{run_name}_metrics.json"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model_path, metrics_path

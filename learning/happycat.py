"""Train a baseline malware classifier on HappyCat `.vlog` datasets.

Example:
    python learning/happycat.py --dataset dataset/unittest --folds 3 --run-name unittest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from learning_kernel import (
    build_model,
    cross_validate,
    evaluate,
    load_dataset,
    save_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HappyCat baseline training entrypoint")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--run-name", default="happycat", help="Name prefix for saved artifacts")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save model and metrics artifacts",
    )
    parser.add_argument(
        "--drop-columns",
        default="",
        help="Comma-separated 1-based feature indices to remove (example: 1,2,5)",
    )
    parser.add_argument(
        "--test-subdir",
        default="test",
        help="Optional test sub-directory inside --dataset",
    )
    return parser.parse_args()


def _parse_drop_columns(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def main() -> None:
    args = parse_args()
    drop_columns = _parse_drop_columns(args.drop_columns)

    train_data = load_dataset(args.dataset, drop_columns=drop_columns)

    cv_results = cross_validate(train_data, folds=args.folds)
    cv_summary = {
        "precision_mean": mean([r.precision for r in cv_results]),
        "recall_mean": mean([r.recall for r in cv_results]),
        "f1_mean": mean([r.f1 for r in cv_results]),
    }

    model = build_model()
    model.fit(train_data.features, train_data.labels)

    train_eval = evaluate(model, train_data.features, train_data.labels)

    metrics: dict = {
        "dataset": args.dataset,
        "folds": args.folds,
        "drop_columns": drop_columns,
        "cv_summary": cv_summary,
        "train": {
            "precision": train_eval.precision,
            "recall": train_eval.recall,
            "f1": train_eval.f1,
            "confusion_matrix": train_eval.confusion.tolist(),
            "classification_report": train_eval.report,
        },
    }

    test_dir = Path(args.dataset) / args.test_subdir
    if test_dir.exists():
        test_data = load_dataset(test_dir, drop_columns=drop_columns)
        test_eval = evaluate(model, test_data.features, test_data.labels)
        metrics["test"] = {
            "precision": test_eval.precision,
            "recall": test_eval.recall,
            "f1": test_eval.f1,
            "confusion_matrix": test_eval.confusion.tolist(),
            "classification_report": test_eval.report,
        }

    model_path, metrics_path = save_artifacts(
        model,
        metrics,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    print("=== Cross-validation summary ===")
    print(cv_summary)
    print("=== Train report ===")
    print(train_eval.report)
    if "test" in metrics:
        print("=== Test report ===")
        print(metrics["test"]["classification_report"])
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()

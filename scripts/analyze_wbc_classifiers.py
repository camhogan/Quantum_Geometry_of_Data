#!/usr/bin/env python3
"""Cross-validated WBC classification using three Laplacian eigenmaps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "results/derived_quantities/wbc_all_weights.npz"
FOLDS = 10
THRESHOLD = 0.5


def evaluate(coordinates, labels, model_factory):
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=False)
    pooled_predictions = np.zeros(len(labels), dtype=np.int64)
    pooled_probabilities = np.zeros(len(labels), dtype=float)
    fold_results = []
    for fold, (train, test) in enumerate(
        splitter.split(coordinates, labels), start=1
    ):
        model = model_factory()
        model.fit(coordinates[train], labels[train])
        probabilities = model.predict_proba(coordinates[test])[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(np.int64)
        pooled_predictions[test] = predictions
        pooled_probabilities[test] = probabilities
        tn, fp, fn, tp = confusion_matrix(
            labels[test], predictions, labels=[0, 1]
        ).ravel()
        fold_results.append(
            {
                "fold": fold,
                "roc_auc": float(roc_auc_score(labels[test], probabilities)),
                "accuracy": float(accuracy_score(labels[test], predictions)),
                "sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
            }
        )

    tn, fp, fn, tp = confusion_matrix(
        labels, pooled_predictions, labels=[0, 1]
    ).ravel()
    aggregate = {}
    for metric in ("roc_auc", "accuracy", "sensitivity", "specificity"):
        values = np.asarray([row[metric] for row in fold_results])
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=0))
    aggregate.update(
        {
            "pooled_roc_auc": float(
                roc_auc_score(labels, pooled_probabilities)
            ),
            "pooled_accuracy": float(
                accuracy_score(labels, pooled_predictions)
            ),
            "pooled_sensitivity": float(tp / (tp + fn)),
            "pooled_specificity": float(tn / (tn + fp)),
            "pooled_confusion_matrix_tn_fp_fn_tp": [
                int(tn), int(fp), int(fn), int(tp)
            ],
        }
    )
    return aggregate, fold_results


def main() -> None:
    with np.load(DERIVED, allow_pickle=False) as archive:
        labels = (archive["labels"] == "M").astype(np.int64)
        coordinate_sets = {
            "0.2": archive["eigenmap_coordinates_w0p2"],
            "0.5": archive["eigenmap_coordinates_w0p5"],
        }

    models = {
        "Logistic regression": lambda: LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000
        ),
        "Gradient boosting": lambda: GradientBoostingClassifier(
            random_state=42
        ),
    }
    results = {
        "protocol": {
            "folds": FOLDS,
            "splitter": "StratifiedKFold(shuffle=False)",
            "decision_threshold": THRESHOLD,
            "features": "first three matrix-Laplacian eigenmap coordinates",
            "labels": "M=1, B=0",
            "standardization_inside_classifier": False,
        },
        "results": {},
    }
    csv_rows = []
    for weight, coordinates in coordinate_sets.items():
        results["results"][weight] = {}
        for model_name, model_factory in models.items():
            aggregate, folds = evaluate(coordinates, labels, model_factory)
            results["results"][weight][model_name] = {
                "aggregate": aggregate,
                "folds": folds,
            }
            csv_rows.append(
                {
                    "weight": weight,
                    "classifier": model_name,
                    **aggregate,
                }
            )

    output = ROOT / "results/derived_quantities"
    with (output / "wbc_classifier_results.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(results, stream, indent=2)
    with (output / "wbc_classifier_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Train and persist the production (v1) and candidate (v2) model artifacts.

Generates two scikit-learn classifiers on the Iris dataset:
  * v1.pkl  - LogisticRegression (baseline production model)
  * v2.pkl  - RandomForestClassifier (candidate canary model)

Also writes a small training baseline CSV used by the drift monitor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# scikit-learn's as_frame full column names -> short API names
FULL_FEATURE_NAMES = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}

MODEL_DIR = Path(__file__).resolve().parent
BASELINE_PATH = MODEL_DIR / "iris_baseline.csv"


def train_and_save() -> dict:
    iris = load_iris(as_frame=True)
    # sklearn >= 1.x: use the combined frame so feature names are always present
    frame = iris.frame.copy()
    X = frame[[FULL_FEATURE_NAMES[f] for f in FEATURE_NAMES]].values
    y = frame["target"].values

    # Persist a training baseline for drift comparisons
    baseline = frame[[FULL_FEATURE_NAMES[f] for f in FEATURE_NAMES]].copy()
    baseline.columns = FEATURE_NAMES
    baseline["target"] = y
    baseline.to_csv(BASELINE_PATH, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "v1": LogisticRegression(max_iter=1000, random_state=42),
        "v2": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        artifact_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, artifact_path)
        results[name] = {
            "artifact": str(artifact_path),
            "accuracy": round(float(acc), 4),
        }
        print(f"[{name}] trained -> {artifact_path} (test accuracy={acc:.4f})")

    summary_path = MODEL_DIR / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    train_and_save()

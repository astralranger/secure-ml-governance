"""Train and persist the production (v1) and candidate (v2) model artifacts.

Generates two scikit-learn classifiers on the Iris dataset:
  * v1.pkl  - LogisticRegression (baseline production model)
  * v2.pkl  - RandomForestClassifier (candidate canary model)

Also writes a small training baseline CSV used by the drift monitor.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..evaluation.risk_evaluator import run_quality_gate
from .metadata import sha256_file, write_manifest

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
SUMMARY_PATH = MODEL_DIR / "model_summary.json"
SCHEMA_VERSION = "iris-inference-v1"


def _mlflow_client():
    """Return MLflow when explicitly enabled, without forcing it on simple demos."""
    if os.getenv("MLFLOW_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return None
    try:
        import mlflow

        if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "secure-ml-governance"))
        return mlflow
    except Exception as exc:  # tracking must not prevent an artifact from being created
        print(f"[warn] MLflow tracking disabled: {exc}")
        return None


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
    dataset_sha256 = sha256_file(BASELINE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "v1": LogisticRegression(max_iter=1000, random_state=42),
        "v2": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    mlflow = _mlflow_client()
    results = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "sklearn-iris",
            "baseline_sha256": dataset_sha256,
            "feature_schema_version": SCHEMA_VERSION,
            "features": FEATURE_NAMES,
            "rows": int(len(baseline)),
        },
        "models": {},
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        artifact_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, artifact_path)
        quality_gate = run_quality_gate(str(artifact_path), X_test, y_test)
        model_record = {
            "artifact": str(artifact_path),
            "accuracy": round(float(acc), 4),
            "artifact_sha256": sha256_file(artifact_path),
            "feature_schema_version": SCHEMA_VERSION,
            "quality_gate": quality_gate,
        }
        if mlflow is not None:
            try:
                from mlflow.models import infer_signature

                with mlflow.start_run(run_name=f"iris-{name}") as run:
                    mlflow.log_params({
                        "model_version": name,
                        "model_class": type(model).__name__,
                        "feature_schema_version": SCHEMA_VERSION,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                    })
                    mlflow.log_metrics({"test_accuracy": float(acc)})
                    mlflow.set_tags({
                        "deployment_role": "production" if name == "v1" else "candidate",
                        "quality_gate": "passed" if quality_gate["passed"] else "failed",
                        "dataset_sha256": dataset_sha256,
                    })
                    mlflow.sklearn.log_model(
                        model,
                        name="model",
                        signature=infer_signature(X_train, preds),
                    )
                    model_record["mlflow_run_id"] = run.info.run_id
                    if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() in {"1", "true", "yes"}:
                        registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "iris-governance-model")
                        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", registered_name)
                        model_record["mlflow_registered_model"] = registered_name
                        model_record["mlflow_model_version"] = version.version
            except Exception as exc:
                model_record["mlflow_error"] = str(exc)
                print(f"[warn] MLflow logging failed for {name}: {exc}")
        results["models"][name] = model_record
        print(f"[{name}] trained -> {artifact_path} (test accuracy={acc:.4f})")

    write_manifest(SUMMARY_PATH, results)
    return results


if __name__ == "__main__":
    train_and_save()

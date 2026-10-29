"""Train and persist the production (v1) and candidate (v2) model artifacts.

Generates two scikit-learn pipelines on the UCI Adult Census Income dataset:
  * v1.pkl  - LogisticRegression (baseline production model)
  * v2.pkl  - RandomForestClassifier (candidate canary model)

Also writes a small training baseline CSV used by the drift monitor.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .dataset import FEATURE_NAMES, NUMERIC_BOUNDS, CATEGORIES, SCHEMA_VERSION, load_adult, training_split, sensitive_sex, DATA_DIR

from ..evaluation.risk_evaluator import run_quality_gate
from .metadata import sha256_file, write_manifest

MODEL_DIR = Path(__file__).resolve().parent
BASELINE_PATH = MODEL_DIR / "adult_baseline.csv"
SUMMARY_PATH = MODEL_DIR / "model_summary.json"


def make_pipeline(classifier):
    numeric = [FEATURE_NAMES.index(f) for f in NUMERIC_BOUNDS]
    # Retain protected attributes for auditing, but drop them from model inputs.
    categorical = [FEATURE_NAMES.index(f) for f in CATEGORIES if f not in {"sex", "race"}]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


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
    except Exception as exc:
        raise RuntimeError("MLFLOW_ENABLED is set, but tracking could not be initialized") from exc


def train_and_save() -> dict:
    import pandas as pd
    X_train, X_test, y_train, y_test = training_split()
    final_test, final_labels = load_adult("test")
    # Baseline contains only rows used for fitting, never holdout rows.
    baseline = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    baseline.to_csv(BASELINE_PATH, index=False)
    dataset_sha256 = sha256_file(BASELINE_PATH)

    models = {
        "v1": make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
        "v2": make_pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)),
    }

    mlflow = _mlflow_client()
    results = {
        "library_versions": {"scikit-learn": __import__("sklearn").__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "UCI Adult Census Income",
            "source": "https://archive.ics.uci.edu/dataset/2/adult",
            "source_sha256": {n: sha256_file(DATA_DIR / n) for n in ("adult.data", "adult.test")},
            "split": {"fit": len(X_train), "validation": len(X_test), "official_test": len(final_test)},
            "target_labels": {"0": "<=50K", "1": ">50K"},
            "excluded_predictors": ["race", "sex"],
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
        quality_gate = run_quality_gate(str(artifact_path), X_test, y_test, sensitive_sex(X_test),
                                        max_demographic_parity_gap=0.20, minimum_accuracy=0.80)
        quality_gate["model_artifact"] = f"src/models/{name}.pkl"
        model_record = {
            "artifact": f"src/models/{name}.pkl",
            "accuracy": round(float(acc), 4),
            "official_test_accuracy": round(float(accuracy_score(final_labels, model.predict(final_test.to_numpy(dtype=object)))), 4),
            "artifact_sha256": sha256_file(artifact_path),
            "feature_schema_version": SCHEMA_VERSION,
            "quality_gate": quality_gate,
        }
        if mlflow is not None:
            try:
                from mlflow.models import infer_signature

                with mlflow.start_run(run_name=f"adult-{name}") as run:
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
                    signature_frame = pd.DataFrame(X_train, columns=FEATURE_NAMES)
                    for column in NUMERIC_BOUNDS:
                        signature_frame[column] = pd.to_numeric(signature_frame[column])
                    logged_model = mlflow.sklearn.log_model(
                        model,
                        name="model",
                        serialization_format="cloudpickle",
                        signature=infer_signature(signature_frame, model.predict(X_train)),
                    )
                    model_record["mlflow_run_id"] = run.info.run_id
                    model_record["mlflow_model_uri"] = logged_model.model_uri
                    if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() in {"1", "true", "yes"}:
                        registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "adult-governance-model")
                        version = mlflow.register_model(logged_model.model_uri, registered_name)
                        model_record["mlflow_registered_model"] = registered_name
                        model_record["mlflow_model_version"] = version.version
            except Exception as exc:
                model_record["mlflow_error"] = str(exc)
                raise RuntimeError(f"Required MLflow logging failed for {name}") from exc
        results["models"][name] = model_record
        print(f"[{name}] trained -> {artifact_path} (test accuracy={acc:.4f})")

    write_manifest(SUMMARY_PATH, results)
    return results


if __name__ == "__main__":
    train_and_save()

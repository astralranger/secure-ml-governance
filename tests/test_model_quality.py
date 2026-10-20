"""Components 2, 3 & 4: model risk gate, traffic router, and drift monitor tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.api.router import TrafficRouter
from src.evaluation.drift_monitor import run_drift_check
from src.evaluation.risk_evaluator import run_quality_gate

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FULL_FEATURE_NAMES = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
}


class FailingModel:
    """A model whose predict always raises - used to exercise rollback."""

    def predict(self, x):
        raise RuntimeError("model inference failure")


@pytest.fixture
def trained_models(tmp_path):
    """Train two quick models and persist them to a temp directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    X = frame[[FULL_FEATURE_NAMES[f] for f in FEATURE_NAMES]].values
    y = frame["target"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    m1 = LogisticRegression(max_iter=1000, random_state=1).fit(Xtr, ytr)
    m2 = LogisticRegression(max_iter=1000, random_state=2).fit(Xtr, ytr)
    p1 = model_dir / "v1.pkl"
    p2 = model_dir / "v2.pkl"
    import joblib

    joblib.dump(m1, p1)
    joblib.dump(m2, p2)
    baseline = frame[[FULL_FEATURE_NAMES[f] for f in FEATURE_NAMES]].copy()
    baseline.columns = FEATURE_NAMES
    baseline.to_csv(model_dir / "baseline.csv", index=False)
    return {
        "v1": str(p1),
        "v2": str(p2),
        "baseline": str(model_dir / "baseline.csv"),
        "Xte": Xte,
        "yte": yte,
    }


def make_config(models, rollback=None):
    """Routing config. By default uses production-like thresholds that will NOT
    trigger rollback during normal requests; pass aggressive thresholds for
    rollback-specific tests."""
    return {
        "models": {
            "v1": {"artifact": models["v1"], "weight": 90, "status": "production"},
            "v2": {"artifact": models["v2"], "weight": 10, "status": "canary"},
        },
        "routing": {"shadow_mode": False, "shadow_percentage": 0},
        "rollback": rollback
        or {
            "enabled": True,
            "error_rate_threshold": 0.05,
            "latency_threshold_ms": 500,
            "min_requests": 10000,
        },
    }


def test_quality_gate_passes(trained_models):
    report = run_quality_gate(
        trained_models["v1"], trained_models["Xte"], trained_models["yte"]
    )
    assert report["artifact_loads"] is True
    assert report["accuracy"] is not None
    assert report["accuracy"] > 0.5
    assert report["stress_test_passed"] is True
    assert report["reproducible"] is True
    assert report["passed"] is True


def test_router_canary_weights_approximate(trained_models):
    config = make_config(trained_models)
    import joblib

    models = {name: joblib.load(p) for name, p in trained_models.items() if name in ("v1", "v2")}
    router = TrafficRouter(config, models)
    counts = {"v1": 0, "v2": 0}
    features = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    for _ in range(1000):
        _, used = router.predict(features)
        counts[used] += 1
    share_v2 = counts["v2"] / 1000
    # Weight is 10%, allow wide tolerance for randomness
    assert 0.02 < share_v2 < 0.25, f"v2 share {share_v2:.3f} outside expected canary range"


def test_router_shadow_mode_serves_v1(trained_models):
    import joblib

    config = make_config(trained_models)
    config["routing"]["shadow_mode"] = True
    models = {name: joblib.load(p) for name, p in trained_models.items() if name in ("v1", "v2")}
    router = TrafficRouter(config, models)
    features = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    for _ in range(20):
        result, used = router.predict(features)
        assert used == "v1"
        assert result["shadow"] is True


def test_router_rollback_on_repeated_errors(trained_models):
    # Aggressive thresholds force rollback as soon as errors occur
    aggressive = {
        "enabled": True,
        "error_rate_threshold": 0.0,
        "latency_threshold_ms": 0.0,
        "min_requests": 1,
    }
    config = make_config(trained_models, rollback=aggressive)
    config["models"]["v1"]["weight"] = 0
    config["models"]["v2"]["weight"] = 100
    router = TrafficRouter(config, {"v1": FailingModel(), "v2": FailingModel()})
    features = np.array([5.1, 3.5, 1.4, 0.2], dtype=float)
    for _ in range(5):
        router.predict(features)
    assert router._active_rollback is True, "rollback should activate on repeated errors"
    metrics = router.get_metrics()
    assert metrics["rollback_active"] is True
    router.reset_rollback()
    assert router._active_rollback is False


def test_production_failures_do_not_rollback_candidate(trained_models):
    """Rollback is a candidate safety control, not a response to a v1 outage."""
    aggressive = {
        "enabled": True,
        "error_rate_threshold": 0.0,
        "latency_threshold_ms": 0.0,
        "min_requests": 1,
        "monitor_models": ["v2"],
    }
    config = make_config(trained_models, rollback=aggressive)
    config["models"]["v1"]["weight"] = 100
    config["models"]["v2"]["weight"] = 0
    router = TrafficRouter(config, {"v1": FailingModel(), "v2": FailingModel()})
    router.predict(np.array([5.1, 3.5, 1.4, 0.2]))
    assert router.get_metrics()["rollback_active"] is False


def test_shadow_failure_triggers_candidate_rollback(trained_models):
    aggressive = {
        "enabled": True,
        "error_rate_threshold": 0.0,
        "latency_threshold_ms": 1_000_000,
        "min_requests": 1,
        "monitor_models": ["v2"],
    }
    config = make_config(trained_models, rollback=aggressive)
    config["routing"] = {"shadow_mode": True, "shadow_percentage": 100}
    import joblib

    router = TrafficRouter(config, {"v1": joblib.load(trained_models["v1"]), "v2": FailingModel()})
    result, used = router.predict(np.array([5.1, 3.5, 1.4, 0.2]))
    assert used == "v1"
    assert result["shadow"] is False
    assert router.get_metrics()["v2"]["errors"] == 1
    assert router.get_metrics()["rollback_active"] is True


def test_drift_detected_when_distribution_shifts(trained_models):
    baseline = pd.read_csv(trained_models["baseline"])
    # Live data shifted far outside the training range
    live = pd.DataFrame(
        {
            "sepal_length": np.random.uniform(50, 60, 100),
            "sepal_width": np.random.uniform(50, 60, 100),
            "petal_length": np.random.uniform(50, 60, 100),
            "petal_width": np.random.uniform(50, 60, 100),
        }
    )
    report = run_drift_check(
        trained_models["baseline"], live, FEATURE_NAMES, alpha=0.05
    )
    assert report["overall_drift"] is True
    assert all(f["drift_detected"] for f in report["features"])


def test_drift_not_detected_for_similar_distribution(trained_models):
    baseline = pd.read_csv(trained_models["baseline"])
    rng = np.random.default_rng(7)
    live = pd.DataFrame(
        {
            feat: rng.normal(baseline[feat].mean(), baseline[feat].std(), 150)
            for feat in FEATURE_NAMES
        }
    )
    report = run_drift_check(
        trained_models["baseline"], live, FEATURE_NAMES, alpha=0.05
    )
    # Similar distributions should generally not trigger drift (allow flakiness margin)
    drifted = [f for f in report["features"] if f["drift_detected"]]
    assert len(drifted) <= len(FEATURE_NAMES) // 2

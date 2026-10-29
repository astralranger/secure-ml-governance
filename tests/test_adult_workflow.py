"""Adult-specific regression checks for preprocessing, governance and live releases."""
import subprocess
import sys
import threading

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.dataset import EXAMPLE, FEATURE_NAMES, load_adult, training_split, sensitive_sex
from src.models.train_models import MODEL_DIR
from src.evaluation.risk_evaluator import run_quality_gate
from src.evaluation.drift_monitor import compute_drift
from src.api.router import TrafficRouter


def test_official_split_and_labels():
    train, y = load_adult("train")
    test, test_y = load_adult("test")
    assert len(train) == 32561 and len(test) == 16281
    assert set(y) == set(test_y) == {0, 1}
    assert train["occupation"].isna().any()
    assert "income" not in train.columns


def test_missing_categories_are_imputed(client, auth_headers):
    response = client.post("/predict", json={**EXAMPLE, "workclass": "?", "occupation": None}, headers=auth_headers)
    assert response.status_code == 200
    assert response.json["prediction"]["income_label"] in {"<=50K", ">50K"}
    assert len(response.json["prediction"]["probabilities"]) == 2


@pytest.mark.parametrize("changes", [{"occupation": "DROP TABLE"}, {"age": True}, {"age": "39"},
                                    {"education_num": 17}, {"hours_per_week": 0}, {"income": ">50K"}])
def test_invalid_adult_record_rejected(client, auth_headers, changes):
    assert client.post("/predict", json={**EXAMPLE, **changes}, headers=auth_headers).status_code == 400


def test_categorical_drift_and_zero_variance():
    baseline = pd.DataFrame({"workclass": ["Private"] * 100})
    same = compute_drift(baseline, baseline.copy(), ["workclass"]).to_dict()
    assert not same["overall_drift"]
    shifted = compute_drift(baseline, pd.DataFrame({"workclass": ["State-gov"] * 100}), ["workclass"]).to_dict()
    assert shifted["overall_drift"]
    assert shifted["features"][0]["test"] == "chi_square"


def test_fairness_threshold_can_block_real_candidate():
    _, X, _, y = training_split()
    report = run_quality_gate(str(MODEL_DIR / "v2.pkl"), X, y, sensitive_sex(X), max_demographic_parity_gap=0.0)
    assert report["accuracy"] >= 0.80
    assert report["fairness_passed"] is False
    assert report["passed"] is False


def test_gate_rejects_tampered_artifact(tmp_path):
    # Keep the genuine manifest but corrupt the artifact. Gate must reject before unpickling.
    (tmp_path / "model_summary.json").write_bytes((MODEL_DIR / "model_summary.json").read_bytes())
    artifact = tmp_path / "v2.pkl"
    artifact.write_bytes(b"corrupted artifact")
    result = subprocess.run([sys.executable, "-m", "src.evaluation.pre_deploy_gate", "--artifact", str(artifact)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "integrity" in result.stdout


def test_drift_endpoint_with_real_adult_events(client, auth_headers):
    frame, _ = load_adult()
    for row in frame.dropna().head(30).to_dict("records"):
        assert client.post("/predict", json=row, headers=auth_headers).status_code == 200
    response = client.get("/drift", headers=auth_headers)
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert len(response.json["features"]) == 14


def test_live_routing_changes_without_restart(client, auth_headers):
    response = client.post("/admin/routing", json={"v2_weight": 100}, headers=auth_headers)
    assert response.status_code == 200
    for _ in range(3):
        assert client.post("/predict", json=EXAMPLE, headers=auth_headers).json["model_version"] == "v2"
    assert client.post("/admin/routing", json={"v2_weight": 101}, headers=auth_headers).status_code == 400
    assert client.post("/admin/routing", json={"v2_weight": 0}).status_code == 401


def test_shadow_audit_has_separate_record(client, auth_headers, app):
    assert client.post("/admin/routing", json={"v2_weight": 0, "shadow_mode": True, "shadow_percentage": 100}, headers=auth_headers).status_code == 200
    response = client.post("/predict", json=EXAMPLE, headers=auth_headers)
    assert response.json["model_version"] == "v1" and response.json["shadow"]
    app.extensions["traffic_router"]._shadow_executor.submit(lambda: None).result(timeout=10)
    import src.api.app as module
    from sqlalchemy import select
    with module._Session() as session:
        logs = session.execute(select(module.PredictionLog).where(module.PredictionLog.request_id == response.json["request_id"])).scalars().all()
        assert {(row.model_version, row.shadow) for row in logs} == {("v1", 0), ("v2", 1)}


def test_slow_shadow_does_not_block_production():
    started, release = threading.Event(), threading.Event()

    class SlowCandidate:
        def predict(self, x):
            started.set()
            assert release.wait(timeout=10)
            return np.array([0])

    production = joblib.load(MODEL_DIR / "v1.pkl")
    router = TrafficRouter({"models": {"v1": {"weight": 100}, "v2": {"weight": 0}},
                            "routing": {"shadow_mode": True, "shadow_percentage": 100}},
                           {"v1": production, "v2": SlowCandidate()})
    try:
        result, used = router.predict(np.array([EXAMPLE[f] for f in FEATURE_NAMES], dtype=object))
        assert used == "v1" and result["shadow"]
        assert started.wait(timeout=5)
        assert router.get_metrics()["v2"]["requests"] == 0
    finally:
        release.set()
        router._shadow_executor.shutdown(wait=True)

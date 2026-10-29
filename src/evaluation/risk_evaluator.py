"""Component 2: Model Risk Evaluation & Quality Assurance Gate.

Provides pre-deployment checks:
  * artifact load test
  * accuracy / performance on a hold-out set
  * fairness / bias proxy (demographic parity across a sensitive slice)
  * stress test against adversarial / edge-case inputs
  * reproducibility: deterministic outputs for fixed test vectors
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


@dataclass
class RiskReport:
    model_artifact: str
    artifact_loads: bool
    accuracy: Optional[float]
    per_class_accuracy: Dict[str, float]
    fairness: Dict[str, float]
    fairness_passed: bool
    stress_test_passed: bool
    stress_failures: int
    reproducible: bool
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_model(artifact_path: str):
    """Load a persisted model artifact; raises if it cannot be loaded."""
    return joblib.load(artifact_path)


def _predict_safe(model, x: np.ndarray):
    """Run prediction, returning None on any exception (for stress testing)."""
    try:
        if hasattr(model, "predict_proba"):
            return int(model.predict(x.reshape(1, -1))[0])
        return int(model.predict(x.reshape(1, -1))[0])
    except Exception:
        return None


def _stress_vector_handled(model, x: np.ndarray) -> bool:
    """A stress vector is 'handled' when the model either predicts cleanly or
    rejects non-finite input gracefully. Crashing on finite input is a failure."""
    non_finite = any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in x)
    try:
        model.predict(x.reshape(1, -1))
        return True
    except Exception:
        # An exception on non-finite input is an acceptable rejection
        return non_finite


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive: Optional[np.ndarray] = None,
    artifact_path: str = "",
    max_demographic_parity_gap: Optional[float] = None,
    minimum_accuracy: float = 0.80,
) -> RiskReport:
    """Run the full QA / risk gate suite on a candidate model."""

    # ---- 1. artifact load ------------------------------------------------ #
    artifact_loads = True
    try:
        if artifact_path:
            load_model(artifact_path)
    except Exception:
        artifact_loads = False

    # ---- 2. accuracy ----------------------------------------------------- #
    accuracy = None
    per_class: Dict[str, float] = {}
    if artifact_loads and X_test is not None and y_test is not None:
        try:
            preds = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, preds))
            for cls in np.unique(y_test):
                mask = y_test == cls
                per_class[f"class_{int(cls)}"] = round(
                    float(accuracy_score(y_test[mask], preds[mask])), 4
                )
        except Exception:
            accuracy = None

    # ---- 3. fairness proxy: demographic parity across sensitive slice ---- #
    fairness: Dict[str, float] = {}
    fairness_passed = True
    if artifact_loads and sensitive is not None and len(sensitive) == len(X_test):
        preds = model.predict(X_test)
        for group in np.unique(sensitive):
            mask = sensitive == group
            if mask.sum() == 0:
                continue
            pos_rate = float((preds[mask] == 1).mean()) if preds.max() >= 1 else 0.0
            fairness[f"group_{int(group)}_positive_rate"] = round(pos_rate, 4)
        if len(fairness) >= 2:
            rates = list(fairness.values())
            fairness["demographic_parity_gap"] = round(float(max(rates) - min(rates)), 4)
            if max_demographic_parity_gap is not None:
                fairness_passed = fairness["demographic_parity_gap"] <= max_demographic_parity_gap

    # ---- 4. stress test: adversarial / edge inputs must not crash ------- #
    # Non-finite inputs raising a clean rejection is *graceful* handling - the
    # security gate blocks them before they reach the model. Only finite inputs
    # failing (or a total crash with no rejection) count as stress failures.
    stress_failures = 0
    probe = np.asarray(X_test[0], dtype=object)
    numeric_positions = [i for i, value in enumerate(probe) if isinstance(value, (int, float, np.number))]
    stress_vectors = [probe.copy()]
    for magnitude in (0.0, 1000.0, -1000.0, np.nan, np.inf):
        vector = probe.copy()
        for i in numeric_positions:
            vector[i] = magnitude
        stress_vectors.append(vector)
    for vec in stress_vectors:
        handled = _stress_vector_handled(model, vec)
        if not handled:
            stress_failures += 1

    # ---- 5. reproducibility: same input -> same output, N times ---------- #
    reproducible = True
    if artifact_loads:
        first = _predict_safe(model, probe)
        reproducible = first is not None
        for _ in range(9):
            if _predict_safe(model, probe) != first:
                reproducible = False
                break

    # ---- gate decision --------------------------------------------------- #
    passed = bool(
        artifact_loads
        and accuracy is not None
        and accuracy >= minimum_accuracy
        and stress_failures == 0
        and reproducible
        and fairness_passed
    )

    return RiskReport(
        model_artifact=artifact_path,
        artifact_loads=artifact_loads,
        accuracy=round(accuracy, 4) if accuracy is not None else None,
        per_class_accuracy=per_class,
        fairness=fairness,
        fairness_passed=fairness_passed,
        stress_test_passed=stress_failures == 0,
        stress_failures=stress_failures,
        reproducible=reproducible,
        passed=passed,
    )


def run_quality_gate(
    artifact_path: str,
    X_test,
    y_test,
    sensitive=None,
    max_demographic_parity_gap: Optional[float] = None,
    minimum_accuracy: float = 0.80,
) -> Dict[str, Any]:
    """Convenience entry point used by tests and CI."""
    model = load_model(artifact_path)
    report = evaluate_model(
        model, X_test, y_test, sensitive, artifact_path, max_demographic_parity_gap, minimum_accuracy
    )
    return report.to_dict()

"""Component 3: Dynamic Routing Layer (Canary & Shadow Deployment).

Responsibilities:
  * Canary routing: weighted traffic split between production (v1) and candidate (v2)
  * Shadow routing: 100% of traffic served by v1 while duplicating payloads to v2
    in the background (dark traffic) - end users are never affected
  * Automated rollback: monitor error rate & latency; revert 100% to v1 on degradation
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


class TrafficRouter:
    def __init__(self, config: Dict[str, Any], models: Dict[str, object]):
        self.config = config
        self.models = models  # e.g. {"v1": model_obj, "v2": model_obj}

        models_cfg = config.get("models", {})
        self.weights: Dict[str, float] = {
            name: float(info.get("weight", 0))
            for name, info in models_cfg.items()
            if name in self.models
        }
        self.total_weight = sum(self.weights.values())

        routing = config.get("routing", {})
        self.shadow_mode = bool(routing.get("shadow_mode", False))
        # Earlier project configs used 0 while shadow mode meant "mirror every
        # request". Preserve that behavior while making partial mirroring
        # available with explicit values 1..100.
        self.shadow_percentage = float(routing.get("shadow_percentage", 100)) or 100.0

        rollback = config.get("rollback", {})
        self.rollback_enabled = bool(rollback.get("enabled", True))
        self.error_rate_threshold = float(rollback.get("error_rate_threshold", 0.05))
        self.latency_threshold_ms = float(rollback.get("latency_threshold_ms", 500))
        self.min_requests = int(rollback.get("min_requests", 20))
        configured_monitored = rollback.get("monitor_models")
        self.monitored_models = set(configured_monitored or [
            name for name, info in models_cfg.items()
            if info.get("status") in {"canary", "candidate"}
        ])

        self._lock = threading.Lock()
        # Metrics keys must cover every configured model (even if its artifact
        # failed to load) plus a "v1" fallback used by rollback selection.
        metric_names = set(self.weights.keys()) | set(self.models.keys()) | {"v1"}
        self._metrics: Dict[str, Dict[str, Any]] = {
            name: {"requests": 0, "errors": 0, "latencies": deque(maxlen=1000)}
            for name in metric_names
        }
        self._active_rollback = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _record(self, model_name: str, latency_ms: float, error: bool) -> None:
        with self._lock:
            m = self._metrics[model_name]
            m["requests"] += 1
            if error:
                m["errors"] += 1
            m["latencies"].append(latency_ms)

    def _should_rollback(self, model_name: str) -> bool:
        if not self.rollback_enabled or self._active_rollback:
            return False
        # A production-model outage needs its own incident path.  Rolling a
        # candidate back because v1 failed would hide the real problem.
        if model_name not in self.monitored_models:
            return False
        m = self._metrics[model_name]
        if m["requests"] < self.min_requests:
            return False
        error_rate = m["errors"] / m["requests"]
        recent = list(m["latencies"])[-50:]
        avg_latency = sum(recent) / len(recent) if recent else 0.0
        if error_rate > self.error_rate_threshold:
            return True
        if avg_latency > self.latency_threshold_ms:
            return True
        return False

    def _activate_rollback(self) -> None:
        with self._lock:
            self._active_rollback = True

    def _select_model(self) -> str:
        """Pick the model to serve for the current request."""
        if self._active_rollback:
            return "v1"
        if self.shadow_mode:
            # Active model serves; candidate only shadows duplicates
            return "v1"
        # Canary: weighted random split
        names = [name for name, weight in self.weights.items() if weight > 0]
        weights = [self.weights[n] for n in names]
        if self.total_weight <= 0 or not names:
            return names[0] if names else "v1"
        return random.choices(names, weights=weights, k=1)[0]

    @staticmethod
    def _predict_with(model: object, features: np.ndarray) -> Dict[str, Any]:
        """Run a single model prediction, returning a normalized dict."""
        x = features.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            pred = model.predict(x)
            return {
                "prediction": int(pred[0]),
                "probabilities": [round(float(p), 4) for p in proba[0]],
            }
        pred = model.predict(x)
        return {"prediction": int(pred[0])}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, features: np.ndarray) -> Tuple[Dict[str, Any], str]:
        """Route one inference request.

        Returns (result, model_used). result contains the prediction, the
        model version, latency in ms, and a shadow flag.
        """
        start = time.perf_counter()
        model_name = self._select_model()
        shadow = False
        error = False
        prediction: Dict[str, Any] = {}

        try:
            if model_name not in self.models:
                raise RuntimeError(f"No loaded artifact for routed model '{model_name}'")
            prediction = self._predict_with(self.models[model_name], features)

            # Shadow routing: duplicate to the candidate model in the background.
            # The shadow call's result is discarded - it never reaches the user.
            if (
                self.shadow_mode
                and model_name == "v1"
                and "v2" in self.models
                and random.random() < self.shadow_percentage / 100
            ):
                shadow_start = time.perf_counter()
                try:
                    self._predict_with(self.models["v2"], features)
                    shadow = True
                    self._record("v2", (time.perf_counter() - shadow_start) * 1000.0, False)
                except Exception:
                    # Shadow failures must not affect the live response
                    self._record("v2", (time.perf_counter() - shadow_start) * 1000.0, True)
                if self._should_rollback("v2"):
                    self._activate_rollback()
        except Exception as exc:  # noqa: BLE001 - record and return a safe error
            error = True
            prediction = {"error": str(exc)}

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record(model_name, latency_ms, error)

        # Continuous rollback evaluation on the model that just served
        if self._should_rollback(model_name):
            self._activate_rollback()

        return {
            "prediction": prediction,
            "model_version": model_name,
            "latency_ms": round(latency_ms, 2),
            "shadow": shadow,
        }, model_name

    def get_metrics(self) -> Dict[str, Any]:
        """Snapshot of per-model traffic, error, and latency metrics."""
        with self._lock:
            out: Dict[str, Any] = {}
            for name, m in self._metrics.items():
                recent = list(m["latencies"])[-100:]
                out[name] = {
                    "requests": m["requests"],
                    "errors": m["errors"],
                    "error_rate": round(m["errors"] / m["requests"], 4)
                    if m["requests"]
                    else 0.0,
                    "avg_latency_ms": round(sum(recent) / len(recent), 2)
                    if recent
                    else 0.0,
                }
            out["rollback_active"] = self._active_rollback
            out["weights"] = dict(self.weights)
            out["shadow_mode"] = self.shadow_mode
        return out

    def reset_rollback(self) -> None:
        """Manually clear an active rollback (e.g., after a hotfix deploy)."""
        with self._lock:
            self._active_rollback = False

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Dynamically re-balance canary weights at runtime (zero-downtime)."""
        with self._lock:
            self.weights = {k: float(v) for k, v in weights.items()}
            self.total_weight = sum(self.weights.values())

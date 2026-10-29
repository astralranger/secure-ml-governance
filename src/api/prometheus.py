"""Small dependency-free Prometheus exposition formatter for router metrics."""

from __future__ import annotations

from typing import Any


def _label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_router_metrics(metrics: dict[str, Any]) -> str:
    """Render the router's snapshot in Prometheus text exposition format."""
    lines = [
        "# HELP ml_governance_model_requests_total Inference requests by model, including shadow evaluations.",
        "# TYPE ml_governance_model_requests_total counter",
        "# HELP ml_governance_model_errors_total Failed inferences by served model.",
        "# TYPE ml_governance_model_errors_total counter",
        "# HELP ml_governance_model_average_latency_ms Recent average inference latency.",
        "# TYPE ml_governance_model_average_latency_ms gauge",
    ]
    for model, values in metrics.items():
        if not isinstance(values, dict) or "requests" not in values:
            continue
        labels = f'{{model_version="{_label(model)}"}}'
        lines.extend([
            f"ml_governance_model_requests_total{labels} {values['requests']}",
            f"ml_governance_model_errors_total{labels} {values['errors']}",
            f"ml_governance_model_average_latency_ms{labels} {values['avg_latency_ms']}",
        ])
    lines.extend([
        "# HELP ml_governance_rollback_active Whether automatic candidate rollback is active.",
        "# TYPE ml_governance_rollback_active gauge",
        f"ml_governance_rollback_active {int(bool(metrics.get('rollback_active', False)))}",
        "# HELP ml_governance_model_traffic_weight Configured traffic weight by model.",
        "# TYPE ml_governance_model_traffic_weight gauge",
    ])
    for model, weight in metrics.get("weights", {}).items():
        lines.append(f'ml_governance_model_traffic_weight{{model_version="{_label(model)}"}} {weight}')
    return "\n".join(lines) + "\n"

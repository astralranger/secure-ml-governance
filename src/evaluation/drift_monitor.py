"""Component 4: Continuous Drift Monitoring.

Computes statistical distance between live inference inputs and a training
baseline using two-sample KS tests (numeric) and chi-square tests (categorical). Optionally
generates an Evidently AI drift report when the `evidently` package is
installed; otherwise falls back to pure SciPy statistics so the module works
without heavy optional dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


@dataclass
class DriftResult:
    feature: str
    statistic: float
    p_value: float
    drift_detected: bool
    baseline_mean: Optional[float]
    live_mean: Optional[float]
    test: str = "ks_2samp"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    timestamp: str
    n_baseline: int
    n_live: int
    features: List[DriftResult]
    overall_drift: bool
    evidentl_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "n_baseline": self.n_baseline,
            "n_live": self.n_live,
            "features": [f.to_dict() for f in self.features],
            "overall_drift": self.overall_drift,
            "evidentl_report_path": self.evidentl_report_path,
        }


def compute_drift(
    baseline: pd.DataFrame,
    live: pd.DataFrame,
    feature_names: List[str],
    alpha: float = 0.05,
) -> DriftReport:
    """Per-feature KS test between baseline and live distributions.

    A feature is flagged as drifted when the KS test p-value < alpha, meaning
    the two samples are unlikely to be drawn from the same distribution.
    """
    from datetime import datetime, timezone

    results: List[DriftResult] = []
    for feat in feature_names:
        if not pd.api.types.is_numeric_dtype(baseline[feat]):
            b_counts = baseline[feat].fillna("__MISSING__").value_counts()
            l_counts = live[feat].fillna("__MISSING__").value_counts()
            categories = b_counts.index.union(l_counts.index)
            table = np.array([b_counts.reindex(categories, fill_value=0),
                              l_counts.reindex(categories, fill_value=0)])
            if table.shape[1] <= 1 or not table.sum(axis=1).all():
                stat, p = 0.0, 1.0
            else:
                stat, p, _, _ = chi2_contingency(table)
            results.append(DriftResult(feat, round(float(stat), 4), round(float(p), 6),
                                       bool(p < alpha), None, None, "chi_square"))
            continue
        b = baseline[feat].dropna().values
        l = live[feat].dropna().values
        if len(b) == 0 or len(l) == 0:
            results.append(
                DriftResult(
                    feature=feat,
                    statistic=float("nan"),
                    p_value=float("nan"),
                    drift_detected=False,
                    baseline_mean=float(np.nanmean(b)) if len(b) else float("nan"),
                    live_mean=float(np.nanmean(l)) if len(l) else float("nan"),
                )
            )
            continue
        stat, p = ks_2samp(b, l)
        results.append(
            DriftResult(
                feature=feat,
                statistic=round(float(stat), 4),
                p_value=round(float(p), 6),
                drift_detected=bool(p < alpha),
                baseline_mean=round(float(b.mean()), 4),
                live_mean=round(float(l.mean()), 4),
            )
        )

    overall = any(r.drift_detected for r in results)
    return DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_baseline=int(len(baseline)),
        n_live=int(len(live)),
        features=results,
        overall_drift=overall,
    )


def try_evidently_report(
    baseline: pd.DataFrame,
    live: pd.DataFrame,
    output_path: str = "drift_report.html",
) -> Optional[str]:
    """Generate an Evidently AI data-drift report if available.

    Returns the path to the HTML report, or None when Evidently is not
    installed (graceful degradation).
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline, current_data=live)
        report.save_html(output_path)
        return output_path
    except Exception:
        return None


def run_drift_check(
    baseline_path: str,
    live_df: pd.DataFrame,
    feature_names: List[str],
    alpha: float = 0.05,
    use_evidently: bool = False,
) -> Dict[str, Any]:
    """Convenience: load baseline CSV, run drift check, optionally render Evidently."""
    baseline = pd.read_csv(baseline_path)
    report = compute_drift(baseline, live_df, feature_names, alpha=alpha)

    evidently_path = None
    if use_evidently:
        evidently_path = try_evidently_report(baseline, live_df)
        report.evidentl_report_path = evidently_path

    return report.to_dict()

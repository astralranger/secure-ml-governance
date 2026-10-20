"""Runnable candidate-model admission gate used by CI/CD.

The test suite proves the gate implementation works.  This module evaluates
the actual candidate artifact and returns a non-zero exit code when it must
not be deployed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from .risk_evaluator import run_quality_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the candidate pre-deploy quality gate")
    parser.add_argument("--artifact", default="src/models/v2.pkl")
    parser.add_argument("--minimum-accuracy", type=float, default=0.80)
    parser.add_argument("--maximum-demographic-parity-gap", type=float, default=0.20)
    args = parser.parse_args()

    iris = load_iris(as_frame=True)
    feature_columns = list(iris.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.frame[feature_columns].values,
        iris.target.values,
        test_size=0.2,
        random_state=42,
        stratify=iris.target.values,
    )
    # A transparent demo-sensitive slice; real deployments should provide a
    # reviewed, legally appropriate sensitive attribute instead.
    sensitive = (X_test[:, 0] >= X_train[:, 0].mean()).astype(int)
    report = run_quality_gate(
        str(Path(args.artifact)), X_test, y_test, sensitive,
        max_demographic_parity_gap=args.maximum_demographic_parity_gap,
    )
    report["minimum_accuracy"] = args.minimum_accuracy
    report["maximum_demographic_parity_gap"] = args.maximum_demographic_parity_gap
    report["passed"] = bool(report["passed"] and report["accuracy"] >= args.minimum_accuracy)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

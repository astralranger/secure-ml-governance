"""Runnable candidate-model admission gate used by CI/CD.

The test suite proves the gate implementation works.  This module evaluates
the actual candidate artifact and returns a non-zero exit code when it must
not be deployed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..models.dataset import training_split, sensitive_sex, SCHEMA_VERSION, DATA_DIR
from ..models.metadata import read_manifest, sha256_file

from .risk_evaluator import run_quality_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the candidate pre-deploy quality gate")
    parser.add_argument("--artifact", default="src/models/v2.pkl")
    parser.add_argument("--minimum-accuracy", type=float, default=0.80)
    parser.add_argument("--maximum-demographic-parity-gap", type=float, default=0.20)
    args = parser.parse_args()

    artifact = Path(args.artifact)
    manifest = read_manifest(artifact.parent / "model_summary.json") or {}
    record = manifest.get("models", {}).get(artifact.stem, {})
    valid = (artifact.exists() and record.get("artifact_sha256") == sha256_file(artifact)
             and record.get("feature_schema_version") == SCHEMA_VERSION)
    for name, expected in manifest.get("dataset", {}).get("source_sha256", {}).items():
        valid = valid and (DATA_DIR / name).exists() and sha256_file(DATA_DIR / name) == expected
    baseline = artifact.parent / "adult_baseline.csv"
    valid = valid and baseline.exists() and sha256_file(baseline) == manifest.get("dataset", {}).get("baseline_sha256")
    if not valid:
        print(json.dumps({"passed": False, "error": "Artifact, schema, or dataset integrity verification failed"}))
        return 1
    _, X_test, _, y_test = training_split()
    sensitive = sensitive_sex(X_test)
    report = run_quality_gate(
        str(Path(args.artifact)), X_test, y_test, sensitive,
        max_demographic_parity_gap=args.maximum_demographic_parity_gap,
        minimum_accuracy=args.minimum_accuracy,
    )
    report["integrity_verified"] = True
    report["sensitive_attribute"] = "sex (0=Female, 1=Male)"
    report["minimum_accuracy"] = args.minimum_accuracy
    report["maximum_demographic_parity_gap"] = args.maximum_demographic_parity_gap
    report["passed"] = bool(report["passed"] and report["accuracy"] is not None and report["accuracy"] >= args.minimum_accuracy)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

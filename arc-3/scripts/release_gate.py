"""Read-only release evidence gate. Does not upload, submit, or grant approval.

Checks completeness and artifact binding, not the truth of human assessments.
Official rules review and actual evaluation remain independent requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = (
    "arc3_rules", "licenses_disclosure", "data_permissions", "runtime_offline",
    "dynamics_weights", "reasoner_weights", "model_integration",
    "same_game_learning", "heldout_evaluation", "regression_tests",
    # Substantive performance gate (trustworthy-internal-game milestone, Step 5):
    # improved LEVEL COMPLETION or ACTION EFFICIENCY on UNTOUCHED dev games, across
    # REPEATED SEEDS, against MATCHED component-disabled controls. Controlled-world
    # results (Steps 1-4/7) do NOT satisfy this — it requires the improvements wired
    # into the LIVE agent and measured on real games the tuning never touched.
    "substantive_improvement",
    "competition_rehearsal", "user_release_approval",
)
BOUND_INPUTS = ("agent/my_agent.py", "notebooks/submission.ipynb",
                "notebooks/kernel-metadata.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_file(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Missing relative artifact path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must stay within project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or out-of-project artifact")
    return path


def evaluate(root, manifest):
    errors = []
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        return ["Invalid release manifest schema"]
    if manifest.get("competition") != "arc-prize-2026-arc-agi-3":
        errors.append("Wrong competition")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return errors + ["Invalid artifacts mapping"]
    for name in BOUND_INPUTS:
        if name not in artifacts:
            errors.append(f"Unbound release input: {name}")
    for name, expected in artifacts.items():
        try:
            if sha256(local_file(root, name)) != expected:
                errors.append(f"Changed artifact: {name}")
        except (OSError, ValueError):
            errors.append(f"Invalid artifact: {name}")
    checks = manifest.get("checks", {})
    if not isinstance(checks, dict):
        return errors + ["Invalid checks mapping"]
    for check in REQUIRED:
        record = checks.get(check)
        if not isinstance(record, dict) or record.get("status") != "PASS":
            errors.append(f"Unresolved: {check}")
            continue
        evidence = record.get("evidence")
        if not isinstance(evidence, str) or evidence not in artifacts:
            errors.append(f"Unbound evidence: {check}")
        if not isinstance(record.get("reviewed_by"), str) or not record["reviewed_by"].strip():
            errors.append(f"Missing reviewer: {check}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "release/readiness.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        errors = evaluate(root, json.loads(args.manifest.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        errors = [f"Cannot read valid release evidence: {type(exc).__name__}"]
    print(json.dumps({"status": "BLOCKED" if errors else "PASS", "issues": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

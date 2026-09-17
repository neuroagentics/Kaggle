import json
import subprocess
import sys
from pathlib import Path

from scripts.release_gate import REQUIRED, BOUND_INPUTS, evaluate, sha256


def ready_fixture(tmp_path):
    # Synthetic evidence tests gate mechanics, never release eligibility.
    artifacts = {}
    for name in (*BOUND_INPUTS, "evidence/report.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
        artifacts[name] = sha256(path)
    return {"schema_version": 1, "competition": "arc-prize-2026-arc-agi-3",
            "artifacts": artifacts,
            "checks": {key: {"status": "PASS", "reviewed_by": "test-only",
                             "evidence": "evidence/report.json"} for key in REQUIRED}}


def test_complete_fixture_and_changed_input(tmp_path):
    manifest = ready_fixture(tmp_path)
    assert evaluate(tmp_path, manifest) == []
    (tmp_path / BOUND_INPUTS[0]).write_text("changed", encoding="utf-8")
    assert any("Changed artifact" in e for e in evaluate(tmp_path, manifest))


def test_missing_rule_review_or_unbound_evidence_blocks(tmp_path):
    manifest = ready_fixture(tmp_path)
    manifest["checks"]["arc3_rules"]["status"] = "PENDING"
    manifest["checks"]["dynamics_weights"]["evidence"] = "missing"
    errors = evaluate(tmp_path, manifest)
    assert "Unresolved: arc3_rules" in errors
    assert "Unbound evidence: dynamics_weights" in errors


def test_wrong_competition_and_path_escape(tmp_path):
    manifest = ready_fixture(tmp_path)
    manifest["competition"] = "arc-prize-2026-arc-agi-2"
    manifest["artifacts"]["../outside"] = "abc"
    errors = evaluate(tmp_path, manifest)
    assert "Wrong competition" in errors
    assert "Invalid artifact: ../outside" in errors


def test_current_manifest_fails_closed_and_submit_uses_gate():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/release_gate.py")],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "BLOCKED"
    recipe = (root / "Makefile").read_text().split("submit: notebook", 1)[1].split("status:", 1)[0]
    assert recipe.index("scripts/release_gate.py") < recipe.index("kernels push")

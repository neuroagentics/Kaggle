"""Ensure the active notebook cannot silently drift from its build inputs."""
import json
from pathlib import Path

import pytest

from build_release import build_release
from hyper_arc.contender.agentic_memory import AgenticMemoryRecord


def test_active_notebook_matches_generated_source_and_release_identity(tmp_path):
    record = build_release(tmp_path)
    current = Path(__file__).parents[1] / "release/current"
    assert (tmp_path / "notebook6a96ea823f.ipynb").read_bytes() == (current / "notebook6a96ea823f.ipynb").read_bytes()
    saved = json.loads((current / "release.json").read_text())
    assert saved["source_tree_sha256"] == record["source_tree_sha256"]
    assert saved["notebook_sha256"] == record["notebook_sha256"]
    assert saved["status"] == "UNQUALIFIED_CANDIDATE"


@pytest.mark.parametrize("scope,outputs,message", [
    ("validation", [[[2]]], "builder-only"),
    ("builder", [[[9]]], "individually full-task exact"),
])
def test_static_memory_cannot_invent_success_or_promote_validation(scope, outputs, message):
    with pytest.raises(ValueError, match=message):
        AgenticMemoryRecord.create(source_task_id="a", summary="identity",
            code="def transform(grid):\n    return grid", model="stub",
            task_data={"train": [{"input": [[1]], "output": [[1]]}],
                       "test": [{"input": [[2]]}]}, validation_scope=scope,
            expected_test_outputs=outputs)


def test_failure_memory_is_available_to_the_next_development_task(tmp_path, monkeypatch):
    import main as runtime
    from hyper_arc.contender.failure_memory import FailureRecord
    task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}
    seen = []
    def solver(*args):
        bank, learned = args[11], args[12]
        seen.append(len(bank.records))
        learned.append(FailureRecord.create(source_task_id=args[0], family="identity",
                       reason="demo mismatch", residual=1, task_data=args[1]))
        return [{"attempt_1": [[2]], "attempt_2": [[2]]}], 1.0, True, 1
    monkeypatch.setattr(runtime, "solve_task", solver)
    challenges = tmp_path / "challenges.json"
    challenges.write_text(json.dumps({"a": task, "b": task}))
    memory_path = tmp_path / "static_failures.json"
    runtime.main(["--challenge-file", str(challenges), "--output", str(tmp_path / "submission.json"),
                  "--checkpoint", str(tmp_path / "checkpoint.pt"), "--enable-failure-memory",
                  "--failure-memory", str(memory_path), "--session-transfer", "--no-resume",
                  "--task-failure-policy", "abort"])
    assert seen == [0, 1]
    assert not memory_path.exists()  # the static bank was not overwritten

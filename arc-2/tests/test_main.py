"""End-to-end orchestration and Kaggle submission regression tests."""

from __future__ import annotations

import pytest

from hyper_arc.hpm import GlobalMemoryBank
from hyper_arc.contender.experience_bank import empty_experience_bank
from hyper_arc.contender.agentic_memory import AgenticMemoryBank, AgenticMemoryRecord
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)
import json
from pathlib import Path
from types import SimpleNamespace
import time

from build_global_memory import load_tasks
from main import load_checkpoint, main, save_checkpoint, solve_task, validate_submission


def identity_task() -> dict:
    return {
        "train": [
            {"input": [[1, 0]], "output": [[1, 0]]},
            {"input": [[2], [0]], "output": [[2], [0]]},
        ],
        "test": [
            {"input": [[3, 0]]},
            {"input": [[4], [0]]},
        ],
    }


def test_solve_task_emits_one_two_attempt_object_per_test_input():
    task = identity_task()
    attempts, score, exact, program_len = solve_task(
        "identity",
        task,
        GlobalMemoryBank(),
        global_prior_weight=0.0,
        timeout_sec=1.0,
        max_iterations=1,
    )

    assert attempts == [
        {"attempt_1": [[3, 0]], "attempt_2": [[3, 0]]},
        {"attempt_1": [[4], [0]], "attempt_2": [[4], [0]]},
    ]
    assert score == 1.0
    assert exact is True
    assert program_len == 0
    validate_submission({"identity": attempts}, {"identity": task})


def test_competition_mode_refuses_non_agentic_configuration():
    with pytest.raises(ValueError, match="enable-agentic-ai"):
        main(["--competition-run"])


def test_solve_task_reuses_exact_model_authored_memory():
    task = {
        "train": [
            {
                "input": [[1, 1, 1], [2, 2, 2]],
                "output": [[1, 2, 1], [2, 1, 2]],
            }
        ],
        "test": [{"input": [[3, 3, 3], [4, 4, 4]]}],
    }
    code = (
        "def transform(grid):\n"
        "    a = grid[0][0]\n"
        "    b = grid[1][0]\n"
        "    return [[a if (r+c)%2 == 0 else b for c in range(len(grid[0]))] "
        "for r in range(len(grid))]"
    )
    record = AgenticMemoryRecord.create(
        source_task_id="source",
        summary="checkerboard from two row colors",
        code=code,
        model="test",
        task_data=task,
        validation_scope="builder",
        expected_test_outputs=[[[3, 4, 3], [4, 3, 4]]],
    )

    attempts, score, exact, _ = solve_task(
        "memory-target",
        task,
        GlobalMemoryBank(),
        global_prior_weight=0.0,
        timeout_sec=1.0,
        max_iterations=1,
        agentic_memory=AgenticMemoryBank((record,)),
    )

    assert attempts[0]["attempt_1"] == [[3, 4, 3], [4, 3, 4]]
    assert score == 1.0 and exact is True


def test_solve_task_uses_exact_relational_plan_as_upside_attempt():
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 2, 2, 0],
                    [0, 2, 0, 2, 0],
                    [0, 2, 2, 2, 0],
                ],
                "output": [
                    [0, 2, 2, 2, 0],
                    [0, 2, 0, 2, 0],
                    [0, 2, 2, 2, 0],
                    [0, 2, 2, 2, 0],
                    [0, 2, 0, 2, 0],
                    [0, 2, 2, 2, 0],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 4, 4, 4, 0],
                    [0, 4, 0, 4, 0],
                    [0, 4, 4, 4, 0],
                ]
            }
        ],
    }
    expected = [
        [0, 4, 4, 4, 0],
        [0, 4, 0, 4, 0],
        [0, 4, 4, 4, 0],
        [0, 4, 4, 4, 0],
        [0, 4, 0, 4, 0],
        [0, 4, 4, 4, 0],
    ]

    attempts, score, exact, _ = solve_task(
        "relational",
        task,
        GlobalMemoryBank(),
        global_prior_weight=0.0,
        timeout_sec=1.0,
        max_iterations=1,
    )

    assert expected in attempts[0].values()
    assert score == 1.0
    assert exact is True


def test_solve_task_uses_recursive_world_model_as_upside_attempt():
    task = {
        "train": [
            {
                "input": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
            }
        ],
        "test": [{"input": [[0, 2, 0], [0, 0, 0], [0, 0, 0]]}],
    }
    expected = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    model = RecursiveReasoningWorldModel(
        config=WorldModelConfig(
            max_depth=1,
            max_memory_seeds=0,
            learn_memory=False,
        )
    )

    attempts, score, exact, _ = solve_task(
        "world-model",
        task,
        GlobalMemoryBank(),
        global_prior_weight=0.0,
        timeout_sec=1.0,
        max_iterations=1,
        world_model=model,
    )

    assert expected in attempts[0].values()
    assert score == 1.0
    assert exact is True


def test_unresolved_task_attempts_neural_induction_before_recursive_repair(monkeypatch):
    import main as entrypoint

    monkeypatch.setattr(entrypoint, "exact_candidates", lambda *_: [])
    monkeypatch.setattr(entrypoint, "exact_relational_candidates", lambda *_: [])
    calls = []
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }

    class Agent:
        def solve(self, *_args, **_kwargs):
            calls.append("induce-iterate")
            assert 0 < _kwargs["deadline"] - time.monotonic() < 4.0
            return SimpleNamespace(hypotheses=(), failures=())

    class World:
        def solve(self, *_args, **_kwargs):
            calls.append("recurse")
            return SimpleNamespace(
                hypotheses=(SimpleNamespace(complexity=1),),
                test_predictions=((((4,),),),),
            )

    attempts, _, exact, _ = solve_task(
        "order", task, GlobalMemoryBank(), 0.0, 5.0, 1,
        agentic_reasoner=Agent(), world_model=World(),
    )
    assert calls == ["induce-iterate", "recurse"]
    assert exact and attempts[0]["attempt_1"] == [[4]]


def test_exact_neural_fit_does_not_spend_recursive_budget(monkeypatch):
    import main as entrypoint

    monkeypatch.setattr(entrypoint, "exact_candidates", lambda *_: [])
    monkeypatch.setattr(entrypoint, "exact_relational_candidates", lambda *_: [])
    calls = []
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]]}],
    }

    class Agent:
        def solve(self, *_args, **_kwargs):
            calls.append("induce-iterate")
            return SimpleNamespace(
                hypotheses=(SimpleNamespace(complexity=1),),
                test_predictions=((((4,),),),),
                failures=(),
            )

    class World:
        def solve(self, *_args, **_kwargs):
            calls.append("recurse")
            raise AssertionError("recursive repair should be skipped")

    attempts, _, exact, _ = solve_task(
        "order", task, GlobalMemoryBank(), 0.0, 5.0, 1,
        agentic_reasoner=Agent(), world_model=World(),
    )
    assert calls == ["induce-iterate"]
    assert exact and attempts[0]["attempt_1"] == [[4]]


def test_validate_submission_rejects_split_attempt_objects():
    task = identity_task()
    invalid = {
        "identity": [
            {"attempt_1": [[3, 0]]},
            {"attempt_2": [[3, 0]]},
        ]
    }
    with pytest.raises(ValueError):
        validate_submission(invalid, {"identity": task})


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.pt"
    bank = GlobalMemoryBank()
    bank.add([("rotate", {"degrees": 90})])
    submission = {"task": [{"attempt_1": [[1]], "attempt_2": [[1]]}]}
    save_checkpoint(path, submission, {"task"}, bank)

    restored_bank = GlobalMemoryBank()
    restored_submission, completed = load_checkpoint(path, restored_bank)

    assert restored_submission == submission
    assert completed == {"task"}
    assert len(restored_bank) == 1


def test_checkpoint_rejects_a_different_run_signature(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"task": []}, {"task"}, GlobalMemoryBank(), "first")

    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(
            path,
            GlobalMemoryBank(),
            expected_run_signature="second",
        )


def test_checkpoint_does_not_restore_disabled_memory(tmp_path):
    path = tmp_path / "checkpoint.pt"
    seeded = GlobalMemoryBank()
    seeded.add([("rotate", {"degrees": 90})])
    save_checkpoint(path, {}, set(), seeded, "same")
    disabled = GlobalMemoryBank()

    load_checkpoint(
        path,
        disabled,
        expected_run_signature="same",
        restore_memory=False,
    )

    assert len(disabled) == 0


def test_main_writes_valid_multi_test_submission(tmp_path):
    challenge_path = tmp_path / "arc-agi_test_challenges.json"
    output_path = tmp_path / "submission.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    challenge_path.write_text(
        json.dumps({"identity": identity_task()}), encoding="utf-8"
    )
    experience_path = tmp_path / "experience.json"
    empty_experience_bank().save(experience_path)
    world_memory_path = tmp_path / "world-memory.json"
    HyperbolicWorldMemory().save(world_memory_path)
    seed_path = tmp_path / "seed.json"
    seed = GlobalMemoryBank()
    seed.add([("rotate", {"degrees": 90})])
    seed.save(seed_path)

    result = main(
        [
            "--challenge-file",
            str(challenge_path),
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
            "--seed-bank",
            str(seed_path),
            "--enable-legacy-seed-memory",
            "--experience-bank",
            str(experience_path),
            "--world-memory",
            str(world_memory_path),
            "--max-iterations",
            "1",
            "--task-timeout",
            "1",
        ]
    )

    submission = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert len(submission["identity"]) == 2
    assert set(submission["identity"][0]) == {"attempt_1", "attempt_2"}
    assert not checkpoint_path.exists()
    run_manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["channels"]["seed_records"] == 1
    assert run_manifest["tasks"]["failures"] == []


def test_main_fails_closed_when_seed_bank_is_missing(tmp_path):
    challenge_path = tmp_path / "arc-agi_test_challenges.json"
    challenge_path.write_text(
        json.dumps({"identity": identity_task()}), encoding="utf-8"
    )
    experience_path = tmp_path / "experience.json"
    empty_experience_bank().save(experience_path)

    with pytest.raises(FileNotFoundError, match="Required seed bank"):
        main(
            [
                "--challenge-file",
                str(challenge_path),
                "--output",
                str(tmp_path / "submission.json"),
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--seed-bank",
                str(tmp_path / "missing.json"),
                "--enable-legacy-seed-memory",
                "--experience-bank",
                str(experience_path),
                "--max-iterations",
                "1",
            ]
        )


def test_world_model_can_run_without_unpromoted_world_memory(tmp_path):
    challenge_path = tmp_path / "arc-agi_test_challenges.json"
    challenge_path.write_text(
        json.dumps({"identity": identity_task()}), encoding="utf-8"
    )
    output_path = tmp_path / "submission.json"

    result = main(
        [
            "--challenge-file",
            str(challenge_path),
            "--output",
            str(output_path),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--world-memory",
            str(tmp_path / "missing-world-memory.json"),
            "--enable-world-model",
            "--max-iterations",
            "1",
            "--task-timeout",
            "1",
        ]
    )

    assert result == 0
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["world_model_enabled"] is True
    assert manifest["configuration"]["world_memory_enabled"] is False


def test_seed_builder_loads_tasks_inside_aggregate_file(tmp_path):
    tasks = {"task-a": identity_task(), "task-b": identity_task()}
    path = tmp_path / "arc-agi_training_challenges.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")

    assert load_tasks(tmp_path) == tasks


def test_notebook_bootstrap_is_hermetic_and_does_not_install_dependencies():
    notebook_path = Path(__file__).parents[1] / "release" / "current" / "notebook6a96ea823f.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])

    compile(source, str(notebook_path), "exec")
    assert "hyper_arc_solver_bundle.*.zip" in source
    assert "bundle_manifest.json" in source
    assert "hashlib.sha256" in source
    assert "bundle_self_test.py" in source
    assert "FAILED_NO_SUBMISSION" in source
    assert "FALLBACK_READY" not in source
    assert "DEGRADED_FALLBACK" not in source
    assert "submission.candidate.json" in source
    assert "CANDIDATE_PATH.replace(OUTPUT_PATH)" in source
    assert "hyper_arc_solver.log" in source
    assert "'--no-index', '--no-deps'" in source
    assert "Path('/tmp/hyper_arc_model_deps')" in source
    assert "geoopt" not in source
    assert "solver_code.zip" not in source
    assert "main.py plus hyper_arc" not in source
    assert "validate_submission" in source
    assert "--task-timeout', str(RELEASE['task_seconds'])" in source
    assert "qualify_run(run_manifest)" in source
    assert "--enable-agentic-ai" in source
    assert "--enable-agentic-memory" in source
    assert "--model-path" in source
    assert "CPU fallback is forbidden" in source
    assert "--seed-bank" not in source
    assert "--experience-bank" not in source
    assert "--world-memory" not in source


def test_notebook_emits_no_submission_when_agentic_runtime_is_missing(tmp_path):
    notebook_path = Path(__file__).parents[1] / "release" / "current" / "notebook6a96ea823f.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    input_root = tmp_path / "input"
    work_root = tmp_path / "working"
    competition = input_root / "arc-prize-2026-arc-agi-2"
    competition.mkdir(parents=True)
    work_root.mkdir()
    challenge_path = competition / "arc-agi_test_challenges.json"
    challenges = {"identity": identity_task()}
    challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
    model_path = input_root / "gemma-4-e4b-it" / "1"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    source = source.replace("Path('/kaggle/input')", f"Path({str(input_root)!r})")
    source = source.replace("Path('/kaggle/working')", f"Path({str(work_root)!r})")
    source = source.replace(
        "Path('/tmp/hyper_arc_runtime')", f"Path({str(tmp_path / 'runtime')!r})"
    )

    with pytest.raises(RuntimeError, match="No submission artifact"):
        exec(compile(source, str(notebook_path), "exec"), {})

    assert not (work_root / "submission.json").exists()
    status = json.loads((work_root / "deployment_status.json").read_text())
    assert status["status"] == "FAILED_NO_SUBMISSION"

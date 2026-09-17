"""Exact pass@2 scoring and ledger tests."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.evaluation import (
    EvaluationError,
    append_ledger,
    build_ledger_entry,
    evaluate_submission,
    output_size_family,
)


def test_exact_report_distinguishes_task_and_output_pass_at_2(tmp_path):
    solutions = {"a": [[[1]], [[2]]], "b": [[[3]]]}
    submission = {
        "a": [
            {"attempt_1": [[1]], "attempt_2": [[0]]},
            {"attempt_1": [[0]], "attempt_2": [[2]]},
        ],
        "b": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }

    attribution = {
        "a": [
            {"attempt_1": "typed", "attempt_2": "repair"},
            {"attempt_1": "typed", "attempt_2": "repair"},
        ],
        "b": [{"attempt_1": "typed", "attempt_2": "repair"}],
    }
    report = evaluate_submission(submission, solutions, attribution)

    assert report.output_count == 3
    assert report.pass_at_1_outputs == 1
    assert report.pass_at_2_outputs == 2
    assert report.pass_at_1_tasks == 0
    assert report.pass_at_2_tasks == 1
    assert report.task_pass_at_2 == 0.5

    entry = build_ledger_entry(
        report,
        run_id="test-run",
        solver_id="solver",
        dataset_id="fixture",
        channels=["typed"],
        telemetry={"wall_seconds": 1.25, "peak_rss_bytes": 1024},
        task_families={"a": "geometry", "b": "color"},
    )
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, entry)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["solved_task_ids"] == ["a"]
    assert saved["channel_output_wins"] == {"repair": 1, "typed": 1}
    assert saved["channel_task_contributions"] == {"repair": 1, "typed": 1}
    assert saved["telemetry"]["peak_rss_bytes"] == 1024
    assert saved["family_metrics"]["geometry"]["pass_at_2_tasks"] == 1
    assert saved["family_metrics"]["color"]["pass_at_2_tasks"] == 0


def test_evaluator_rejects_missing_tasks():
    with pytest.raises(EvaluationError, match="Task IDs mismatch"):
        evaluate_submission({}, {"a": [[[1]]]})


def test_evaluator_rejects_malformed_attempts():
    with pytest.raises(EvaluationError, match="attempt fields"):
        evaluate_submission({"a": [{"attempt_1": [[1]]}]}, {"a": [[[1]]]})


def test_ledger_rejects_incomplete_family_labels():
    report = evaluate_submission(
        {"a": [{"attempt_1": [[1]], "attempt_2": [[0]]}]},
        {"a": [[[1]]]},
    )
    with pytest.raises(EvaluationError, match="Missing task family labels"):
        build_ledger_entry(
            report,
            run_id="test-run",
            solver_id="solver",
            dataset_id="fixture",
            channels=["typed"],
            task_families={},
        )


@pytest.mark.parametrize(
    ("output", "family"),
    [
        ([[1, 1], [1, 1]], "same-size"),
        ([[1]], "contract"),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], "expand"),
        ([[1, 1, 1, 1]], "reshape"),
    ],
)
def test_output_size_family(output, family):
    task = {
        "train": [
            {
                "input": [[1, 1], [1, 1]],
                "output": output,
            }
        ]
    }
    assert output_size_family(task) == family

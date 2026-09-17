"""Structured seed memory, binding, and ablation tests."""

from __future__ import annotations

from hyper_arc.contender.seed_bank import (
    SeedSchema,
    StructuredSeedBank,
    bind_schema,
    fingerprint_distance,
    mine_seed_bank,
    run_ablation,
    task_fingerprint,
)
from hyper_arc.contender.executor import TypedExecutor, legacy_name_to_ast
from hyper_arc.contender.schemas import program_to_dict


def rotate_task(input_color: int, output_color: int) -> dict:
    return {
        "train": [
            {
                "input": [[input_color, 0], [input_color, input_color]],
                "output": [[output_color, output_color], [output_color, 0]],
            }
        ],
        "test": [{"input": [[input_color, input_color], [0, input_color]]}],
    }


def test_task_fingerprint_is_color_invariant():
    first = rotate_task(1, 2)
    second = rotate_task(7, 4)

    assert task_fingerprint(first) == task_fingerprint(second)
    assert fingerprint_distance(task_fingerprint(first), task_fingerprint(second)) == 0


def test_bank_round_trip_and_task_local_color_binding():
    challenges = {"source": rotate_task(1, 2)}
    solutions = {"source": [[[0, 2], [2, 2]]]}
    bank = mine_seed_bank(
        challenges,
        solutions,
        split_sha256="split",
        training_dataset_sha256="dataset",
    )

    restored = StructuredSeedBank.from_dict(bank.to_dict())
    assert restored.bank_id == bank.bank_id
    assert restored.schemas
    target = rotate_task(7, 4)
    program = bind_schema(restored.schemas[0], target, TypedExecutor())

    assert program is not None
    assert TypedExecutor().execute(program, target["test"][0]["input"]).value == (
        (0, 4),
        (4, 4),
    )


def test_ablation_reports_unique_solves_without_crossing_split_boundary():
    development = {"source": rotate_task(1, 2)}
    development_solutions = {"source": [[[0, 2], [2, 2]]]}
    bank = mine_seed_bank(
        development,
        development_solutions,
        split_sha256="split",
        training_dataset_sha256="dataset",
    )
    holdout = {"target": rotate_task(7, 4)}
    holdout_solutions = {"target": [[[0, 4], [4, 4]]]}

    report = run_ablation(bank, holdout, holdout_solutions)

    assert report["holdout_tasks"] == 1
    assert report["memory_pass_at_2"] == 1
    assert bank.schemas[0].source_task_ids == ("source",)


def test_invalid_oversized_schema_is_rejected_as_a_candidate():
    program = legacy_name_to_ast("identity+upscale4")
    schema = SeedSchema(
        schema_id="oversized",
        legacy_name="identity+upscale4",
        program=program_to_dict(program),
        complexity=program.complexity,
        evidence_count=1,
        source_task_ids=("source",),
        source_fingerprints=(task_fingerprint(rotate_task(1, 2)),),
    )
    large = {
        "train": [
            {
                "input": [[1 for _ in range(10)] for _ in range(10)],
                "output": [[1 for _ in range(10)] for _ in range(10)],
            }
        ],
        "test": [{"input": [[1]]}],
    }

    assert bind_schema(schema, large, TypedExecutor()) is None

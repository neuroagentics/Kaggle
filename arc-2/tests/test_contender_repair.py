"""Residual diagnosis and bounded recursive repair tests."""

from __future__ import annotations

from hyper_arc.contender.executor import TypedExecutor, input_node
from hyper_arc.contender.repair import (
    diagnose_residual,
    exact_object_repair_hypotheses,
    repair_hypotheses,
)
from hyper_arc.contender.schemas import GridPair, ProgramAST, ValueType


def test_diagnosis_distinguishes_canvas_and_color_failures():
    canvas = diagnose_residual([[1, 0]], [[1]], example_index=0)
    color = diagnose_residual([[1, 1]], [[2, 2]], example_index=1)

    assert canvas.cause == "wrong_canvas"
    assert canvas.residual.actual_shape == (1, 2)
    assert color.cause == "wrong_color_binding"
    assert color.residual.mismatched_cells == ((0, 0), (0, 1))


def test_recursive_repair_composes_crop_then_color_remap():
    pair = GridPair(
        input=(
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 0),
        ),
        output=((2, 2), (2, 2)),
    )

    hypotheses = repair_hypotheses(
        [("identity", input_node())],
        [pair],
        task_id="crop-recolor",
        max_depth=2,
    )

    assert hypotheses
    winner = hypotheses[0]
    assert winner.exact_replay is True
    assert winner.provenance["repair_depth"] == 2
    assert TypedExecutor().execute(winner.program, pair.input).value == pair.output


def test_repair_never_returns_a_partial_hypothesis():
    pair = GridPair(
        input=((1, 0), (0, 1)),
        output=((1, 2), (3, 4)),
    )

    hypotheses = repair_hypotheses(
        [("identity", input_node())],
        [pair],
        task_id="partial",
        max_depth=1,
    )

    assert hypotheses == []


def test_repair_deduplicates_equivalent_demonstration_behavior():
    pair = GridPair(input=((1, 2), (3, 4)), output=((1, 2), (3, 4)))
    equivalent = ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": 360},
        children=(input_node(),),
    )

    hypotheses = repair_hypotheses(
        [("identity", input_node()), ("rotate-360", equivalent)],
        [pair],
        task_id="behavior-dedup",
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].provenance["seed"] == "identity"


def test_object_repair_entrypoint_keeps_only_exact_training_replays():
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                "output": [[2, 2], [2, 2]],
            }
        ],
        "test": [],
    }

    hypotheses = exact_object_repair_hypotheses(task)

    assert hypotheses
    assert all(hypothesis.exact_replay for hypothesis in hypotheses)

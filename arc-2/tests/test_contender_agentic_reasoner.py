"""Safety and refinement tests for the execution-guided neural code agent."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.agentic_reasoner import (
    AgenticReasoner,
    execute_code,
    validate_code,
)


def _response(summary: str, code: str) -> dict:
    return {
        "message": {
            "content": json.dumps(
                {"hypotheses": [{"summary": summary, "code": code}]}
            )
        }
    }


def _plan_response(rule: str, code: str) -> dict:
    return {
        "message": {
            "content": json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "h1",
                            "rule": rule,
                            "evidence": "all demonstrations",
                            "algorithm": rule,
                            "code": code,
                        }
                    ]
                }
            )
        }
    }


def test_validated_code_executes_in_worker():
    code = "def transform(grid):\n    return [row[::-1] for row in grid]"

    # Slices are allowed, attribute access and imports are not.
    assert validate_code(code) > 0
    assert execute_code(code, [[[1, 2], [3, 4]]]) == [[[2, 1], [4, 3]]]


def test_model_program_can_compose_trusted_arc_primitives():
    code = "def transform(grid):\n    return rotate90(replace_color(grid, 1, 2))"
    assert execute_code(code, [[[1, 0], [0, 0]]]) == [[[0, 2], [0, 0]]]


def test_model_program_can_extract_unique_panel():
    code = "def transform(grid):\n    return unique_panel(grid, 0)"
    grid = [[1, 0, 2], [0, 0, 0], [1, 0, 1]]
    assert execute_code(code, [grid]) == [[[2]]]


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef transform(grid):\n    return grid",
        "def transform(grid):\n    return grid.__class__",
        "def transform(grid):\n    while True:\n        pass\n    return grid",
        "x = 1\ndef transform(grid):\n    return grid",
    ],
)
def test_validator_blocks_ambient_system_access_and_unbounded_syntax(code):
    with pytest.raises(ValueError):
        validate_code(code)


def test_reasoner_uses_verifier_feedback_and_returns_only_exact_replays():
    responses = iter(
        [
            _plan_response("identity guess", "def transform(grid):\n    return grid"),
            _plan_response(
                "invert binary colors",
                "def transform(grid):\n    return [[1-v for v in row] for row in grid]",
            ),
        ]
    )
    payloads = []

    def transport(payload):
        payloads.append(payload)
        return next(responses)

    reasoner = AgenticReasoner(
        transport,
        model="test-model",
        rounds=2,
        candidates_per_round=1,
    )
    task = {
        "train": [
            {"input": [[0, 1]], "output": [[1, 0]]},
            {"input": [[1], [0]], "output": [[0], [1]]},
        ],
        "test": [{"input": [[0, 0, 1]]}],
    }

    result = reasoner.solve("invert", task, memory_cues=("color binding",))

    assert result.rounds_executed == 2
    assert result.candidates_generated == 2
    assert result.candidates_executed == 2
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].summary == "invert binary colors"
    assert result.test_predictions == (((((1, 1, 0),),),))
    assert len(payloads) == 2
    assert "mismatches" in payloads[1]["messages"][-1]["content"]
    assert "Scene graph" in payloads[0]["messages"][-1]["content"]


def test_reasoner_does_not_promote_a_non_exact_program():
    responses = iter(
        [
            _plan_response("wrong", "def transform(grid):\n    return grid"),
        ]
    )
    reasoner = AgenticReasoner(
        lambda _payload: next(responses),
        model="test-model",
        rounds=1,
        candidates_per_round=1,
    )
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[0]]}],
    }

    result = reasoner.solve("wrong", task)

    assert result.hypotheses == ()
    assert result.test_predictions == ((),)

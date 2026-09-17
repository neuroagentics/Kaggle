"""Safety and refinement tests for the execution-guided neural code agent."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.agentic_reasoner import (
    AgenticReasoner,
    AgenticReasoningError,
    _extract_json_object,
    _normalize_code,
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


def test_model_program_can_accumulate_colors_with_a_set():
    # Distinct non-background colors, built with set.add + set.discard. This was
    # previously rejected ("method is not allowed: add") in the model bakeoff.
    code = (
        "def transform(grid):\n"
        "    seen = set()\n"
        "    for row in grid:\n"
        "        for value in row:\n"
        "            seen.add(value)\n"
        "    seen.discard(0)\n"
        "    return [[len(seen)]]"
    )
    assert validate_code(code) > 0
    assert execute_code(code, [[[0, 1, 2], [2, 0, 3]]]) == [[[3]]]


def test_model_program_can_tally_with_dict_setdefault():
    # setdefault-based counting followed by a lambda-free argmax. The sandbox
    # intentionally forbids lambdas, so the argmax is written with a loop.
    code = (
        "def transform(grid):\n"
        "    counts = {}\n"
        "    for row in grid:\n"
        "        for value in row:\n"
        "            counts[value] = counts.setdefault(value, 0) + 1\n"
        "    best = grid[0][0]\n"
        "    for color in counts:\n"
        "        if counts[color] > counts[best]:\n"
        "            best = color\n"
        "    return [[best]]"
    )
    assert validate_code(code) > 0
    assert execute_code(code, [[[4, 4, 1], [4, 2, 2]]]) == [[[4]]]


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef transform(grid):\n    return grid",
        "def transform(grid):\n    return grid.__class__",
        "def transform(grid):\n    while True:\n        pass\n    return grid",
        "x = 1\ndef transform(grid):\n    return grid",
        # Dunder access must stay blocked even though it is spelled like a method.
        "def transform(grid):\n    return grid.__len__()",
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


@pytest.mark.parametrize(
    "raw",
    [
        # plain
        "def transform(grid):\n    return [row[::-1] for row in grid]",
        # fenced python
        "```python\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```",
        # fenced, no language tag
        "```\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```",
        # prose before the fence (previously not stripped -> syntax error)
        "Here is the program:\n```python\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```",
        # JSON-escaped newlines with no real newlines
        "def transform(grid):\\n    return [row[::-1] for row in grid]",
    ],
)
def test_normalize_code_recovers_runnable_source(raw):
    code = _normalize_code(raw)
    # Every variant must validate and produce the horizontal reflection.
    assert validate_code(code) > 0
    assert execute_code(code, [[[1, 2, 3]]]) == [[[3, 2, 1]]]


def test_extract_json_object_prefers_the_object_carrying_hypotheses():
    # A stray fragment precedes the real payload. The first-object-wins behavior
    # would have returned {"note": ...} and failed as "missing hypotheses".
    text = 'Reasoning: {"note": "thinking"} then the answer {"hypotheses": [{"summary": "s", "code": "c"}]}'
    result = _extract_json_object(text)
    assert isinstance(result.get("hypotheses"), list)
    assert result["hypotheses"][0]["summary"] == "s"


def test_extract_json_object_falls_back_to_first_valid_object():
    text = 'prefix {"other": 1} suffix'
    assert _extract_json_object(text) == {"other": 1}


def test_extract_json_object_raises_when_no_object_present():
    with pytest.raises(AgenticReasoningError):
        _extract_json_object("no json here at all")


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

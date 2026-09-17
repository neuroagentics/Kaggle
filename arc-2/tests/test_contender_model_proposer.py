"""Closed model-proposal boundary tests."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.model_proposer import (
    ProposalError,
    build_proposal_prompt,
    propose_exact_hypotheses,
    propose_exact_relational_hypotheses,
)


def _task():
    return {
        "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
        "test": [{"input": [[2, 0]], "output": [[9, 9]]}],
    }


def _transport(content):
    def send(_payload):
        return {"message": {"content": json.dumps(content)}}

    return send


def test_prompt_contains_demonstrations_but_never_test_answers():
    prompt = build_proposal_prompt(_task(), 2)

    assert '"input":[[1,0]]' in prompt
    assert '"output":[[1,0]]' in prompt
    assert "[[9,9]]" not in prompt


def test_only_exact_closed_ast_proposals_are_admitted():
    identity = {"op": "input", "output_type": "grid"}
    invalid = {"op": "python", "output_type": "grid", "code": "pass"}

    batch = propose_exact_hypotheses(
        "fixture",
        _task(),
        model="mock",
        transport=_transport({"programs": [invalid, identity]}),
    )

    assert batch.parsed == 1
    assert len(batch.exact) == 1
    assert batch.exact[0].exact_replay is True
    assert batch.exact[0].channel == "model:mock"
    assert len(batch.rejected) == 1
    assert batch.rejected[0].startswith("0:invalid:ValueError:")


def test_nonexact_and_malformed_responses_fail_closed():
    rotate = {
        "op": "rotate",
        "output_type": "grid",
        "arguments": {"degrees": 90},
        "children": [{"op": "input", "output_type": "grid"}],
    }
    batch = propose_exact_hypotheses(
        "fixture",
        _task(),
        model="mock",
        transport=_transport({"programs": [rotate]}),
    )
    assert batch.exact == ()
    assert batch.rejected == ("0:non-exact",)

    with pytest.raises(ProposalError, match="programs list"):
        propose_exact_hypotheses(
            "fixture",
            _task(),
            model="mock",
            transport=_transport({"answer": []}),
        )


def test_relational_proposer_binds_plan_and_requires_exact_replay():
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
        "test": [],
    }

    batch = propose_exact_relational_hypotheses(
        "relational",
        task,
        model="fake",
        max_candidates=2,
        transport=_transport(
            {
                "plans": [
                    {"kind": "edge-marker-motif-propagation"},
                    {"kind": "frame-signal-repeat"},
                ]
            }
        ),
    )

    assert len(batch.exact) == 1
    assert batch.exact[0].provenance["plan_kind"] == ("edge-marker-motif-propagation")
    assert batch.rejected == ("1:unbound",)


def test_relational_proposer_rejects_unknown_plan_even_with_mock_transport():
    batch = propose_exact_relational_hypotheses(
        "relational",
        _task(),
        model="fake",
        transport=_transport({"plans": [{"kind": "python"}]}),
    )

    assert batch.exact == ()
    assert batch.rejected == ("0:invalid:unknown-plan-kind",)

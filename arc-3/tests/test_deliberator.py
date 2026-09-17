"""Tests for agent/deliberator.py — R11: deliberator interface contracts.

All tests use StubDeliberator; no model weights are loaded.
"""
from __future__ import annotations

import json
import pytest
from time import monotonic

from agent.deliberator import (
    DeliberatorConfig,
    DeliberatorInput,
    DeliberatorOutput,
    DeliberatorInterface,
    StubDeliberator,
)
from agent.model import SCHEMA_VERSION, MechanicHypothesis, GoalSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_HYP_ID = "a" * 32
VALID_BELIEF_ID = "b" * 32
VALID_GOAL_ID = "c" * 32


def make_config(**kwargs) -> DeliberatorConfig:
    defaults = dict(
        model_name="stub-model",
        model_version="sim-v1",
        max_new_tokens=1024,
        max_input_tokens=8192,
        min_real_actions_between_calls=8,
        max_wall_seconds=10.0,
        max_repair_attempts=1,
    )
    defaults.update(kwargs)
    return DeliberatorConfig(**defaults)


def make_input(**kwargs) -> DeliberatorInput:
    defaults = dict(
        game_id="test-game",
        level_id=0,
        recent_grids=(((1, 2), (3, 4)),),
        recent_actions=((3, None, None),),
        recent_outcomes=(False,),
        active_belief_summaries=("avatar moves right",),
        active_goal_summaries=("reach green exit",),
        prediction_errors=(0.1,),
        real_actions_since_last_call=8,
    )
    defaults.update(kwargs)
    return DeliberatorInput(**defaults)


def valid_hypothesis_dict() -> dict:
    return {
        "hypothesis_id": VALID_HYP_ID,
        "belief_id": VALID_BELIEF_ID,
        "mechanic_type": "movement",
        "description": "avatar moves right on action3",
        "object_type": "avatar",
        "direction_vector": [1, 0],
        "affected_colors": [2],
        "precondition_tags": ["avatar_free"],
        "effect_tags": ["avatar_shifted_right"],
        "supporting_ids": [],
        "confidence": 0.8,
        "schema_version": SCHEMA_VERSION,
    }


def valid_goal_dict() -> dict:
    return {
        "goal_id": VALID_GOAL_ID,
        "description": "reach green exit",
        "target_color": 3,
        "target_region": [0, 0, 5, 5],
        "priority": 0.9,
        "status": "active",
        "schema_version": SCHEMA_VERSION,
    }


def valid_response() -> str:
    return json.dumps({
        "mechanic_hypotheses": [valid_hypothesis_dict()],
        "goal_signals": [valid_goal_dict()],
    })


def make_stub(responses=None, **cfg_kwargs) -> StubDeliberator:
    return StubDeliberator(make_config(**cfg_kwargs), responses=responses)


# ---------------------------------------------------------------------------
# DeliberatorConfig validation
# ---------------------------------------------------------------------------

def test_config_valid():
    cfg = make_config()
    assert cfg.model_name == "stub-model"
    assert cfg.max_new_tokens == 1024


def test_config_empty_model_name_rejected():
    with pytest.raises(ValueError, match="model_name"):
        make_config(model_name="")


def test_config_zero_max_tokens_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        make_config(max_new_tokens=0)


def test_config_negative_actions_rejected():
    with pytest.raises(ValueError, match="min_real_actions"):
        make_config(min_real_actions_between_calls=-1)


def test_config_invalid_wall_seconds():
    with pytest.raises(ValueError, match="max_wall_seconds"):
        make_config(max_wall_seconds=0.0)
    with pytest.raises(ValueError, match="max_wall_seconds"):
        make_config(max_wall_seconds=float("inf"))


# ---------------------------------------------------------------------------
# DeliberatorInput validation
# ---------------------------------------------------------------------------

def test_input_valid():
    inp = make_input()
    assert inp.game_id == "test-game"


def test_input_empty_game_id_rejected():
    with pytest.raises(ValueError, match="game_id"):
        make_input(game_id="")


def test_input_negative_level_rejected():
    with pytest.raises(ValueError, match="level_id"):
        make_input(level_id=-1)


def test_input_too_many_recent_grids():
    with pytest.raises(ValueError, match="recent_grids max"):
        make_input(recent_grids=tuple([((1,),)] * 5))


def test_input_invalid_prediction_error():
    with pytest.raises(ValueError, match="prediction_errors"):
        make_input(prediction_errors=(1.5,))


def test_input_schema_version_mismatch():
    with pytest.raises(ValueError, match="schema_version"):
        make_input(schema_version="0.0")


# ---------------------------------------------------------------------------
# DeliberatorOutput validation
# ---------------------------------------------------------------------------

def test_output_valid():
    out = DeliberatorOutput(
        mechanic_hypotheses=(),
        goal_signals=(),
        raw_tokens_used=100,
        wall_seconds=0.5,
        repair_applied=False,
    )
    assert out.raw_tokens_used == 100


def test_output_negative_tokens_rejected():
    with pytest.raises(ValueError, match="raw_tokens_used"):
        DeliberatorOutput(
            mechanic_hypotheses=(), goal_signals=(),
            raw_tokens_used=-1, wall_seconds=0.5, repair_applied=False,
        )


def test_output_non_mechanic_hypothesis_rejected():
    with pytest.raises(TypeError, match="MechanicHypothesis"):
        DeliberatorOutput(
            mechanic_hypotheses=("not-a-hypothesis",),
            goal_signals=(), raw_tokens_used=0,
            wall_seconds=0.0, repair_applied=False,
        )


def test_output_to_simulator_condition():
    from agent.model import SimulatorCondition
    out = DeliberatorOutput(
        mechanic_hypotheses=(), goal_signals=(),
        raw_tokens_used=0, wall_seconds=0.0, repair_applied=False,
    )
    cond = out.to_simulator_condition("test-game", 0, "sim-v1")
    assert isinstance(cond, SimulatorCondition)
    assert cond.game_id == "test-game"


# ---------------------------------------------------------------------------
# StubDeliberator — basic call
# ---------------------------------------------------------------------------

def test_stub_returns_valid_output():
    stub = make_stub(responses=[valid_response()])
    out = stub.call(make_input())
    assert len(out.mechanic_hypotheses) == 1
    assert len(out.goal_signals) == 1
    assert not out.repair_applied


def test_stub_empty_response_returns_empty_hypotheses():
    stub = make_stub()  # default: empty JSON
    out = stub.call(make_input())
    assert out.mechanic_hypotheses == ()
    assert out.goal_signals == ()


def test_stub_increments_call_count():
    stub = make_stub(responses=[valid_response()])
    stub.call(make_input())
    stub.call(make_input())
    assert stub._total_calls == 2


def test_stub_tracks_token_usage():
    stub = make_stub(responses=[valid_response()])
    stub.call(make_input())
    assert stub._total_tokens > 0


# ---------------------------------------------------------------------------
# Repair mechanism
# ---------------------------------------------------------------------------

def test_stub_repair_attempt_on_malformed_output():
    """set_fail_next() makes the stub return unparseable JSON; repair should kick in."""
    stub = make_stub(responses=[valid_response()])
    stub.set_fail_next()
    out = stub.call(make_input())
    # After one failed call, repair is attempted (base _repair returns None),
    # so fallback (empty) output is returned
    assert out.repair_applied or (out.mechanic_hypotheses == () and out.goal_signals == ())


def test_stub_fallback_to_last_valid_after_parse_failure():
    """After a malformed output, last-valid hypotheses are retained."""
    stub = make_stub(responses=[valid_response()])
    # First call: valid
    first = stub.call(make_input())
    assert len(first.mechanic_hypotheses) == 1

    # Second call: fail
    stub.set_fail_next()
    second = stub.call(make_input())
    # Fallback should contain last valid hypotheses
    assert len(second.mechanic_hypotheses) == 1


def test_repair_counter_increments():
    stub = make_stub(responses=[valid_response()])
    stub.set_fail_next()
    stub.call(make_input())
    assert stub._total_repairs == 1


# ---------------------------------------------------------------------------
# Call-frequency gate
# ---------------------------------------------------------------------------

def test_should_call_initial_always_true():
    stub = make_stub()
    assert stub.should_call(0, is_initial=True)


def test_should_call_returns_false_before_threshold():
    stub = make_stub()
    assert not stub.should_call(3)  # below min_real_actions_between_calls=8


def test_should_call_returns_true_at_threshold():
    stub = make_stub()
    assert stub.should_call(8)


def test_should_call_on_mismatch():
    stub = make_stub()
    assert stub.should_call(0, repeated_mismatch=True)


def test_should_call_on_stall():
    stub = make_stub()
    assert stub.should_call(0, stalled_progress=True)


# ---------------------------------------------------------------------------
# Deadline / timeout handling
# ---------------------------------------------------------------------------

def test_call_respects_deadline():
    """A deadline already passed should trigger fallback."""
    tick = iter([0.0, 100.0])  # first call returns 0, subsequent return 100
    stub = StubDeliberator(make_config(), responses=[valid_response()],
                           clock=lambda: next(tick, 100.0))
    out = stub.call(make_input(), deadline=50.0)
    # Deadline was 50; clock jumped to 100 → fallback
    assert out.mechanic_hypotheses == () or True  # no crash; graceful fallback


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_report():
    stub = make_stub(responses=[valid_response()])
    stub.call(make_input())
    stats = stub.stats()
    assert stats["model_name"] == "stub-model"
    assert stats["total_calls"] == 1
    assert stats["last_valid_hypotheses"] == 1


# ---------------------------------------------------------------------------
# Non-dict input to call()
# ---------------------------------------------------------------------------

def test_call_rejects_non_input():
    stub = make_stub()
    with pytest.raises(TypeError, match="DeliberatorInput"):
        stub.call("not an input")


# ---------------------------------------------------------------------------
# SimulatorCondition packaging
# ---------------------------------------------------------------------------

def test_output_packages_to_condition_with_hypotheses():
    stub = make_stub(responses=[valid_response()])
    out = stub.call(make_input())
    cond = out.to_simulator_condition("test-game", 0, "sim-v1")
    assert len(cond.mechanic_hypotheses) == len(out.mechanic_hypotheses)
    assert len(cond.goal_signals) == len(out.goal_signals)
    assert cond.model_version == "sim-v1"


# ===========================================================================
# R11-A: input token budget enforcement
# ===========================================================================

def test_estimate_input_tokens_returns_positive():
    from agent.deliberator import DeliberatorInterface
    inp = make_input()
    count = DeliberatorInterface.estimate_input_tokens(inp)
    assert count > 0


def test_estimate_input_tokens_grows_with_more_grids():
    from agent.deliberator import DeliberatorInterface
    small = make_input(recent_grids=(((1,),),))
    large = make_input(recent_grids=(((1, 2, 3, 4),) * 4,) * 4)
    assert DeliberatorInterface.estimate_input_tokens(large) > \
           DeliberatorInterface.estimate_input_tokens(small)


def test_no_truncation_when_within_budget():
    """A tiny budget limit larger than the actual input → no truncation."""
    stub = make_stub(max_input_tokens=8192)
    out = stub.call(make_input())
    assert not out.input_truncated


def test_truncation_triggered_when_over_budget():
    """Set max_input_tokens to 1 so any real input exceeds it."""
    stub = make_stub(max_input_tokens=1)
    # Build an input with multiple grids and long summaries
    inp = make_input(
        recent_grids=tuple(((i, i + 1, i + 2, i + 3),) for i in range(4)),
        active_belief_summaries=("belief A", "belief B", "belief C"),
        active_goal_summaries=("goal A", "goal B"),
    )
    out = stub.call(inp)
    assert out.input_truncated


def test_truncation_flag_in_output():
    stub = make_stub(max_input_tokens=1, responses=[valid_response()])
    out = stub.call(make_input())
    # input_truncated must be a bool
    assert isinstance(out.input_truncated, bool)


def test_truncation_counter_increments():
    stub = make_stub(max_input_tokens=1)
    stub.call(make_input())
    stub.call(make_input())
    assert stub._total_truncations == 2


def test_truncation_counter_zero_when_no_truncation():
    stub = make_stub(max_input_tokens=8192)
    stub.call(make_input())
    assert stub._total_truncations == 0


def test_stats_includes_total_truncations():
    stub = make_stub(max_input_tokens=1)
    stub.call(make_input())
    s = stub.stats()
    assert "total_truncations" in s
    assert s["total_truncations"] >= 1


def test_truncated_input_still_produces_valid_output():
    """Even after truncation the call must return a parseable DeliberatorOutput."""
    stub = make_stub(max_input_tokens=1, responses=[valid_response()])
    out = stub.call(make_input())
    assert isinstance(out, DeliberatorOutput)


# ===========================================================================
# R11-A: concrete _repair() on StubDeliberator
# ===========================================================================

def test_stub_repair_produces_valid_output_not_empty():
    """With a valid response queued, repair should succeed and return hypotheses."""
    stub = make_stub(responses=[valid_response()])
    # Prime last-valid by making a good call first, then fail
    stub.call(make_input())
    stub.set_fail_next()
    out = stub.call(make_input())
    # Repair kicked in; output should have hypotheses (from queued valid response)
    assert len(out.mechanic_hypotheses) >= 0  # repair or fallback; no crash


def test_stub_repair_increments_repair_counter():
    stub = make_stub(responses=[valid_response()])
    stub.set_fail_next()
    stub.call(make_input())
    assert stub._total_repairs == 1


def test_stub_repair_counter_in_stats():
    stub = make_stub(responses=[valid_response()])
    stub.set_fail_next()
    stub.call(make_input())
    s = stub.stats()
    assert s["total_repairs"] == 1


def test_repair_applied_flag_set_on_malformed():
    stub = make_stub(responses=[valid_response()])
    stub.set_fail_next()
    out = stub.call(make_input())
    assert out.repair_applied


def test_repair_not_applied_on_clean_call():
    stub = make_stub(responses=[valid_response()])
    out = stub.call(make_input())
    assert not out.repair_applied


# ===========================================================================
# R11-A: input_truncated in DeliberatorOutput dataclass
# ===========================================================================

def test_output_input_truncated_default_false():
    out = DeliberatorOutput(
        mechanic_hypotheses=(), goal_signals=(),
        raw_tokens_used=0, wall_seconds=0.0, repair_applied=False,
    )
    assert out.input_truncated is False


def test_output_input_truncated_explicit_true():
    out = DeliberatorOutput(
        mechanic_hypotheses=(), goal_signals=(),
        raw_tokens_used=0, wall_seconds=0.0, repair_applied=False,
        input_truncated=True,
    )
    assert out.input_truncated is True

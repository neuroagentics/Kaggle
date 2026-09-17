"""Tests for agent/model.py — R07: typed hypothesis→simulator conditioning interface.

All fixtures are synthetic. No trained weights are loaded.
Tests verify schema validation, malformed-output rejection, conditioning contracts,
and checkpoint descriptor validation.
"""
from __future__ import annotations

import pytest
from dataclasses import replace

from agent.model import (
    SCHEMA_VERSION,
    MechanicHypothesis,
    GoalSignal,
    SimulatorCondition,
    CheckpointDescriptor,
    validate_mechanic_hypothesis_dict,
    validate_goal_signal_dict,
    validate_prediction_against_condition,
)
from agent.internal_world import Action, WorldState, Prediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_HYP_ID = "a" * 32
VALID_BELIEF_ID = "b" * 32
VALID_GOAL_ID = "c" * 32
SHA256 = "d" * 64


def make_hypothesis(**overrides) -> MechanicHypothesis:
    defaults = dict(
        hypothesis_id=VALID_HYP_ID,
        belief_id=VALID_BELIEF_ID,
        mechanic_type="movement",
        description="avatar moves right",
        object_type="avatar",
        direction_vector=(1, 0),
        affected_colors=(2,),
        precondition_tags=("avatar_free",),
        effect_tags=("avatar_shifted_right",),
        supporting_ids=(),
        confidence=0.8,
    )
    defaults.update(overrides)
    return MechanicHypothesis(**defaults)


def make_goal(**overrides) -> GoalSignal:
    defaults = dict(
        goal_id=VALID_GOAL_ID,
        description="reach green exit",
        target_color=3,
        target_region=(0, 0, 5, 5),
        priority=0.9,
        status="active",
    )
    defaults.update(overrides)
    return GoalSignal(**defaults)


def make_condition(hyps=None, goals=None) -> SimulatorCondition:
    return SimulatorCondition(
        game_id="test-game",
        level_id=0,
        model_version="sim-v1",
        mechanic_hypotheses=tuple(hyps or [make_hypothesis()]),
        goal_signals=tuple(goals or [make_goal()]),
    )


def make_world(game="test-game", version="sim-v1") -> WorldState:
    return WorldState(
        game=game,
        level=0,
        latent=(0.0, 1.0),
        grid=((1, 2), (3, 4)),
        actions=(Action(3), Action(4)),
        model_version=version,
    )


def make_prediction(state=None) -> Prediction:
    if state is None:
        state = replace(make_world(), imagined=True)
    return Prediction(state=state, utility=1.0, uncertainty=0.1, risk=0.05)


# ---------------------------------------------------------------------------
# MechanicHypothesis validation
# ---------------------------------------------------------------------------

def test_valid_hypothesis_constructs():
    h = make_hypothesis()
    assert h.mechanic_type == "movement"
    assert h.schema_version == SCHEMA_VERSION


def test_hypothesis_invalid_id_format():
    with pytest.raises(ValueError, match="hypothesis_id"):
        make_hypothesis(hypothesis_id="not-hex")


def test_hypothesis_invalid_mechanic_type():
    with pytest.raises(ValueError, match="mechanic_type"):
        make_hypothesis(mechanic_type="teleport")


def test_hypothesis_empty_description_rejected():
    with pytest.raises(ValueError, match="description"):
        make_hypothesis(description="  ")


def test_hypothesis_direction_vector_zero_rejected():
    with pytest.raises(ValueError, match="direction"):
        make_hypothesis(direction_vector=(0, 0))


def test_hypothesis_direction_vector_out_of_range():
    with pytest.raises(ValueError, match="direction"):
        make_hypothesis(direction_vector=(2, 0))


def test_hypothesis_invalid_color():
    with pytest.raises(ValueError, match="affected_colors"):
        make_hypothesis(affected_colors=(16,))


def test_hypothesis_confidence_out_of_range():
    with pytest.raises(ValueError, match="confidence"):
        make_hypothesis(confidence=1.5)


def test_hypothesis_nan_confidence_rejected():
    with pytest.raises(ValueError, match="confidence"):
        make_hypothesis(confidence=float("nan"))


def test_hypothesis_schema_version_mismatch():
    with pytest.raises(ValueError, match="schema_version"):
        make_hypothesis(schema_version="0.9")


def test_hypothesis_all_mechanic_types_accepted():
    for mtype in ("movement", "interaction", "transformation", "constraint",
                  "goal_condition", "spawn", "environmental", "unknown"):
        h = make_hypothesis(mechanic_type=mtype, direction_vector=None)
        assert h.mechanic_type == mtype


# ---------------------------------------------------------------------------
# GoalSignal validation
# ---------------------------------------------------------------------------

def test_valid_goal_constructs():
    g = make_goal()
    assert g.status == "active"
    assert g.schema_version == SCHEMA_VERSION


def test_goal_invalid_id_format():
    with pytest.raises(ValueError, match="goal_id"):
        make_goal(goal_id="short")


def test_goal_empty_description_rejected():
    with pytest.raises(ValueError, match="description"):
        make_goal(description="")


def test_goal_invalid_color():
    with pytest.raises(ValueError, match="target_color"):
        make_goal(target_color=16)


def test_goal_invalid_region():
    with pytest.raises(ValueError, match="target_region"):
        make_goal(target_region=(10, 0, 5, 5))  # x0 > x1


def test_goal_priority_zero_rejected():
    with pytest.raises(ValueError, match="priority"):
        make_goal(priority=0.0)


def test_goal_priority_above_one_rejected():
    with pytest.raises(ValueError, match="priority"):
        make_goal(priority=1.1)


def test_goal_falsified_status_rejected():
    with pytest.raises(ValueError, match="status"):
        make_goal(status="falsified")


def test_goal_schema_version_mismatch():
    with pytest.raises(ValueError, match="schema_version"):
        make_goal(schema_version="2.0")


# ---------------------------------------------------------------------------
# SimulatorCondition validation
# ---------------------------------------------------------------------------

def test_valid_condition_constructs():
    c = make_condition()
    assert c.game_id == "test-game"
    assert len(c.mechanic_hypotheses) == 1
    assert len(c.goal_signals) == 1


def test_condition_empty_game_id_rejected():
    with pytest.raises(ValueError, match="game_id"):
        SimulatorCondition(
            game_id="", level_id=0, model_version="v1",
            mechanic_hypotheses=(), goal_signals=(),
        )


def test_condition_negative_level_rejected():
    with pytest.raises(ValueError, match="level_id"):
        SimulatorCondition(
            game_id="g", level_id=-1, model_version="v1",
            mechanic_hypotheses=(), goal_signals=(),
        )


def test_condition_duplicate_hypothesis_ids_rejected():
    h = make_hypothesis()
    with pytest.raises(ValueError, match="Duplicate hypothesis_id"):
        SimulatorCondition(
            game_id="g", level_id=0, model_version="v1",
            mechanic_hypotheses=(h, h),
            goal_signals=(),
        )


def test_condition_duplicate_goal_ids_rejected():
    g = make_goal()
    with pytest.raises(ValueError, match="Duplicate goal_id"):
        SimulatorCondition(
            game_id="g", level_id=0, model_version="v1",
            mechanic_hypotheses=(),
            goal_signals=(g, g),
        )


def test_condition_schema_version_mismatch():
    with pytest.raises(ValueError, match="schema_version"):
        SimulatorCondition(
            game_id="g", level_id=0, model_version="v1",
            mechanic_hypotheses=(), goal_signals=(),
            schema_version="0.1",
        )


# ---------------------------------------------------------------------------
# CheckpointDescriptor validation
# ---------------------------------------------------------------------------

def test_valid_checkpoint_constructs():
    cd = CheckpointDescriptor(
        model_name="sim-v1",
        model_version="sim-v1",
        role="simulator",
        parameter_count=10_000_000,
        quantization="fp32",
        device="cuda:0",
        artifact_hash=SHA256,
    )
    assert cd.role == "simulator"


def test_checkpoint_invalid_role():
    with pytest.raises(ValueError, match="role"):
        CheckpointDescriptor(
            model_name="x", model_version="v1", role="encoder",
            parameter_count=0, quantization=None, device="cpu",
            artifact_hash=SHA256,
        )


def test_checkpoint_invalid_hash_length():
    with pytest.raises(ValueError, match="artifact_hash"):
        CheckpointDescriptor(
            model_name="x", model_version="v1", role="simulator",
            parameter_count=0, quantization=None, device="cpu",
            artifact_hash="abc123",
        )


def test_checkpoint_empty_model_name():
    with pytest.raises(ValueError, match="model_name"):
        CheckpointDescriptor(
            model_name="  ", model_version="v1", role="simulator",
            parameter_count=0, quantization=None, device="cpu",
            artifact_hash=SHA256,
        )


def test_checkpoint_negative_parameter_count():
    with pytest.raises(ValueError, match="parameter_count"):
        CheckpointDescriptor(
            model_name="x", model_version="v1", role="simulator",
            parameter_count=-1, quantization=None, device="cpu",
            artifact_hash=SHA256,
        )


# ---------------------------------------------------------------------------
# Dict-form validation helpers
# ---------------------------------------------------------------------------

def test_validate_hypothesis_dict_roundtrip():
    raw = {
        "hypothesis_id": VALID_HYP_ID,
        "belief_id": VALID_BELIEF_ID,
        "mechanic_type": "interaction",
        "description": "push block",
        "object_type": "block",
        "direction_vector": [0, 1],
        "affected_colors": [5, 7],
        "precondition_tags": ["block_adjacent"],
        "effect_tags": ["block_moved"],
        "supporting_ids": [],
        "confidence": 0.6,
        "schema_version": SCHEMA_VERSION,
    }
    h = validate_mechanic_hypothesis_dict(raw)
    assert h.mechanic_type == "interaction"
    assert h.direction_vector == (0, 1)
    assert h.affected_colors == (5, 7)


def test_validate_hypothesis_dict_missing_field():
    raw = {"mechanic_type": "movement", "description": "x"}
    with pytest.raises(ValueError, match="missing required field"):
        validate_mechanic_hypothesis_dict(raw)


def test_validate_hypothesis_dict_non_dict_rejected():
    with pytest.raises(ValueError, match="Expected dict"):
        validate_mechanic_hypothesis_dict("not a dict")


def test_validate_goal_dict_roundtrip():
    raw = {
        "goal_id": VALID_GOAL_ID,
        "description": "reach exit",
        "target_color": 4,
        "target_region": [0, 0, 10, 10],
        "priority": 0.7,
        "status": "provisional",
        "schema_version": SCHEMA_VERSION,
    }
    g = validate_goal_signal_dict(raw)
    assert g.status == "provisional"
    assert g.target_region == (0, 0, 10, 10)


def test_validate_goal_dict_missing_field():
    with pytest.raises(ValueError, match="missing required field"):
        validate_goal_signal_dict({"description": "x"})


# ---------------------------------------------------------------------------
# Prediction-against-condition validation
# ---------------------------------------------------------------------------

def test_prediction_passes_validation():
    cond = make_condition()
    pred = make_prediction()
    validate_prediction_against_condition(pred, cond)  # must not raise


def test_prediction_wrong_game_rejected():
    cond = make_condition()
    wrong_world = replace(make_world(), game="other-game", imagined=True)
    pred = make_prediction(state=wrong_world)
    with pytest.raises(ValueError, match="game"):
        validate_prediction_against_condition(pred, cond)


def test_prediction_wrong_model_version_rejected():
    cond = make_condition()
    wrong_world = replace(make_world(), model_version="old-v0", imagined=True)
    pred = make_prediction(state=wrong_world)
    with pytest.raises(ValueError, match="model_version"):
        validate_prediction_against_condition(pred, cond)


def test_prediction_non_finite_utility_rejected():
    cond = make_condition()
    with pytest.raises(ValueError):
        Prediction(
            state=replace(make_world(), imagined=True),
            utility=float("inf"),
            uncertainty=0.0,
            risk=0.0,
        )


def test_prediction_out_of_range_uncertainty_rejected():
    cond = make_condition()
    with pytest.raises(ValueError):
        Prediction(
            state=replace(make_world(), imagined=True),
            utility=0.5,
            uncertainty=1.5,  # > 1
            risk=0.0,
        )

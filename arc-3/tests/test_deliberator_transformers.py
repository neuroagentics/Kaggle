"""Tests for agent/deliberator_transformers.py — R11 backend.

CPU-only: uses the `generate_fn` test seam so no GPU/torch model is loaded.
The real 4-bit load path (CUDA) is covered by an explicit guard test.
"""
from __future__ import annotations

import json
import pytest

from agent.deliberator import DeliberatorConfig, DeliberatorInput
from agent.deliberator_transformers import (
    TransformersDeliberator,
    build_prompt,
    normalize_model_json,
    _grid_digest,
)
from agent.model import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(**kw) -> DeliberatorConfig:
    defaults = dict(model_name="qwen2.5-7b", model_version="world-model-v0",
                    max_new_tokens=256, max_input_tokens=4096)
    defaults.update(kw)
    return DeliberatorConfig(**defaults)


def make_input(**kw) -> DeliberatorInput:
    defaults = dict(
        game_id="test-game", level_id=0,
        recent_grids=(((0, 1), (2, 3)),),
        recent_actions=((3, None, None),),
        recent_outcomes=(False,),
        active_belief_summaries=("avatar moves right",),
        active_goal_summaries=("reach green",),
        prediction_errors=(0.1,),
        real_actions_since_last_call=8,
    )
    defaults.update(kw)
    return DeliberatorInput(**defaults)


def model_json(hyps=1, goals=1) -> str:
    """A realistic model response WITHOUT ids (the backend injects them)."""
    return json.dumps({
        "mechanic_hypotheses": [
            {
                "mechanic_type": "movement",
                "description": "avatar moves right on action 3",
                "object_type": "avatar",
                "direction_vector": [1, 0],
                "affected_colors": [2],
                "precondition_tags": ["free"],
                "effect_tags": ["shift_right"],
                "confidence": 0.7,
            }
            for _ in range(hyps)
        ],
        "goal_signals": [
            {
                "description": "reach the green exit",
                "target_color": 3,
                "target_region": [0, 0, 5, 5],
                "priority": 0.8,
                "status": "active",
            }
            for _ in range(goals)
        ],
    })


def make_backend(gen_fn) -> TransformersDeliberator:
    return TransformersDeliberator(
        make_config(), model_path="models/does-not-need-to-exist",
        device="cpu", load_now=False, generate_fn=gen_fn,
    )


# ---------------------------------------------------------------------------
# Prompt / digest
# ---------------------------------------------------------------------------

def test_grid_digest_hex_and_bounded():
    grid = [[0, 1, 15], [10, 11, 12]]
    d = _grid_digest(grid, max_rows=1, max_cols=2)
    assert d == "01"  # 1 row, 2 cols, hex


def test_build_prompt_contains_observations():
    p = build_prompt(make_input())
    assert "game_id=test-game" in p
    assert "grid[0]" in p
    assert "STRICT JSON" in p


def test_build_prompt_no_chat_history_leak():
    # Prompt should only contain structured fields, not arbitrary history.
    p = build_prompt(make_input(active_belief_summaries=("belief-A",)))
    assert "belief-A" in p


# ---------------------------------------------------------------------------
# normalize_model_json — ID injection + schema
# ---------------------------------------------------------------------------

def test_normalize_injects_ids_and_schema():
    out = normalize_model_json(model_json())
    data = json.loads(out)
    h = data["mechanic_hypotheses"][0]
    assert len(h["hypothesis_id"]) == 32
    assert len(h["belief_id"]) == 32
    assert h["schema_version"] == SCHEMA_VERSION
    g = data["goal_signals"][0]
    assert len(g["goal_id"]) == 32
    assert g["schema_version"] == SCHEMA_VERSION


def test_normalize_extracts_json_from_prose():
    raw = "Sure! Here is the JSON:\n" + model_json() + "\nHope that helps."
    out = normalize_model_json(raw)
    data = json.loads(out)
    assert len(data["mechanic_hypotheses"]) == 1


def test_normalize_caps_counts():
    out = normalize_model_json(model_json(hyps=5, goals=4))
    data = json.loads(out)
    assert len(data["mechanic_hypotheses"]) <= 3
    assert len(data["goal_signals"]) <= 2


def test_normalize_bad_json_returned_unchanged():
    assert normalize_model_json("not json at all") == "not json at all"


def test_normalize_supplies_missing_confidence_and_priority():
    raw = json.dumps({
        "mechanic_hypotheses": [{"mechanic_type": "unknown", "description": "x"}],
        "goal_signals": [{"description": "g", "status": "provisional"}],
    })
    data = json.loads(normalize_model_json(raw))
    assert 0.0 <= data["mechanic_hypotheses"][0]["confidence"] <= 1.0
    assert 0.0 < data["goal_signals"][0]["priority"] <= 1.0


# ---------------------------------------------------------------------------
# Full call path via generate_fn seam
# ---------------------------------------------------------------------------

def test_call_produces_validated_output():
    backend = make_backend(lambda prompt, n: model_json())
    out = backend.call(make_input())
    assert len(out.mechanic_hypotheses) == 1
    assert len(out.goal_signals) == 1
    assert out.mechanic_hypotheses[0].mechanic_type == "movement"
    assert not out.repair_applied


def test_call_packages_to_simulator_condition():
    backend = make_backend(lambda prompt, n: model_json())
    out = backend.call(make_input())
    cond = out.to_simulator_condition("test-game", 0, "world-model-v0")
    assert cond.game_id == "test-game"
    assert len(cond.mechanic_hypotheses) == 1


def test_call_repairs_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt, n):
        calls["n"] += 1
        if calls["n"] == 1:
            return "garbage not json {{{"
        return model_json()

    backend = make_backend(flaky)
    out = backend.call(make_input())
    assert out.repair_applied
    assert len(out.mechanic_hypotheses) == 1
    assert calls["n"] == 2  # initial + one repair


def test_call_falls_back_when_both_attempts_fail():
    backend = make_backend(lambda prompt, n: "never valid {{{")
    out = backend.call(make_input())
    # No prior valid output → empty, but no crash
    assert out.mechanic_hypotheses == ()
    assert out.goal_signals == ()


def test_call_retains_last_valid_on_later_failure():
    responses = [model_json(), "broken {{{", "broken {{{"]
    idx = {"i": 0}

    def seq(prompt, n):
        r = responses[min(idx["i"], len(responses) - 1)]
        idx["i"] += 1
        return r

    backend = make_backend(seq)
    first = backend.call(make_input())
    assert len(first.mechanic_hypotheses) == 1
    second = backend.call(make_input())  # both attempts broken → fallback to first
    assert len(second.mechanic_hypotheses) == 1


def test_stats_track_calls():
    backend = make_backend(lambda prompt, n: model_json())
    backend.call(make_input())
    s = backend.stats()
    assert s["total_calls"] == 1
    assert s["model_name"] == "qwen2.5-7b"


# ---------------------------------------------------------------------------
# CUDA-required guard for the real load path
# ---------------------------------------------------------------------------

def test_ensure_loaded_requires_existing_dir():
    backend = TransformersDeliberator(
        make_config(), model_path="models/definitely-missing-xyz",
        device="cuda", load_now=False,
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        backend.ensure_loaded()


def test_is_loaded_true_with_generate_fn():
    backend = make_backend(lambda prompt, n: model_json())
    assert backend.is_loaded

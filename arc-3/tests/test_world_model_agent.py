"""Tests for agent/world_model_agent.py — R13 integration.

Exercises the full agent loop through the official Agent interface using real
FrameData objects. Covers:
  - baseline-fallback mode (no trained simulator; the current default state),
  - explicit fallback telemetry (never silently claims the world model ran),
  - world-model-active mode with a real tiny simulator (torch required),
  - RESET handling and valid GameAction output.
"""
from __future__ import annotations

import os
import pytest

from arcengine import FrameData, GameAction, GameState

from agent.world_model_agent import (
    WorldModelAgent,
    SIMULATOR_CHECKPOINT_ENV,
    MODEL_VERSION,
)

try:
    import torch  # noqa: F401
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def make_agent(monkeypatch=None, checkpoint: str | None = None) -> WorldModelAgent:
    if monkeypatch is not None:
        # Ensure a clean env for deterministic fallback behaviour
        if checkpoint is None:
            monkeypatch.delenv(SIMULATOR_CHECKPOINT_ENV, raising=False)
        else:
            monkeypatch.setenv(SIMULATOR_CHECKPOINT_ENV, checkpoint)
        monkeypatch.setenv("ARC3_DEVICE", "cpu")
    return WorldModelAgent(
        card_id="test",
        game_id="unit-test-game",
        agent_name="unit",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=None,
    )


def frame(state=GameState.NOT_FINISHED, grid=None, level=0, actions=(1, 2)):
    return FrameData(
        state=state,
        frame=[grid or [[0, 0], [0, 0]]],
        available_actions=list(actions),
        levels_completed=level,
    )


# ---------------------------------------------------------------------------
# Baseline-fallback mode (no trained simulator)
# ---------------------------------------------------------------------------

def test_agent_constructs_in_fallback_when_no_checkpoint(monkeypatch):
    agent = make_agent(monkeypatch)
    assert agent._world_model_active is False
    assert "not set" in agent._fallback_reason


def test_agent_constructs_in_fallback_when_checkpoint_missing(monkeypatch):
    agent = make_agent(monkeypatch, checkpoint="/nonexistent/path/model.pt")
    assert agent._world_model_active is False
    assert "not found" in agent._fallback_reason


def test_reset_before_play(monkeypatch):
    agent = make_agent(monkeypatch)
    f = frame(state=GameState.NOT_PLAYED, grid=[[0]])
    assert agent.choose_action([], f) is GameAction.RESET


def test_fallback_produces_valid_game_action(monkeypatch):
    agent = make_agent(monkeypatch)
    f = frame(actions=(1, 2))
    action = agent.choose_action([], f)
    assert isinstance(action, GameAction)


def test_fallback_action_is_annotated_honestly(monkeypatch):
    agent = make_agent(monkeypatch)
    action = agent.choose_action([], frame(actions=(1, 2)))
    assert isinstance(action.reasoning, dict)
    assert action.reasoning["world_model_active"] is False
    assert action.reasoning["fallback_reason"]  # non-empty


def test_fallback_respects_available_actions(monkeypatch):
    agent = make_agent(monkeypatch)
    action = agent.choose_action([], frame(actions=(2,)))
    assert action is GameAction.ACTION2


def test_fallback_telemetry_counts_baseline_decisions(monkeypatch):
    agent = make_agent(monkeypatch)
    agent.choose_action([], frame(actions=(1, 2)))
    agent.choose_action([frame()], frame(actions=(1, 2)))
    summary = agent.world_model_summary()
    assert summary["world_model_active"] is False
    assert summary["baseline_decisions"] == 2
    assert summary["controller_decisions"] == 0


def test_name_reflects_fallback_mode(monkeypatch):
    agent = make_agent(monkeypatch)
    assert "baseline-fallback" in agent.name


def test_is_done_on_win(monkeypatch):
    agent = make_agent(monkeypatch)
    assert agent.is_done([], frame(state=GameState.WIN))
    assert not agent.is_done([], frame(state=GameState.NOT_FINISHED))


# ---------------------------------------------------------------------------
# World-model-active mode (requires torch + a real tiny checkpoint)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH, reason="PyTorch not available")
def test_world_model_active_with_trained_checkpoint(monkeypatch, tmp_path):
    from agent.simulator import ArcSimulator

    # Save a tiny (real, if untrained) simulator checkpoint. For the purpose of
    # this test the weights need only load and run; production requires a
    # genuinely trained checkpoint (R10).
    model = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    ckpt = tmp_path / "sim.pt"
    model.save_checkpoint(ckpt, model_version=MODEL_VERSION)

    agent = make_agent(monkeypatch, checkpoint=str(ckpt))
    assert agent._world_model_active is True
    assert agent._controller is not None

    action = agent.choose_action([], frame(grid=[[0, 1], [2, 3]], actions=(1, 2)))
    assert isinstance(action, GameAction)
    assert action.reasoning["world_model_active"] is True
    assert action.reasoning["controller"] == "internal-world-v0"


@pytest.mark.skipif(not _TORCH, reason="PyTorch not available")
def test_world_model_records_controller_decisions(monkeypatch, tmp_path):
    from agent.simulator import ArcSimulator

    model = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    ckpt = tmp_path / "sim.pt"
    model.save_checkpoint(ckpt, model_version=MODEL_VERSION)

    agent = make_agent(monkeypatch, checkpoint=str(ckpt))
    agent.choose_action([], frame(grid=[[0, 1], [2, 3]], actions=(1, 2)))
    agent.choose_action(
        [frame()], frame(grid=[[1, 1], [2, 3]], level=0, actions=(1, 2))
    )
    summary = agent.world_model_summary()
    assert summary["world_model_active"] is True
    assert summary["controller_decisions"] >= 1
    assert "controller" in summary


@pytest.mark.skipif(not _TORCH, reason="PyTorch not available")
def test_world_model_name_reflects_active_mode(monkeypatch, tmp_path):
    from agent.simulator import ArcSimulator

    model = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    ckpt = tmp_path / "sim.pt"
    model.save_checkpoint(ckpt, model_version=MODEL_VERSION)

    agent = make_agent(monkeypatch, checkpoint=str(ckpt))
    assert "world-model" in agent.name


@pytest.mark.skipif(not _TORCH, reason="PyTorch not available")
def test_world_model_runtime_error_degrades_to_baseline(monkeypatch, tmp_path):
    """If the controller raises mid-step, the agent must fall back, not crash."""
    from agent.simulator import ArcSimulator

    model = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    ckpt = tmp_path / "sim.pt"
    model.save_checkpoint(ckpt, model_version=MODEL_VERSION)
    agent = make_agent(monkeypatch, checkpoint=str(ckpt))

    # Sabotage the controller so decide() raises.
    class _Boom:
        def decide(self, obs):
            raise RuntimeError("boom")
        def correct(self, obs, decision):
            pass
    agent._controller = _Boom()

    action = agent.choose_action([], frame(grid=[[0, 1], [2, 3]], actions=(1, 2)))
    assert isinstance(action, GameAction)
    assert action.reasoning["world_model_active"] is False
    assert "boom" in action.reasoning["fallback_reason"]

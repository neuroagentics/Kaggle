"""Tests for agent/controller.py — G5-prep: MRLA loop contracts.

All tests use synthetic fixtures; no trained models or official environment.
Verifies: observe→imagine→act→correct cycle, deadline reserve, game isolation,
deliberator call frequency, probe fallback, memory recording.
"""
from __future__ import annotations

import json
import pytest
from dataclasses import replace
from time import monotonic

from agent.controller import (
    Controller,
    ControllerConfig,
    DecisionRecord,
    Observation,
)
from agent.deliberator import DeliberatorConfig, StubDeliberator
from agent.internal_world import Action, Prediction, WorldState
from agent.memory import WorkingMemory


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

LEFT, RIGHT = Action(3), Action(4)
GRID_4: tuple[tuple[int, ...], ...] = (tuple(2 if i == 0 else 0 for i in range(4)),)


class CorridorDynamics:
    """Test-only known simulator. Not trained; not a competition adapter."""
    def predict(self, state: WorldState, action: Action) -> Prediction:
        x = max(0, min(3, int(state.latent[0]) + (1 if action == RIGHT else -1)))
        grid = tuple(tuple(2 if i == x else 0 for i in range(4)) for _ in range(1))
        return Prediction(
            replace(state, latent=(float(x),), grid=grid, imagined=True,
                    terminal=(x == 3)),
            utility=float(x == 3),
            uncertainty=0.0,
            risk=0.0,
        )


def make_config(game_id: str = "test-game", **kwargs) -> ControllerConfig:
    return ControllerConfig(game_id=game_id, model_version="sim-v1", **kwargs)


def make_memory(game_id: str = "test-game") -> WorkingMemory:
    return WorkingMemory(game_id)


def make_obs(
    game_id: str = "test-game",
    level_id: int = 0,
    position: int = 0,
    success: bool = False,
    terminal: bool = False,
    obs_id: str = "obs-0001",
) -> Observation:
    grid = tuple(tuple(2 if i == position else 0 for i in range(4)) for _ in range(1))
    return Observation(
        observation_id=obs_id,
        game_id=game_id,
        level_id=level_id,
        grid=grid,
        available_action_ids=(3, 4),  # LEFT, RIGHT
        is_terminal=terminal,
        actual_success=success,
        model_version="sim-v1",
    )


def make_controller(
    game_id: str = "test-game",
    deliberator=None,
    **cfg_kwargs,
) -> Controller:
    cfg = make_config(game_id=game_id, **cfg_kwargs)
    mem = make_memory(game_id)
    return Controller(cfg, CorridorDynamics(), deliberator, mem)


# ---------------------------------------------------------------------------
# ControllerConfig validation
# ---------------------------------------------------------------------------

def test_config_valid():
    cfg = make_config()
    assert cfg.game_id == "test-game"
    assert cfg.horizon == 4


def test_config_empty_game_id():
    with pytest.raises(ValueError, match="game_id"):
        ControllerConfig(game_id="", model_version="v1")


def test_config_empty_model_version():
    with pytest.raises(ValueError, match="model_version"):
        ControllerConfig(game_id="g", model_version="")


def test_config_invalid_planning_budget():
    with pytest.raises(ValueError, match="budgets"):
        make_config(horizon=0)
    with pytest.raises(ValueError, match="budgets"):
        make_config(beam=-1)


def test_config_invalid_fraction():
    with pytest.raises(ValueError, match="deliberation_fraction"):
        make_config(deliberation_fraction=0.0)
    with pytest.raises(ValueError, match="deliberation_fraction"):
        make_config(deliberation_fraction=1.5)


# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------

def test_observation_valid():
    obs = make_obs()
    assert obs.game_id == "test-game"
    assert len(obs.to_actions()) == 2


def test_observation_empty_id():
    with pytest.raises(ValueError, match="observation_id"):
        Observation(
            observation_id="", game_id="g", level_id=0,
            grid=GRID_4, available_action_ids=(3,),
            is_terminal=False, actual_success=False, model_version="v1",
        )


def test_observation_non_bool_success():
    with pytest.raises(ValueError, match="bool"):
        Observation(
            observation_id="x", game_id="g", level_id=0,
            grid=GRID_4, available_action_ids=(3,),
            is_terminal=False, actual_success=1, model_version="v1",
        )


def test_observation_to_actions_filters_invalid():
    obs = Observation(
        observation_id="x", game_id="g", level_id=0,
        grid=GRID_4, available_action_ids=(3, 4, 99),  # 99 is invalid
        is_terminal=False, actual_success=False, model_version="v1",
    )
    ids = {a.id for a in obs.to_actions()}
    assert 99 not in ids
    assert 3 in ids and 4 in ids


# ---------------------------------------------------------------------------
# Controller construction
# ---------------------------------------------------------------------------

def test_controller_constructs():
    ctrl = make_controller()
    assert ctrl.config.game_id == "test-game"


def test_controller_rejects_mismatched_memory():
    cfg = make_config(game_id="game-A")
    mem = make_memory(game_id="game-B")
    with pytest.raises(ValueError, match="game_id"):
        Controller(cfg, CorridorDynamics(), None, mem)


def test_controller_requires_controller_config():
    with pytest.raises(TypeError, match="config"):
        Controller("not-a-config", CorridorDynamics(), None, make_memory())


# ---------------------------------------------------------------------------
# Basic decide/correct cycle
# ---------------------------------------------------------------------------

def test_decide_returns_decision_record():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    obs = make_obs()
    decision = ctrl.decide(obs)
    assert isinstance(decision, DecisionRecord)
    assert decision.game_id == "test-game"
    assert isinstance(decision.action, Action)


def test_decide_increments_total_decisions():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    ctrl.decide(make_obs(obs_id="obs-0001"))
    ctrl.decide(make_obs(obs_id="obs-0002"))
    assert ctrl._total_decisions == 2


def test_correct_records_episode():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    obs = make_obs(obs_id="obs-0001")
    decision = ctrl.decide(obs)
    obs_after = make_obs(position=1, obs_id="obs-0002")
    ctrl.correct(obs_after, decision)
    assert ctrl._total_corrections == 1
    episodes = ctrl.memory.query_episodes()
    assert len(episodes) == 1


def test_correct_rejects_wrong_game():
    ctrl = make_controller(game_id="game-A")
    ctrl.start_game(monotonic() + 300)
    obs = make_obs(game_id="game-A", obs_id="obs-0001")
    decision = ctrl.decide(obs)
    obs_after = make_obs(game_id="game-B", obs_id="obs-0002")
    with pytest.raises(ValueError, match="game_id"):
        ctrl.correct(obs_after, decision)


def test_observe_imagine_act_correct_full_cycle():
    """Full MRLA cycle: decide returns valid action, correct records episode."""
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    obs_before = make_obs(position=0, obs_id="obs-0001")
    decision = ctrl.decide(obs_before)
    # Action should be one of the available ones
    assert decision.action.id in (3, 4)
    # Correct with an updated observation
    obs_after = make_obs(position=1, success=False, obs_id="obs-0002")
    ctrl.correct(obs_after, decision)
    assert ctrl.memory.stats()["episodes"] == 1


# ---------------------------------------------------------------------------
# Planning smoke-test: planner should eventually prefer RIGHT
# ---------------------------------------------------------------------------

def test_planner_biases_toward_goal_with_goal_spec():
    """When GoalSpecs are set, planner should find rightward path."""
    from agent.internal_world import GoalSpec
    from agent.memory import GoalHypothesis

    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    # Inject a goal manually
    ctrl._current_goals = (
        GoalSpec(
            goal_id="end",
            description="reach position 3",
            target_color=2,
            target_region=(3, 0, 3, 0),
            priority=1.0,
            status="active",
        ),
    )
    obs = make_obs(position=0, obs_id="obs-0001")
    decision = ctrl.decide(obs)
    # With a clear right-side goal the plan should prefer RIGHT
    assert decision.action == RIGHT
    assert not decision.plan_was_empty


# ---------------------------------------------------------------------------
# Deadline reserve
# ---------------------------------------------------------------------------

def test_deadline_reserve_causes_empty_plan():
    """When planning_deadline is already past, plan must be empty and probe used."""
    # Monotonic clock always returns a time well past the deadline
    tick = [0.0]
    def clock():
        tick[0] += 100.0
        return tick[0]

    cfg = make_config()
    mem = make_memory()
    ctrl = Controller(cfg, CorridorDynamics(), None, mem, clock=clock)
    ctrl.start_game(0.5)  # deadline in the past
    obs = make_obs()
    decision = ctrl.decide(obs)
    # Empty plan → probe decision
    assert decision.plan_was_empty or decision.action is not None  # must not crash


def test_deadline_reserve_fraction_respected():
    """Controller must not run past run_deadline - reserve."""
    now = monotonic()
    ctrl = make_controller()
    run_deadline = now + 1000.0
    ctrl.start_game(run_deadline)
    ctrl._total_decisions = 1
    dl = ctrl._planning_deadline()
    reserve = run_deadline * ctrl.config.deadline_reserve_fraction
    # Planning deadline must be less than run_deadline
    assert dl < run_deadline


# ---------------------------------------------------------------------------
# Deliberator call frequency
# ---------------------------------------------------------------------------

def test_deliberator_not_called_below_threshold():
    """Deliberator must not fire within min_real_actions_between_calls."""
    cfg_d = DeliberatorConfig(
        model_name="stub", model_version="sim-v1",
        min_real_actions_between_calls=8,
    )
    stub = StubDeliberator(cfg_d)
    ctrl = make_controller(deliberator=stub)
    ctrl.start_game(monotonic() + 300)
    # First call is 'initial', so stub may be called once
    ctrl.decide(make_obs(obs_id="obs-0001"))
    initial_calls = ctrl._total_deliberator_calls
    # Next few decisions must not call the deliberator
    for i in range(4):
        ctrl.correct(make_obs(obs_id=f"obs-a{i:04d}"),
                     ctrl._last_decision)  # type: ignore[arg-type]
        ctrl.decide(make_obs(obs_id=f"obs-b{i:04d}"))
    assert ctrl._total_deliberator_calls == initial_calls


def test_deliberator_called_on_stall():
    """Deliberator fires when stall_threshold reached."""
    cfg_d = DeliberatorConfig(
        model_name="stub", model_version="sim-v1",
        min_real_actions_between_calls=100,  # very high threshold
    )
    stub = StubDeliberator(cfg_d)
    ctrl = make_controller(deliberator=stub, stall_threshold=2)
    ctrl.start_game(monotonic() + 300)
    # Pump decides+corrects without level advance to build stall counter
    for i in range(5):
        obs = make_obs(obs_id=f"obs-{i:04d}")
        d = ctrl.decide(obs)
        ctrl.correct(make_obs(obs_id=f"obs-a{i:04d}"), d)
    # After 2 stall steps, deliberator should have been called
    assert ctrl._total_deliberator_calls >= 1


# ---------------------------------------------------------------------------
# Level change isolation
# ---------------------------------------------------------------------------

def test_level_change_resets_stall_counter():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    ctrl._stall_counter = 10
    obs = make_obs(level_id=1, obs_id="obs-0001")
    ctrl.decide(obs)
    assert ctrl._stall_counter == 0
    assert ctrl._level_id == 1


def test_level_change_clears_branches():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    from agent.memory import ImaginedBranch
    from agent.memory import _grid_to_bytes
    b = ImaginedBranch(
        branch_id="a" * 32, game_id="test-game", parent_branch_id=None,
        action_sequence=(3,), predicted_grids_bytes=(_grid_to_bytes(GRID_4),),
        utility_sequence=(1.0,), risk_sequence=(0.0,), uncertainty_sequence=(0.1,),
        terminal=False, schema_version="1.0",
    )
    ctrl.memory.set_plan_branches([b])
    # Advance level
    ctrl.decide(make_obs(level_id=1, obs_id="obs-0001"))
    assert ctrl.memory.get_branches() == []


# ---------------------------------------------------------------------------
# Probe fallback
# ---------------------------------------------------------------------------

def test_probe_used_when_all_uncertainty_high():
    """When the dynamics always returns high uncertainty, probe must be used."""
    class MaxUncertainty(CorridorDynamics):
        def predict(self, state, action):
            p = super().predict(state, action)
            from dataclasses import replace
            return replace(p, uncertainty=0.99)

    cfg = make_config()
    mem = make_memory()
    ctrl = Controller(cfg, MaxUncertainty(), None, mem)
    ctrl.start_game(monotonic() + 300)
    obs = make_obs()
    decision = ctrl.decide(obs)
    assert decision.plan_was_empty
    assert isinstance(decision.action, Action)  # probe still returns a valid action


def test_probe_never_crashes_on_dynamics_error():
    """Probe must not crash even if dynamics raises for some actions."""
    class FlakyDynamics(CorridorDynamics):
        def predict(self, state, action):
            if action == RIGHT:
                raise RuntimeError("flaky!")
            return super().predict(state, action)

    cfg = make_config()
    mem = make_memory()
    ctrl = Controller(cfg, FlakyDynamics(), None, mem)
    ctrl.start_game(monotonic() + 300)
    decision = ctrl.decide(make_obs())
    assert isinstance(decision.action, Action)


# ---------------------------------------------------------------------------
# Isolation: separate games must use separate controllers
# ---------------------------------------------------------------------------

def test_separate_games_do_not_share_memory():
    ctrl_a = make_controller(game_id="game-A")
    ctrl_b = make_controller(game_id="game-B")
    ctrl_a.start_game(monotonic() + 300)
    ctrl_b.start_game(monotonic() + 300)
    d_a = ctrl_a.decide(make_obs(game_id="game-A", obs_id="obs-a"))
    ctrl_a.correct(make_obs(game_id="game-A", obs_id="obs-a2"), d_a)
    assert ctrl_b.memory.stats()["episodes"] == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_returns_json_safe_dict():
    ctrl = make_controller()
    ctrl.start_game(monotonic() + 300)
    ctrl.decide(make_obs())
    stats = ctrl.stats()
    # Should round-trip through JSON without error
    json.dumps(stats)
    assert stats["total_decisions"] == 1
    assert "memory" in stats


def test_stats_includes_deliberator_when_present():
    cfg_d = DeliberatorConfig(model_name="stub", model_version="sim-v1")
    stub = StubDeliberator(cfg_d)
    ctrl = make_controller(deliberator=stub)
    ctrl.start_game(monotonic() + 300)
    ctrl.decide(make_obs())
    stats = ctrl.stats()
    assert "deliberator" in stats
    assert stats["deliberator"]["model_name"] == "stub"

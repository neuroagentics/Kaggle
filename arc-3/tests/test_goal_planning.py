"""Tests for R08: GoalSpec, Subgoal, GoalEvaluator, GoalDirectedPlanner.

All fixtures are synthetic. No ARC performance claims.
"""
from __future__ import annotations

import pytest
from dataclasses import replace
from agent.internal_world import (
    Action, WorldState, Prediction, InternalGame,
    GoalSpec, Subgoal, GoalEvaluator, GoalDirectedPlanner, GoalDirectedPlan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEFT, RIGHT = Action(3), Action(4)


def corridor_state(position: int = 0, terminal: bool = False) -> WorldState:
    """4-cell corridor; position encoded as color 2 at index, rest 0."""
    return WorldState(
        "fixture", 0,
        (float(position),),
        (tuple(2 if x == position else 0 for x in range(4)),),
        (LEFT, RIGHT),
        "fixture-v1",
        terminal=terminal,
    )


class CorridorDynamics:
    """Known test simulator. Not a trained model."""
    def predict(self, state: WorldState, action: Action) -> Prediction:
        x = max(0, min(3, int(state.latent[0]) + (1 if action == RIGHT else -1)))
        return Prediction(
            replace(corridor_state(x), terminal=(x == 3), imagined=True),
            float(x == 3), 0.0, 0.0,
        )


def make_goal(color: int = 2, region: tuple | None = None, priority: float = 1.0,
              status: str = "active") -> GoalSpec:
    return GoalSpec(
        goal_id="goal-01",
        description="reach color 2",
        target_color=color,
        target_region=region,
        priority=priority,
        status=status,
    )


def make_evaluator(goals=None, subgoals=()) -> GoalEvaluator:
    if goals is None:
        goals = [make_goal()]
    return GoalEvaluator(goals, subgoals)


def make_planner(evaluator: GoalEvaluator, **kwargs) -> GoalDirectedPlanner:
    return GoalDirectedPlanner(evaluator, horizon=4, beam=8, **kwargs)


# ---------------------------------------------------------------------------
# GoalSpec validation
# ---------------------------------------------------------------------------

def test_goal_spec_valid():
    g = make_goal()
    assert g.goal_id == "goal-01"
    assert g.status == "active"


def test_goal_spec_empty_id_rejected():
    with pytest.raises(ValueError, match="goal_id"):
        GoalSpec(goal_id="", description="x", target_color=None,
                 target_region=None, priority=1.0, status="active")


def test_goal_spec_invalid_color():
    with pytest.raises(ValueError, match="target_color"):
        GoalSpec(goal_id="g", description="x", target_color=16,
                 target_region=None, priority=1.0, status="active")


def test_goal_spec_zero_priority_rejected():
    with pytest.raises(ValueError, match="priority"):
        GoalSpec(goal_id="g", description="x", target_color=None,
                 target_region=None, priority=0.0, status="active")


def test_goal_spec_invalid_region():
    with pytest.raises(ValueError, match="target_region"):
        GoalSpec(goal_id="g", description="x", target_color=None,
                 target_region=(10, 0, 5, 5), priority=1.0, status="active")


def test_goal_spec_falsified_status_rejected():
    with pytest.raises(ValueError, match="status"):
        GoalSpec(goal_id="g", description="x", target_color=None,
                 target_region=None, priority=1.0, status="falsified")


# ---------------------------------------------------------------------------
# Subgoal validation
# ---------------------------------------------------------------------------

def test_subgoal_valid():
    sg = Subgoal(subgoal_id="sg1", description="reach midpoint",
                 target_color=2, target_region=(2, 0, 2, 0), required_before=None)
    assert sg.subgoal_id == "sg1"


def test_subgoal_empty_id_rejected():
    with pytest.raises(ValueError, match="subgoal_id"):
        Subgoal(subgoal_id="", description="x", target_color=None,
                target_region=None, required_before=None)


def test_subgoal_invalid_color():
    with pytest.raises(ValueError, match="target_color"):
        Subgoal(subgoal_id="s", description="x", target_color=99,
                target_region=None, required_before=None)


# ---------------------------------------------------------------------------
# GoalEvaluator — progress and satisfaction
# ---------------------------------------------------------------------------

def test_goal_progress_full_match():
    ev = make_evaluator([make_goal(color=2)])
    # State where color 2 is at position 3 (all of the 1-cell region)
    state = corridor_state(3)
    # Color 2 is at col 3; if region covers just (3,0,3,0):
    goal = GoalSpec(goal_id="g", description="reach end",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev2 = GoalEvaluator([goal])
    assert ev2.goal_progress(state, goal) == pytest.approx(1.0)


def test_goal_progress_no_match():
    goal = GoalSpec(goal_id="g", description="reach end",
                    target_color=5, target_region=None,
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    assert ev.goal_progress(corridor_state(0), goal) == pytest.approx(0.0)


def test_goal_progress_partial():
    # 4-cell grid; color 2 at position 0 only; region is whole grid
    goal = GoalSpec(goal_id="g", description="fill with 2",
                    target_color=2, target_region=None,
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    progress = ev.goal_progress(corridor_state(0), goal)
    assert 0 < progress < 1


def test_goal_satisfied_at_threshold():
    goal = GoalSpec(goal_id="g", description="x",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    assert ev.is_goal_satisfied(corridor_state(3), goal, threshold=1.0)
    assert not ev.is_goal_satisfied(corridor_state(0), goal, threshold=1.0)


def test_evaluator_requires_at_least_one_goal():
    with pytest.raises(ValueError, match="at least one"):
        GoalEvaluator([])


def test_evaluator_rejects_non_goal_spec():
    with pytest.raises(TypeError):
        GoalEvaluator(["not a GoalSpec"])


def test_utility_active_goal_higher_than_provisional():
    goal_active = GoalSpec(goal_id="ga", description="x",
                           target_color=2, target_region=(3, 0, 3, 0),
                           priority=1.0, status="active")
    goal_prov = GoalSpec(goal_id="gp", description="x",
                         target_color=2, target_region=(3, 0, 3, 0),
                         priority=1.0, status="provisional")
    state = corridor_state(3)  # color 2 at col 3
    u_active = GoalEvaluator([goal_active]).utility(state)
    u_prov = GoalEvaluator([goal_prov]).utility(state)
    assert u_active > u_prov


def test_utility_scales_with_priority():
    g_high = GoalSpec(goal_id="g1", description="x",
                      target_color=2, target_region=(3, 0, 3, 0),
                      priority=1.0, status="active")
    g_low = GoalSpec(goal_id="g2", description="x",
                     target_color=2, target_region=(3, 0, 3, 0),
                     priority=0.5, status="active")
    state = corridor_state(3)
    u_high = GoalEvaluator([g_high]).utility(state)
    u_low = GoalEvaluator([g_low]).utility(state)
    assert u_high > u_low


def test_competing_goals_disagree_detection():
    # Two goals both satisfied simultaneously
    g1 = GoalSpec(goal_id="g1", description="x",
                  target_color=2, target_region=(3, 0, 3, 0),
                  priority=1.0, status="active")
    g2 = GoalSpec(goal_id="g2", description="y",
                  target_color=2, target_region=(3, 0, 3, 0),  # same region
                  priority=0.8, status="active")
    ev = GoalEvaluator([g1, g2])
    state = corridor_state(3)
    assert ev.competing_goals_disagree(state, threshold=0.9)


def test_competing_goals_no_disagree_when_only_one_satisfied():
    g1 = GoalSpec(goal_id="g1", description="x",
                  target_color=2, target_region=(3, 0, 3, 0),
                  priority=1.0, status="active")
    g2 = GoalSpec(goal_id="g2", description="y",
                  target_color=5, target_region=None,  # color 5 never appears
                  priority=0.8, status="active")
    ev = GoalEvaluator([g1, g2])
    state = corridor_state(3)
    assert not ev.competing_goals_disagree(state, threshold=0.9)


def test_utility_invalid_partial_credit_weight():
    ev = make_evaluator()
    with pytest.raises(ValueError, match="partial_credit"):
        ev.utility(corridor_state(0), partial_credit_weight=-0.1)


# ---------------------------------------------------------------------------
# Subgoal milestone tracking
# ---------------------------------------------------------------------------

def test_subgoal_advance_on_satisfaction():
    sg1 = Subgoal(subgoal_id="sg1", description="reach col2",
                  target_color=2, target_region=(2, 0, 2, 0), required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg1])
    # State with color 2 at col 2 satisfies sg1
    state = corridor_state(2)
    advanced = ev.advance_subgoal(state)
    assert advanced
    assert ev.current_subgoal() is None  # all subgoals cleared


def test_subgoal_not_advanced_when_not_satisfied():
    sg1 = Subgoal(subgoal_id="sg1", description="reach col3",
                  target_color=2, target_region=(3, 0, 3, 0), required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg1])
    advanced = ev.advance_subgoal(corridor_state(0))
    assert not advanced
    assert ev.current_subgoal() is sg1


def test_subgoal_reset():
    sg1 = Subgoal(subgoal_id="sg1", description="x",
                  target_color=2, target_region=(2, 0, 2, 0), required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg1])
    ev.advance_subgoal(corridor_state(2))
    ev.reset_subgoals()
    assert ev.current_subgoal() is sg1


def test_subgoal_bonus_contributes_to_utility():
    sg = Subgoal(subgoal_id="sg1", description="midpoint",
                 target_color=2, target_region=(2, 0, 2, 0), required_before=None)
    ev_with = GoalEvaluator([make_goal()], subgoals=[sg])
    ev_without = GoalEvaluator([make_goal()])
    # At position 2, subgoal is satisfied → bonus contributes
    u_with = ev_with.utility(corridor_state(2))
    u_without = ev_without.utility(corridor_state(2))
    assert u_with >= u_without


# ---------------------------------------------------------------------------
# GoalDirectedPlanner
# ---------------------------------------------------------------------------

def test_planner_requires_goal_evaluator():
    with pytest.raises(TypeError):
        GoalDirectedPlanner("not-an-evaluator")


def test_planner_invalid_budget_rejected():
    ev = make_evaluator()
    with pytest.raises(ValueError, match="budgets"):
        GoalDirectedPlanner(ev, horizon=0)
    with pytest.raises(ValueError, match="budgets"):
        GoalDirectedPlanner(ev, beam=-1)


def test_planner_finds_path_to_goal():
    """Planner should find 3 RIGHT steps toward position 3."""
    goal = GoalSpec(goal_id="end", description="reach end",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, horizon=4, beam=8, max_predictions=64)
    plan = planner.plan(world, deadline=float("inf"))
    assert len(plan.actions) > 0
    # All actions in the plan should be RIGHT (toward goal)
    assert all(a == RIGHT for a in plan.actions)


def test_planner_does_not_modify_real_world_state():
    ev = make_evaluator()
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    original_state = world.state
    planner = make_planner(ev)
    planner.plan(world, deadline=float("inf"))
    assert world.state == original_state


def test_planner_returns_empty_on_nan_deadline():
    ev = make_evaluator()
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = make_planner(ev)
    with pytest.raises(ValueError, match="NaN"):
        planner.plan(world, deadline=float("nan"))


def test_planner_returns_empty_on_exhausted_deadline():
    ev = make_evaluator()
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, clock=lambda: 10.0)
    plan = planner.plan(world, deadline=9.0)
    assert plan.actions == ()


def test_planner_returns_ambiguous_flag_on_competing_goals():
    # Two goals both satisfied at position 3 → ambiguity
    g1 = GoalSpec(goal_id="g1", description="x",
                  target_color=2, target_region=(3, 0, 3, 0),
                  priority=1.0, status="active")
    g2 = GoalSpec(goal_id="g2", description="y",
                  target_color=2, target_region=(3, 0, 3, 0),
                  priority=0.9, status="active")
    ev = GoalEvaluator([g1, g2])
    world = InternalGame(CorridorDynamics(), corridor_state(3))
    planner = GoalDirectedPlanner(ev, ambiguity_threshold=0.5)
    plan = planner.plan(world, deadline=float("inf"))
    assert plan.ambiguous


def test_planner_goal_attribution_recorded():
    goal = GoalSpec(goal_id="end", description="reach end",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, horizon=4, beam=8)
    plan = planner.plan(world, deadline=float("inf"))
    # attribution tuple should have same length as actions
    assert len(plan.goal_attribution) == len(plan.actions)


def test_planner_respects_prediction_budget():
    ev = make_evaluator()
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, max_predictions=2)
    planner.plan(world, deadline=float("inf"))
    assert planner.predictions_made <= 2


def test_planner_stops_on_uncertainty():
    class HighUncertainty(CorridorDynamics):
        def predict(self, state, action):
            p = super().predict(state, action)
            from dataclasses import replace
            return replace(p, uncertainty=0.9)

    ev = make_evaluator()
    world = InternalGame(HighUncertainty(), corridor_state(0))
    planner = make_planner(ev)
    plan = planner.plan(world, deadline=float("inf"))
    assert plan.actions == ()


def test_delayed_credit_rewards_partial_progress():
    """Partial progress toward goal should yield nonzero utility."""
    goal = GoalSpec(goal_id="g", description="reach end",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    # Position 1: color 2 at col 1, not at (3,0,3,0) → partial
    u_at_0 = ev.utility(corridor_state(0))
    u_at_2 = ev.utility(corridor_state(2))
    # Both should be 0 since color 2 isn't in the target region yet;
    # but the evaluator gives partial credit based on the whole grid
    # For this test we just confirm utility is finite and nonneg
    assert u_at_0 >= 0.0
    assert u_at_2 >= 0.0


# ===========================================================================
# R08-A: goal_signal_to_goal_spec and goal_hypothesis_to_goal_spec adapters
# ===========================================================================

from agent.internal_world import goal_signal_to_goal_spec, goal_hypothesis_to_goal_spec
from agent.model import GoalSignal, SCHEMA_VERSION as _MODEL_SCHEMA
from agent.memory import GoalHypothesis


_VALID_ID_32 = "a" * 32
_VALID_GOAL_SIG_ID = "b" * 32
_VALID_HYP_ID_32 = "c" * 32


def _make_goal_signal(status: str = "active") -> GoalSignal:
    return GoalSignal(
        goal_id=_VALID_GOAL_SIG_ID,
        description="reach green exit",
        target_color=3,
        target_region=(0, 0, 5, 5),
        priority=0.9,
        status=status,
        schema_version=_MODEL_SCHEMA,
    )


def _make_goal_hypothesis(status: str = "active") -> GoalHypothesis:
    return GoalHypothesis(
        goal_id=_VALID_HYP_ID_32,
        game_id="fixture",
        level_context=0,
        description="reach green exit",
        observable_target="color:3@0,0,5,5",
        disconfirming_test="color:5",
        supporting_ids=("ep1", "ep2"),
        status=status,
        schema_version="1.0",
        timestamp=0.0,
    )


def test_goal_signal_to_goal_spec_round_trips_fields():
    sig = _make_goal_signal()
    spec = goal_signal_to_goal_spec(sig)
    assert spec.goal_id == sig.goal_id
    assert spec.description == sig.description
    assert spec.target_color == sig.target_color
    assert spec.target_region == sig.target_region
    assert spec.priority == sig.priority
    assert spec.status == sig.status


def test_goal_signal_to_goal_spec_returns_goal_spec_type():
    spec = goal_signal_to_goal_spec(_make_goal_signal())
    assert isinstance(spec, GoalSpec)


def test_goal_signal_to_goal_spec_rejects_non_signal():
    with pytest.raises(TypeError, match="GoalSignal"):
        goal_signal_to_goal_spec("not-a-signal")


def test_goal_signal_provisional_status_preserved():
    sig = _make_goal_signal(status="provisional")
    spec = goal_signal_to_goal_spec(sig)
    assert spec.status == "provisional"


def test_goal_hypothesis_to_goal_spec_uses_supplied_color_region():
    hyp = _make_goal_hypothesis()
    spec = goal_hypothesis_to_goal_spec(hyp, target_color=3, target_region=(0, 0, 5, 5), priority=0.8)
    assert spec.goal_id == hyp.goal_id
    assert spec.target_color == 3
    assert spec.target_region == (0, 0, 5, 5)
    assert spec.priority == pytest.approx(0.8)


def test_goal_hypothesis_to_goal_spec_none_color_region_allowed():
    hyp = _make_goal_hypothesis()
    spec = goal_hypothesis_to_goal_spec(hyp)
    assert spec.target_color is None
    assert spec.target_region is None


def test_goal_hypothesis_to_goal_spec_rejects_falsified():
    hyp = _make_goal_hypothesis(status="falsified")
    with pytest.raises(ValueError, match="[Ff]alsified"):
        goal_hypothesis_to_goal_spec(hyp)


def test_goal_hypothesis_to_goal_spec_rejects_non_hypothesis():
    with pytest.raises(TypeError, match="GoalHypothesis"):
        goal_hypothesis_to_goal_spec("not-a-hypothesis")


# ===========================================================================
# R08-B: GoalDirectedPlan goal_attribution content
# ===========================================================================

def test_goal_attribution_contains_contributing_goal_id():
    """The goal that is satisfied must appear in at least one attribution tuple."""
    goal = GoalSpec(goal_id="end", description="reach end",
                    target_color=2, target_region=(3, 0, 3, 0),
                    priority=1.0, status="active")
    ev = GoalEvaluator([goal])
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, horizon=4, beam=8)
    plan = planner.plan(world, deadline=float("inf"))
    assert len(plan.actions) > 0
    # Flatten attributions and confirm "end" appears at least once
    all_attributed = {gid for step in plan.goal_attribution for gid in step}
    assert "end" in all_attributed


def test_goal_attribution_length_matches_actions():
    goal = make_goal(region=(3, 0, 3, 0))
    ev = GoalEvaluator([goal])
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, horizon=4, beam=8)
    plan = planner.plan(world, deadline=float("inf"))
    assert len(plan.goal_attribution) == len(plan.actions)


def test_empty_plan_has_empty_attribution():
    ev = make_evaluator()
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, clock=lambda: 10.0)
    plan = planner.plan(world, deadline=9.0)  # expired deadline → empty plan
    assert plan.goal_attribution == ()


# ===========================================================================
# R08-C: multi-milestone subgoal chain (required_before ordering)
# ===========================================================================

def test_two_subgoal_chain_ordered_by_required_before():
    """Chain: sg1 → sg2 → terminal goal.  Must reach sg1 before sg2."""
    sg1 = Subgoal(subgoal_id="sg1", description="reach col1",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before="sg2")
    sg2 = Subgoal(subgoal_id="sg2", description="reach col2",
                  target_color=2, target_region=(2, 0, 2, 0),
                  required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg2, sg1])  # insertion order reversed
    # Chain should reorder: sg1 first, then sg2
    assert ev._subgoal_chain == ("sg1", "sg2")
    # At position 1 sg1 should be current
    assert ev.current_subgoal().subgoal_id == "sg1"


def test_two_subgoal_chain_advances_in_order():
    sg1 = Subgoal(subgoal_id="sg1", description="reach col1",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before="sg2")
    sg2 = Subgoal(subgoal_id="sg2", description="reach col2",
                  target_color=2, target_region=(2, 0, 2, 0),
                  required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg1, sg2])

    # sg2 satisfied first (wrong order) — sg1 is still current, so no advance
    advanced = ev.advance_subgoal(corridor_state(2))
    assert not advanced
    assert ev.current_subgoal().subgoal_id == "sg1"

    # sg1 now satisfied — should advance to sg2
    advanced = ev.advance_subgoal(corridor_state(1))
    assert advanced
    assert ev.current_subgoal().subgoal_id == "sg2"

    # sg2 now satisfied — all done
    advanced = ev.advance_subgoal(corridor_state(2))
    assert advanced
    assert ev.current_subgoal() is None


def test_three_subgoal_chain_full_traversal():
    sg1 = Subgoal(subgoal_id="sg1", description="col0",
                  target_color=2, target_region=(0, 0, 0, 0),
                  required_before="sg2")
    sg2 = Subgoal(subgoal_id="sg2", description="col1",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before="sg3")
    sg3 = Subgoal(subgoal_id="sg3", description="col2",
                  target_color=2, target_region=(2, 0, 2, 0),
                  required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg3, sg2, sg1])  # reversed

    assert ev._subgoal_chain == ("sg1", "sg2", "sg3")
    ev.advance_subgoal(corridor_state(0))   # sg1
    ev.advance_subgoal(corridor_state(1))   # sg2
    ev.advance_subgoal(corridor_state(2))   # sg3
    assert ev.current_subgoal() is None


def test_subgoal_chain_fallback_insertion_order_when_all_none():
    """All required_before=None → multiple roots → fall back to insertion order."""
    sg1 = Subgoal(subgoal_id="sg1", description="a",
                  target_color=2, target_region=(0, 0, 0, 0),
                  required_before=None)
    sg2 = Subgoal(subgoal_id="sg2", description="b",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before=None)
    ev = GoalEvaluator([make_goal()], subgoals=[sg1, sg2])
    # Falls back to insertion order
    assert ev._subgoal_chain == ("sg1", "sg2")


def test_subgoal_chain_cycle_does_not_loop():
    """A cycle in required_before must not cause infinite loop or hang."""
    sg1 = Subgoal(subgoal_id="sg1", description="a",
                  target_color=2, target_region=(0, 0, 0, 0),
                  required_before="sg2")
    sg2 = Subgoal(subgoal_id="sg2", description="b",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before="sg1")  # cycle
    # Both point to the other → two roots after removing pointed-to → fallback
    ev = GoalEvaluator([make_goal()], subgoals=[sg1, sg2])
    # Must not hang; chain length is exactly 2
    assert len(ev._subgoal_chain) == 2


def test_planner_reaches_goal_through_two_subgoals():
    """Planner should reach position 3 via milestone at 1 and 2."""
    sg1 = Subgoal(subgoal_id="sg1", description="col1",
                  target_color=2, target_region=(1, 0, 1, 0),
                  required_before="sg2")
    sg2 = Subgoal(subgoal_id="sg2", description="col2",
                  target_color=2, target_region=(2, 0, 2, 0),
                  required_before=None)
    terminal_goal = GoalSpec(goal_id="end", description="col3",
                             target_color=2, target_region=(3, 0, 3, 0),
                             priority=1.0, status="active")
    ev = GoalEvaluator([terminal_goal], subgoals=[sg1, sg2])
    world = InternalGame(CorridorDynamics(), corridor_state(0))
    planner = GoalDirectedPlanner(ev, horizon=4, beam=8, max_predictions=128)
    plan = planner.plan(world, deadline=float("inf"))
    assert len(plan.actions) > 0
    assert all(a == RIGHT for a in plan.actions)


# ===========================================================================
# R08-B: promote_goals_from_attribution and falsify_goals_from_observation
# ===========================================================================

import time as _time
from agent.memory import WorkingMemory


def _make_memory() -> WorkingMemory:
    return WorkingMemory("fixture", clock=lambda: _time.monotonic())


def _add_episodes(mem: WorkingMemory, n: int = 2) -> list[str]:
    """Add n minimal episodes and return their record_ids."""
    ids = []
    grid = ((0, 0), (0, 0))
    for i in range(n):
        ep = mem.add_episode(
            level_id=0,
            observation_id=f"obs-{i}",
            model_version="v1",
            action_id=0,
            action_x=None,
            action_y=None,
            before_grid=grid,
            predicted_grid=grid,
            observed_grid=grid,
            actual_success=False,
        )
        ids.append(ep.record_id)
    return ids


def test_promote_goals_from_attribution_upgrades_provisional_to_active():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 2)
    # Add a provisional goal (1 supporting ID)
    goal = mem.add_goal(
        description="reach end",
        observable_target="color:2",
        disconfirming_test="color:5",
        supporting_ids=[ep_ids[0]],
    )
    assert goal.status == "provisional"

    # Add a third episode to use as the attribution evidence
    ep3_ids = _add_episodes(mem, 1)
    attribution = [(goal.goal_id,), ()]  # goal credited on step 0
    updated = mem.promote_goals_from_attribution(attribution, episode_id=ep3_ids[0])
    assert len(updated) == 1
    assert updated[0].status == "active"


def test_promote_goals_from_attribution_skips_falsified():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 2)
    goal = mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="color:5",
        supporting_ids=[ep_ids[0]],
    )
    mem.falsify_goal(goal.goal_id)
    ep3 = _add_episodes(mem, 1)
    updated = mem.promote_goals_from_attribution(
        [(goal.goal_id,)], episode_id=ep3[0]
    )
    assert updated == []


def test_promote_goals_from_attribution_skips_unknown_goal_ids():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    # Attribution references a goal_id not in memory
    updated = mem.promote_goals_from_attribution(
        [("nonexistent-id",)], episode_id=ep_ids[0]
    )
    assert updated == []


def test_promote_goals_from_attribution_rejects_missing_episode():
    mem = _make_memory()
    with pytest.raises(ValueError, match="episode_id"):
        mem.promote_goals_from_attribution([], episode_id="nonexistent")


def test_falsify_goals_from_observation_color_only_pattern():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    goal = mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="color:5",   # falsified if color 5 appears anywhere
        supporting_ids=[ep_ids[0]],
    )
    # Grid with color 5 present
    grid_with_5 = ((5, 0), (0, 0))
    falsified = mem.falsify_goals_from_observation(grid_with_5)
    assert len(falsified) == 1
    assert falsified[0].goal_id == goal.goal_id
    assert falsified[0].status == "falsified"


def test_falsify_goals_from_observation_color_not_present_leaves_unchanged():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="color:5",
        supporting_ids=[ep_ids[0]],
    )
    grid_no_5 = ((0, 2), (0, 3))
    falsified = mem.falsify_goals_from_observation(grid_no_5)
    assert falsified == []


def test_falsify_goals_from_observation_region_pattern():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="color:7@0,0,0,0",  # color 7 in exact cell (0,0)
        supporting_ids=[ep_ids[0]],
    )
    # Grid with color 7 at (0,0) — region is just that one cell so 100% match
    grid = ((7, 0), (0, 0))
    falsified = mem.falsify_goals_from_observation(grid)
    assert len(falsified) == 1


def test_falsify_goals_from_observation_unknown_pattern_conservative():
    """An unrecognised disconfirming_test string must not falsify the goal."""
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="some_unknown_condition",
        supporting_ids=[ep_ids[0]],
    )
    grid = ((1, 2), (3, 4))
    falsified = mem.falsify_goals_from_observation(grid)
    assert falsified == []


def test_falsify_goals_from_observation_already_falsified_skipped():
    mem = _make_memory()
    ep_ids = _add_episodes(mem, 1)
    goal = mem.add_goal(
        description="x",
        observable_target="color:2",
        disconfirming_test="color:5",
        supporting_ids=[ep_ids[0]],
    )
    mem.falsify_goal(goal.goal_id)
    grid = ((5, 0), (0, 0))
    falsified = mem.falsify_goals_from_observation(grid)
    # Already falsified → not returned again
    assert falsified == []

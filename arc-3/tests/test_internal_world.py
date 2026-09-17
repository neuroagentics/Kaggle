"""Deterministic fixtures test control mechanics, never claim ARC performance."""
from dataclasses import replace
import pytest
from agent.internal_world import (
    Action, WorldState, Prediction, InternalGame, RecursivePlanner, EpisodeMemory,
)

LEFT, RIGHT = Action(3), Action(4)


def state(position=0):
    return WorldState("fixture", 0, (float(position),),
                      (tuple(2 if x == position else 0 for x in range(4)),),
                      (LEFT, RIGHT), "fixture-v1")


class Corridor:
    """Test-only known simulator. Not a trained model or competition adapter."""
    def predict(self, current, action):
        x = max(0, min(3, int(current.latent[0]) + (1 if action == RIGHT else -1)))
        return Prediction(replace(state(x), terminal=x == 3), float(x == 3), 0.0, 0.0)


def test_recursive_play_finds_three_step_win_without_touching_live_state():
    world = InternalGame(Corridor(), state())
    plan = RecursivePlanner(horizon=3, beam=8).plan(world, deadline=float("inf"))
    assert plan.actions == (RIGHT, RIGHT, RIGHT)
    assert [p.state.latent for p in plan.predictions] == [(1.0,), (2.0,), (3.0,)]
    assert all(p.state.imagined for p in plan.predictions)
    assert world.state == state()


def test_branching_keeps_sibling_and_real_state_isolated():
    world = InternalGame(Corridor(), state(1))
    a, b = world.fork(), world.fork()
    a.step(LEFT)
    b.step(RIGHT)
    assert (a.state.latent, b.state.latent, world.state.latent) == ((0.0,), (2.0,), (1.0,))
    with pytest.raises(ValueError, match="unavailable"):
        world.step(Action(5))


def test_uncertainty_stops_imagined_control_and_requests_calibration():
    class Unknown(Corridor):
        def predict(self, current, action):
            return replace(super().predict(current, action), uncertainty=0.9)
    plan = RecursivePlanner().plan(InternalGame(Unknown(), state()), deadline=float("inf"))
    assert plan.actions == ()


def test_limits_bound_model_calls():
    planner = RecursivePlanner(horizon=50, max_predictions=3)
    planner.plan(InternalGame(Corridor(), state()), deadline=float("inf"))
    assert planner.predictions_made == 3
    planner = RecursivePlanner(clock=lambda: 10)
    assert not planner.plan(InternalGame(Corridor(), state()), deadline=9).actions
    assert planner.predictions_made == 0


def test_real_feedback_is_immediately_available_and_rejects_imagined_evidence():
    memory = EpisodeMemory("fixture")
    prediction = InternalGame(Corridor(), state()).step(RIGHT).state
    event = memory.record(state(), RIGHT, prediction, state())
    assert event.changed_cell_error == 2
    assert memory.records == [event]
    with pytest.raises(ValueError, match="real endpoints"):
        memory.record(state(), RIGHT, prediction, prediction)
    with pytest.raises(ValueError, match="boundary"):
        memory.record(state(), RIGHT, prediction, replace(state(), game="other"))


def test_action_coordinates_and_arc3_palette_are_checked():
    with pytest.raises(ValueError):
        Action(6, 64, 0)
    with pytest.raises(ValueError):
        Action(4, 0, 0)
    assert replace(state(), grid=((15,),)).grid == ((15,),)
    with pytest.raises(ValueError):
        replace(state(), grid=((16,),))


def test_caller_owned_lists_cannot_mutate_world_or_sibling():
    grid, latent, actions = [[2, 0]], [0.0], [RIGHT]
    frozen = WorldState("fixture", 0, latent, grid, actions, "fixture-v1")
    grid[0][0], latent[0] = 8, 9.0
    actions.clear()
    assert frozen.grid == ((2, 0),)
    assert frozen.latent == (0.0,)
    assert frozen.actions == (RIGHT,)


@pytest.mark.parametrize("options", [
    {"horizon": 1.5}, {"beam": True}, {"max_predictions": 0},
    {"uncertainty_limit": float("nan")}, {"uncertainty_limit": 2},
    {"action_cost": -1}, {"risk_weight": float("inf")},
])
def test_invalid_planner_configuration_rejected(options):
    with pytest.raises(ValueError):
        RecursivePlanner(**options)


def test_late_prediction_is_not_selected():
    ticks = iter([0.0, 2.0])
    planner = RecursivePlanner(clock=lambda: next(ticks))
    assert not planner.plan(InternalGame(Corridor(), state()), deadline=1.0).actions
    assert planner.predictions_made == 1


def test_prediction_version_and_success_type_are_validated():
    memory = EpisodeMemory("fixture")
    predicted = InternalGame(Corridor(), state()).step(RIGHT).state
    with pytest.raises(ValueError, match="version"):
        memory.record(state(), RIGHT, replace(predicted, model_version="stale"), state())
    with pytest.raises(ValueError, match="boolean"):
        memory.record(state(), RIGHT, predicted, state(), actual_success="False")

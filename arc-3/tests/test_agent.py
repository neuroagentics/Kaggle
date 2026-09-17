from __future__ import annotations

from arcengine import FrameData, GameAction, GameState

from agent.my_agent import (
    MyAgent,
    coordinate_proposals,
    frame_delta,
    frame_grid,
    state_key,
)


def make_agent() -> MyAgent:
    return MyAgent(
        card_id="test",
        game_id="unit-test-game",
        agent_name="unit",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=None,
    )


def test_state_key_is_stable_and_includes_level() -> None:
    frame = FrameData(frame=[[[0, 1], [2, 3]]], levels_completed=0)
    same = FrameData(frame=[[[0, 1], [2, 3]]], levels_completed=0)
    later = FrameData(frame=[[[0, 1], [2, 3]]], levels_completed=1)
    assert state_key(frame) == state_key(same)
    assert state_key(frame) != state_key(later)


def test_state_key_ignores_engine_ui_bands() -> None:
    first = FrameData(frame=[[[0] * 8 for _ in range(8)]], levels_completed=0)
    changed = [[0] * 8 for _ in range(8)]
    changed[0][3] = 7
    changed[1][5] = 9
    changed[6][0] = 4
    second = FrameData(frame=[changed], levels_completed=0)
    assert state_key(first) == state_key(second)


def test_state_key_keeps_small_game_content() -> None:
    blank = FrameData(frame=[[[0, 0], [0, 0]]], levels_completed=0)
    changed = FrameData(frame=[[[0, 1], [0, 0]]], levels_completed=0)
    assert state_key(blank) != state_key(changed)


def test_frame_grid_uses_settled_final_animation_frame() -> None:
    frame = FrameData(
        frame=[
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
        ]
    )
    assert frame_grid(frame) == [[2, 2], [2, 2]]


def test_coordinate_proposals_include_changed_region_and_object() -> None:
    previous = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    current = [[0, 0, 0], [0, 5, 5], [0, 0, 0]]
    proposals = coordinate_proposals(current, previous)
    assert (2, 1) in proposals or (1, 1) in proposals
    assert len(proposals) == len(set(proposals))


def test_agent_resets_before_play() -> None:
    agent = make_agent()
    frame = FrameData(state=GameState.NOT_PLAYED, frame=[[[0]]])
    assert agent.choose_action([], frame) is GameAction.RESET


def test_agent_respects_available_actions() -> None:
    agent = make_agent()
    frame = FrameData(
        state=GameState.NOT_FINISHED,
        frame=[[[0, 0], [0, 0]]],
        available_actions=[2],
    )
    assert agent.choose_action([], frame) is GameAction.ACTION2


def test_agent_suppresses_observed_no_op() -> None:
    agent = make_agent()
    frame = FrameData(
        state=GameState.NOT_FINISHED,
        frame=[[[0, 0], [0, 0]]],
        available_actions=[1, 2],
    )
    first = agent.choose_action([], frame)
    second = agent.choose_action([frame], frame)
    assert second is not first


def test_click_candidates_cover_component_centre_and_corner() -> None:
    grid = [[0] * 16 for _ in range(16)]
    for y in range(4, 9):
        for x in range(4, 9):
            grid[y][x] = 3
    proposals = coordinate_proposals(grid)
    assert (6, 6) in proposals
    assert (4, 4) in proposals


def test_frame_delta_classifies_unchanged_sparse_and_resize() -> None:
    blank = [[0] * 10 for _ in range(10)]
    sparse = [row[:] for row in blank]
    sparse[4][5] = 3
    assert frame_delta(blank, blank)["class"] == "unchanged"
    assert frame_delta(blank, sparse) == {
        "class": "sparse",
        "changed_cells": 1,
        "bbox": [5, 4, 5, 4],
        "before_shape": [10, 10],
        "after_shape": [10, 10],
    }
    assert frame_delta(blank, [[0]])["class"] == "resize"


def test_agent_exposes_transition_and_efficiency_telemetry() -> None:
    agent = make_agent()
    first = FrameData(
        state=GameState.NOT_FINISHED,
        frame=[[[0, 0], [0, 0]]],
        available_actions=[1, 2],
        levels_completed=0,
    )
    agent.choose_action([], first)
    progressed = FrameData(
        state=GameState.NOT_FINISHED,
        frame=[[[0, 1], [0, 0]]],
        available_actions=[1, 2],
        levels_completed=1,
    )
    agent.choose_action([first], progressed)

    summary = agent.telemetry_summary()
    assert summary["actions_selected"] == 2
    assert summary["transitions_observed"] == 1
    assert summary["levels_gained"] == 1
    assert summary["actions_per_level"] == 2.0
    assert summary["delta_classes"] == {"sparse": 1}
    events = agent.transition_events()
    assert len(events) == 1
    assert events[0]["level_delta"] == 1
    assert events[0]["delta"]["changed_cells"] == 1


def test_movement_scoring_has_no_stateful_side_effect() -> None:
    agent = make_agent()
    grid = [[0] * 12 for _ in range(12)]
    grid[5][5] = 2
    grid[5][7] = 3
    agent._moves[1] = (1, 0)
    agent._avatar_color = 2
    candidate = (1, None, None)

    first = agent._movement_bonus(candidate, grid)
    second = agent._movement_bonus(candidate, grid)
    assert first == second
    assert agent._visited_targets == set()

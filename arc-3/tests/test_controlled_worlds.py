"""Tests for the Step 1 controlled-world generators.

These lock in the EXACT known transition functions so the learning diagnostic
stays trustworthy: if a world's rule silently changes, the measured changed-cell
accuracy would no longer mean what the Step 1 report claims.
"""
import random

from training.controlled_worlds import (
    make_movement_world, make_collision_world, make_click_world, make_delayed_world,
    generate_trajectory, generate_dataset,
    A_UP, A_DOWN, A_LEFT, A_RIGHT, A_CLICK,
    AGENT, BG, WALL, TARGET, TARGET_ON, BUTTON, DOOR_CLOSED, DOOR_OPEN,
)


def test_movement_moves_agent_and_clears_old_cell():
    rng = random.Random(0)
    w = make_movement_world(rng, 8, 8)
    (r, c) = w.extra["agent"]
    # Move in a direction that stays in bounds.
    if c + 1 < w.w:
        w.step(A_RIGHT)
        assert w.extra["agent"] == (r, c + 1)
        assert w.grid[r][c] == BG
        assert w.grid[r][c + 1] == AGENT


def test_movement_clamps_at_edge():
    rng = random.Random(1)
    w = make_movement_world(rng, 5, 5)
    w.extra["agent"] = (0, 0)
    w.grid = [[BG] * 5 for _ in range(5)]
    w.grid[0][0] = AGENT
    w.step(A_UP)  # already at top row
    assert w.extra["agent"] == (0, 0)  # clamped, no move


def test_collision_wall_blocks_move():
    rng = random.Random(2)
    w = make_collision_world(rng, 6, 6)
    # Place agent directly left of a wall cell and push right into it.
    wall_c = 6 // 2
    wall_rows = [r for r in range(6) if w.grid[r][wall_c] == WALL]
    r = wall_rows[0]
    # Rebuild a clean scenario: wall at (r, wall_c), agent just left of it.
    w.grid = [[BG] * 6 for _ in range(6)]
    w.grid[r][wall_c] = WALL
    w.grid[r][wall_c - 1] = AGENT
    w.extra["agent"] = (r, wall_c - 1)
    before = [row[:] for row in w.grid]
    w.step(A_RIGHT)
    assert w.extra["agent"] == (r, wall_c - 1)   # blocked
    assert w.grid == before                       # completely unchanged


def test_click_activates_only_target():
    rng = random.Random(3)
    w = make_click_world(rng, 7, 7)
    (tr, tc) = w.extra["target"]
    assert w.grid[tr][tc] == TARGET
    w.step(A_CLICK, tc, tr)  # x=col, y=row
    assert w.grid[tr][tc] == TARGET_ON
    # Off-target click changes nothing.
    snapshot = [row[:] for row in w.grid]
    other_r, other_c = (tr + 2) % 7, (tc + 3) % 7
    w.step(A_CLICK, other_c, other_r)
    assert w.grid == snapshot


def test_delayed_opens_door_after_delay_not_immediately():
    rng = random.Random(4)
    w = make_delayed_world(rng, 6, 6, delay=2)
    (br, bc) = w.extra["button"]
    (dr, dc) = w.extra["door"]
    assert w.grid[dr][dc] == DOOR_CLOSED
    w.step(A_CLICK, bc, br)                 # arm
    assert w.grid[dr][dc] == DOOR_CLOSED    # not immediate
    assert w.extra["timer"] == 2
    w.step(A_CLICK, 0, 0)                   # tick 1
    assert w.grid[dr][dc] == DOOR_CLOSED
    w.step(A_CLICK, 0, 0)                   # tick 2 -> opens
    assert w.grid[dr][dc] == DOOR_OPEN


def test_generate_trajectory_schema():
    rec = generate_trajectory("movement", steps=5, seed=0, h=10, w=10)
    assert rec["game_id"] == "cw_movement"
    assert len(rec["steps"]) == 6  # steps + terminal
    last = rec["steps"][-1]
    assert last["action_id"] is None  # terminal has no action
    for s in rec["steps"]:
        assert "grid" in s and "available_actions" in s
        assert len(s["grid"]) == 10 and len(s["grid"][0]) == 10


def test_generate_dataset_distinct_trajectories():
    recs = generate_dataset("click", n_trajectories=4, steps=4, seed=1, h=8, w=8)
    assert len(recs) == 4
    # Different seeds -> at least some different starting target positions.
    firsts = {tuple(tuple(r) for r in rec["steps"][0]["grid"]) for rec in recs}
    assert len(firsts) >= 2

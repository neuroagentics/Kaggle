"""Tiny controlled-world environments with KNOWN transition functions.

Purpose (trustworthy-internal-game milestone, Step 1): before spending GPU-hours
on the real ARC-3 games, prove the simulator can actually LEARN a world at all.
Each world here isolates ONE mechanic with an exactly-known transition rule, so a
failure to fit is diagnosable — it points at representation / loss / transition
alignment, NOT at "the game is just hard".

Mechanics implemented (one per world):
  - movement:  an agent cell moves one step in the commanded direction.
  - collision: same as movement, but a wall blocks the move (state unchanged).
  - click:     clicking a target cell (ACTION6 with x,y) toggles/activates it.
  - delayed:   pressing a button now changes a distant cell N steps LATER
               (delayed consequence — the effect is not co-located in time).

Design constraints (match the real pipeline so nothing is faked):
  - Grids are small (default 12x12), values 0..15, exactly like ARC-3 frames.
  - Trajectories are emitted as the SAME per-step dict schema the JSONL collector
    produces (grid / action_id / action_x / action_y / available_actions / level /
    state), so training/dataset.py consumes them with zero special-casing.
  - Actions use the ARC-3 convention: ids 0..5 are directional/simple controls,
    id 6 is a click carrying (x, y). We map:
        1 = up, 2 = down, 3 = left, 4 = right, 5 = no-op, 6 = click(x,y).
    (0 is reset/unused in-episode.)

These worlds are for measurement, not for scoring. They are never submitted and
never mixed into the real-game splits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

# ARC-3 action ids we use inside controlled worlds.
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_NOOP, A_CLICK = 1, 2, 3, 4, 5, 6
_MOVE_ACTIONS = (A_UP, A_DOWN, A_LEFT, A_RIGHT)
_MOVE_DELTA = {
    A_UP: (-1, 0),
    A_DOWN: (1, 0),
    A_LEFT: (0, -1),
    A_RIGHT: (0, 1),
}

# Color palette (0..15). Distinct values so changed cells are unambiguous.
BG = 0
AGENT = 3
WALL = 5
TARGET = 8
TARGET_ON = 9
BUTTON = 12
DOOR_CLOSED = 14
DOOR_OPEN = 15


Grid = list[list[int]]


def _blank(h: int, w: int) -> Grid:
    return [[BG] * w for _ in range(h)]


def _clone(grid: Grid) -> Grid:
    return [row[:] for row in grid]


@dataclass
class World:
    """A controlled world: holds current grid + hidden state, applies a known rule."""

    name: str
    h: int
    w: int
    grid: Grid
    available: tuple[int, ...]
    # hidden bookkeeping used by some mechanics (e.g. delayed timers)
    extra: dict = field(default_factory=dict)
    _step: Callable[["World", int, int | None, int | None], None] = None  # type: ignore

    def level(self) -> int:
        return int(self.extra.get("level", 0))

    def state(self) -> str:
        return str(self.extra.get("state", "NOT_FINISHED"))

    def step(self, action_id: int, x: int | None = None, y: int | None = None) -> None:
        self._step(self, action_id, x, y)


# ---------------------------------------------------------------------------
# Mechanic 1: MOVEMENT — agent cell moves one step; no walls.
# ---------------------------------------------------------------------------

def _movement_step(world: World, action_id: int, x, y) -> None:
    if action_id not in _MOVE_ACTIONS:
        return  # no-op / click: nothing moves
    (ar, ac) = world.extra["agent"]
    dr, dc = _MOVE_DELTA[action_id]
    nr, nc = ar + dr, ac + dc
    # Clamp to bounds (edge acts as a soft wall so the agent never leaves frame).
    nr = max(0, min(world.h - 1, nr))
    nc = max(0, min(world.w - 1, nc))
    world.grid[ar][ac] = BG
    world.grid[nr][nc] = AGENT
    world.extra["agent"] = (nr, nc)


def make_movement_world(rng: random.Random, h: int = 12, w: int = 12) -> World:
    grid = _blank(h, w)
    ar, ac = rng.randrange(h), rng.randrange(w)
    grid[ar][ac] = AGENT
    return World(
        name="movement", h=h, w=w, grid=grid,
        available=_MOVE_ACTIONS,
        extra={"agent": (ar, ac)}, _step=_movement_step,
    )


# ---------------------------------------------------------------------------
# Mechanic 2: COLLISION — like movement, but a wall blocks the move entirely.
# ---------------------------------------------------------------------------

def _collision_step(world: World, action_id: int, x, y) -> None:
    if action_id not in _MOVE_ACTIONS:
        return
    (ar, ac) = world.extra["agent"]
    dr, dc = _MOVE_DELTA[action_id]
    nr, nc = ar + dr, ac + dc
    if not (0 <= nr < world.h and 0 <= nc < world.w):
        return  # edge blocks
    if world.grid[nr][nc] == WALL:
        return  # wall blocks: state completely unchanged (the key learnable rule)
    world.grid[ar][ac] = BG
    world.grid[nr][nc] = AGENT
    world.extra["agent"] = (nr, nc)


def make_collision_world(rng: random.Random, h: int = 12, w: int = 12) -> World:
    grid = _blank(h, w)
    # A vertical wall down the middle with a single gap.
    wall_c = w // 2
    gap_r = rng.randrange(h)
    for r in range(h):
        if r != gap_r:
            grid[r][wall_c] = WALL
    # Agent starts on the left of the wall.
    ar = rng.randrange(h)
    ac = rng.randrange(0, wall_c)
    grid[ar][ac] = AGENT
    return World(
        name="collision", h=h, w=w, grid=grid,
        available=_MOVE_ACTIONS,
        extra={"agent": (ar, ac)}, _step=_collision_step,
    )


# ---------------------------------------------------------------------------
# Mechanic 3: CLICK — clicking the target cell activates it (TARGET -> TARGET_ON).
# ---------------------------------------------------------------------------

def _click_step(world: World, action_id: int, x, y) -> None:
    if action_id != A_CLICK or x is None or y is None:
        return
    # (x, y) are column, row in ARC convention; clamp defensively.
    c, r = int(x), int(y)
    if not (0 <= r < world.h and 0 <= c < world.w):
        return
    if world.grid[r][c] == TARGET:
        world.grid[r][c] = TARGET_ON  # only the clicked target changes


def make_click_world(rng: random.Random, h: int = 12, w: int = 12) -> World:
    grid = _blank(h, w)
    tr, tc = rng.randrange(h), rng.randrange(w)
    grid[tr][tc] = TARGET
    return World(
        name="click", h=h, w=w, grid=grid,
        available=(A_CLICK,),
        extra={"target": (tr, tc)}, _step=_click_step,
    )


# ---------------------------------------------------------------------------
# Mechanic 4: DELAYED CONSEQUENCE — pressing the button opens a distant door
# exactly `delay` steps later (not on the same step). Tests temporal credit.
# ---------------------------------------------------------------------------

def _delayed_step(world: World, action_id: int, x, y) -> None:
    # First, resolve any pending timer from a PREVIOUS press.
    timer = world.extra.get("timer")
    if timer is not None:
        world.extra["timer"] = timer - 1
        if world.extra["timer"] <= 0:
            dr, dc = world.extra["door"]
            world.grid[dr][dc] = DOOR_OPEN
            world.extra["timer"] = None

    # Then, a press on the button (via up-action while standing on it, or click)
    # arms the timer. We use A_CLICK on the button cell to arm it.
    if action_id == A_CLICK and x is not None and y is not None:
        c, r = int(x), int(y)
        br, bc = world.extra["button"]
        if (r, c) == (br, bc) and world.extra.get("timer") is None \
                and world.grid[world.extra["door"][0]][world.extra["door"][1]] == DOOR_CLOSED:
            world.extra["timer"] = int(world.extra["delay"])


def make_delayed_world(rng: random.Random, h: int = 12, w: int = 12, delay: int = 2) -> World:
    grid = _blank(h, w)
    br, bc = rng.randrange(h), rng.randrange(w // 2)          # button on left
    dr, dc = rng.randrange(h), rng.randrange(w // 2, w)       # door on right
    if (br, bc) == (dr, dc):
        dc = min(w - 1, dc + 1)
    grid[br][bc] = BUTTON
    grid[dr][dc] = DOOR_CLOSED
    return World(
        name="delayed", h=h, w=w, grid=grid,
        available=(A_CLICK,),
        extra={"button": (br, bc), "door": (dr, dc), "delay": delay, "timer": None},
        _step=_delayed_step,
    )


WORLD_FACTORIES: dict[str, Callable[..., World]] = {
    "movement": make_movement_world,
    "collision": make_collision_world,
    "click": make_click_world,
    "delayed": make_delayed_world,
}


# ---------------------------------------------------------------------------
# Trajectory generation — emits the SAME step-dict schema as the JSONL collector.
# ---------------------------------------------------------------------------

def _policy_action(world: World, rng: random.Random) -> tuple[int, int | None, int | None]:
    """Pick an action that actually exercises the mechanic (not blind random).

    - movement/collision: uniformly random legal move.
    - click: click the target cell most of the time; sometimes a random cell to
      teach the model that OFF-target clicks change nothing (false-change control).
    - delayed: click the button, then keep clicking it/random so the delayed
      effect appears at a later step than the trigger.
    """
    if world.name in ("movement", "collision"):
        return (rng.choice(_MOVE_ACTIONS), None, None)
    if world.name == "click":
        tr, tc = world.extra["target"]
        if rng.random() < 0.6 and world.grid[tr][tc] == TARGET:
            return (A_CLICK, tc, tr)  # (x=col, y=row)
        return (A_CLICK, rng.randrange(world.w), rng.randrange(world.h))
    if world.name == "delayed":
        br, bc = world.extra["button"]
        if world.extra.get("timer") is None and rng.random() < 0.7:
            return (A_CLICK, bc, br)  # press the button
        return (A_CLICK, rng.randrange(world.w), rng.randrange(world.h))
    return (A_NOOP, None, None)


def generate_trajectory(
    world_name: str,
    *,
    steps: int,
    seed: int,
    h: int = 12,
    w: int = 12,
    **factory_kwargs,
) -> dict:
    """Generate one trajectory record for `world_name` in JSONL-collector schema.

    Returns a dict: {game_id, steps: [ {grid, action_id, action_x, action_y,
    available_actions, level, state}, ... ]}. The final step has action_id None
    (terminal target), matching the collector convention.
    """
    rng = random.Random(seed)
    world = WORLD_FACTORIES[world_name](rng, h=h, w=w, **factory_kwargs)

    step_records: list[dict] = []
    for _ in range(steps):
        aid, ax, ay = _policy_action(world, rng)
        step_records.append({
            "grid": _clone(world.grid),
            "action_id": int(aid),
            "action_x": int(ax) if ax is not None else None,
            "action_y": int(ay) if ay is not None else None,
            "available_actions": list(world.available),
            "level": world.level(),
            "state": world.state(),
        })
        world.step(aid, ax, ay)

    # Terminal observation (no action) — the last transition's target.
    step_records.append({
        "grid": _clone(world.grid),
        "action_id": None,
        "action_x": None,
        "action_y": None,
        "available_actions": list(world.available),
        "level": world.level(),
        "state": world.state(),
    })
    return {"game_id": f"cw_{world_name}", "steps": step_records}


def generate_dataset(
    world_name: str,
    *,
    n_trajectories: int,
    steps: int,
    seed: int = 0,
    h: int = 12,
    w: int = 12,
    **factory_kwargs,
) -> list[dict]:
    """Generate a list of trajectory records (distinct seeds per trajectory)."""
    return [
        generate_trajectory(
            world_name, steps=steps, seed=seed * 100003 + i,
            h=h, w=w, **factory_kwargs,
        )
        for i in range(n_trajectories)
    ]

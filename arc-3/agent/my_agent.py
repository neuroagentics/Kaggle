"""Lean ARC-AGI-3 effect-model and graph explorer.

The agent learns only from transitions in the current game. It masks stable
engine UI bands, snaps coordinate actions to visible components, estimates
which action/color/region combinations produce novel states, and replays known
edges to return to unexplored frontiers. Confirmed movement actions receive a
small goal-directed bonus toward rare objects.

The frozen-mask/effect-model/frontier pattern is adapted from Shlok Sah's
MIT-0 FrugalExplorer (commit 7dccef15650fa00ef1db1c21cc7279adcf4a9e2a).
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict, deque
from time import perf_counter
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


Grid = list[list[int]]
ActionKey = tuple[int, int | None, int | None]


def frame_delta(before: Grid, after: Grid) -> dict[str, Any]:
    """Return a compact, JSON-safe description of an observed frame change."""
    before_shape = (len(before), len(before[0]) if before else 0)
    after_shape = (len(after), len(after[0]) if after else 0)
    if before_shape != after_shape:
        return {
            "class": "resize",
            "changed_cells": None,
            "bbox": None,
            "before_shape": list(before_shape),
            "after_shape": list(after_shape),
        }

    changed = [
        (x, y)
        for y, row in enumerate(after)
        for x, value in enumerate(row)
        if value != before[y][x]
    ]
    if not changed:
        delta_class = "unchanged"
        bbox = None
    else:
        xs = [x for x, _ in changed]
        ys = [y for _, y in changed]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        total = max(1, after_shape[0] * after_shape[1])
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        if len(changed) <= max(4, math.ceil(total * 0.02)):
            delta_class = "sparse"
        elif bbox_area <= total * 0.25:
            delta_class = "local"
        else:
            delta_class = "global"
    return {
        "class": delta_class,
        "changed_cells": len(changed),
        "bbox": bbox,
        "before_shape": list(before_shape),
        "after_shape": list(after_shape),
    }


def frame_grid(frame: FrameData) -> Grid:
    if not frame.frame:
        return []
    # An action may return a complete animation stack. The last image is the
    # settled board on which the next action will operate; the first image can
    # still describe an in-flight state from the previous transition.
    return [[int(value) for value in row] for row in frame.frame[-1]]


def _masked_grid(grid: Grid) -> Grid:
    """Mask engine UI bands without changing the observation used for actions."""
    if not grid or not grid[0]:
        return []
    height, width = len(grid), len(grid[0])
    masked = [row[:] for row in grid]
    if height < 8 or width < 8:
        return masked
    sentinel = 16
    for y in range(height):
        for x in range(width):
            if x in (0, width - 1) or y in (0, 1, height - 1):
                masked[y][x] = sentinel
    return masked


def state_key(frame: FrameData) -> str:
    grid = _masked_grid(frame_grid(frame))
    height = len(grid)
    width = len(grid[0]) if height else 0
    payload = bytearray(
        (frame.levels_completed & 0xFF, height & 0xFF, width & 0xFF)
    )
    for row in grid:
        payload.extend(value & 0xFF for value in row)
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _background(grid: Grid) -> int:
    if not grid or not grid[0]:
        return 0
    height, width = len(grid), len(grid[0])
    edge = grid[0] + grid[-1]
    edge += [grid[y][0] for y in range(1, height - 1)]
    edge += [grid[y][width - 1] for y in range(1, height - 1)]
    return Counter(edge).most_common(1)[0][0]


def _components(grid: Grid) -> list[tuple[int, int, int, int, int, int]]:
    """Return (area, color, cx, cy, min_x, min_y) per color component."""
    if not grid or not grid[0]:
        return []
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    x_lo, x_hi = (0, width) if width < 8 else (1, width - 1)
    y_lo, y_hi = (0, height) if height < 8 else (2, height - 1)
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int, int, int, int, int]] = []
    for y in range(y_lo, y_hi):
        for x in range(x_lo, x_hi):
            color = grid[y][x]
            if color == background or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in (
                    (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)
                ):
                    if (
                        x_lo <= nx < x_hi
                        and y_lo <= ny < y_hi
                        and (nx, ny) not in seen
                        and grid[ny][nx] == color
                    ):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            centre_x = round(sum(x0 for x0, _ in cells) / len(cells))
            centre_y = round(sum(y0 for _, y0 in cells) / len(cells))
            result.append(
                (
                    len(cells), color, centre_x, centre_y,
                    min(x0 for x0, _ in cells), min(y0 for _, y0 in cells),
                )
            )
    return sorted(result, key=lambda item: (item[0], item[1], item[3], item[2]))


def coordinate_proposals(grid: Grid, previous: Grid | None = None) -> list[tuple[int, int]]:
    """Prioritize changed regions, rare components, and real colored pixels."""
    if not grid or not grid[0]:
        return [(32, 32)]
    height, width = len(grid), len(grid[0])
    x_lo, x_hi = (0, width) if width < 8 else (1, width - 1)
    y_lo, y_hi = (0, height) if height < 8 else (2, height - 1)
    proposals: list[tuple[int, int]] = []
    if previous and len(previous) == height and len(previous[0]) == width:
        changed = [
            (x, y)
            for y in range(y_lo, y_hi)
            for x in range(x_lo, x_hi)
            if previous[y][x] != grid[y][x]
        ]
        if changed:
            proposals.append(
                (
                    round(sum(x for x, _ in changed) / len(changed)),
                    round(sum(y for _, y in changed) / len(changed)),
                )
            )

    for area, _, cx, cy, min_x, min_y in _components(grid):
        proposals.append((cx, cy))
        if area > 3:
            proposals.append((min_x, min_y))

    background = _background(grid)
    for y0 in range(y_lo, y_hi, 8):
        for x0 in range(x_lo, x_hi, 8):
            found = next(
                (
                    (x, y)
                    for y in range(y0, min(y_hi, y0 + 8))
                    for x in range(x0, min(x_hi, x0 + 8))
                    if grid[y][x] != background
                ),
                None,
            )
            if found:
                proposals.append(found)

    result: list[tuple[int, int]] = []
    for x, y in proposals:
        point = (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
        if point not in result:
            result.append(point)
    return result[:64]


class Node:
    def __init__(self) -> None:
        self.candidates: list[ActionKey] = []
        self.tested: dict[ActionKey, bool] = {}
        self.edges: dict[ActionKey, str] = {}


class MyAgent(Agent):
    """Training-free, scoring-aware task-local world-model explorer."""

    MAX_ACTIONS = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int.from_bytes(
            hashlib.blake2b(self.game_id.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        self._rng = random.Random(seed)
        self._last_state: str | None = None
        self._last_grid: Grid | None = None
        self._last_level = 0
        self._last_action: ActionKey | None = None
        self._nodes: dict[str, Node] = {}
        self._state_visits: Counter[str] = Counter()
        self._states_seen: set[str] = set()
        self._plan: deque[tuple[str, ActionKey]] = deque()
        self._no_ops: set[tuple[str, ActionKey]] = set()
        self._deadly: set[tuple[str, ActionKey]] = set()
        self._action_stats: dict[int, list[float]] = defaultdict(lambda: [1.0, 2.0])
        self._color_stats: dict[int, list[float]] = defaultdict(lambda: [1.0, 2.0])
        self._region_stats: dict[tuple[int, int], list[float]] = defaultdict(
            lambda: [1.0, 2.0]
        )
        self._progress_actions: Counter[int] = Counter()
        self._move_votes: Counter[tuple[int, int, int, int]] = Counter()
        self._moves: dict[int, tuple[int, int]] = {}
        self._avatar_color: int | None = None
        self._visited_targets: set[tuple[int, int, int]] = set()
        self._transition_events: deque[dict[str, Any]] = deque(maxlen=512)
        self._telemetry: Counter[str] = Counter()
        self._delta_classes: Counter[str] = Counter()
        self._selection_reasons: Counter[str] = Counter()
        self._decision_latency_ms_total = 0.0
        self._decision_latency_ms_max = 0.0

    @property
    def name(self) -> str:
        return f"{super().name}.settled-world-model-v2.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def telemetry_summary(self) -> dict[str, Any]:
        """Summarize task-local behavior without writing from the hot path."""
        actions = self._telemetry["actions_selected"]
        transitions = self._telemetry["transitions_observed"]
        levels = self._telemetry["levels_gained"]
        return {
            "schema_version": 1,
            "game_id": self.game_id,
            "actions_selected": actions,
            "transitions_observed": transitions,
            "unique_states": len(self._states_seen),
            "current_level_unique_states": len(self._state_visits),
            "changed_transitions": self._telemetry["changed_transitions"],
            "novel_transitions": self._telemetry["novel_transitions"],
            "no_ops": self._telemetry["no_ops"],
            "no_op_rate": (
                self._telemetry["no_ops"] / transitions if transitions else 0.0
            ),
            "deadly_transitions": self._telemetry["deadly_transitions"],
            "levels_gained": levels,
            "actions_per_level": actions / levels if levels else None,
            "replans": self._telemetry["replans"],
            "plan_invalidations": self._telemetry["plan_invalidations"],
            "movement_model": {
                str(action): [delta[0], delta[1]]
                for action, delta in sorted(self._moves.items())
            },
            "avatar_color": self._avatar_color,
            "delta_classes": dict(sorted(self._delta_classes.items())),
            "selection_reasons": dict(sorted(self._selection_reasons.items())),
            "decision_latency_ms": {
                "mean": self._decision_latency_ms_total / actions if actions else 0.0,
                "max": self._decision_latency_ms_max,
            },
        }

    def transition_events(self) -> list[dict[str, Any]]:
        """Return the bounded transition trace for local evaluation tooling."""
        return list(self._transition_events)

    @staticmethod
    def _centroid(grid: Grid, color: int) -> tuple[float, float, int] | None:
        cells = [
            (x, y)
            for y in range(2, len(grid) - 1)
            for x in range(1, len(grid[0]) - 1)
            if grid[y][x] == color
        ]
        if not cells:
            return None
        return (
            sum(x for x, _ in cells) / len(cells),
            sum(y for _, y in cells) / len(cells),
            len(cells),
        )

    def _learn_move(self, action: ActionKey, before: Grid, after: Grid) -> None:
        if action[1] is not None or not before or not after:
            return
        action_id = action[0]
        colors = set(value for row in before for value in row)
        colors &= set(value for row in after for value in row)
        colors.discard(_background(before))
        colors.discard(0)
        for color in colors:
            old = self._centroid(before, color)
            new = self._centroid(after, color)
            if old is None or new is None or not (1 <= old[2] <= 40):
                continue
            if abs(new[2] - old[2]) > old[2] // 2 + 2:
                continue
            dx = round(new[0] - old[0])
            dy = round(new[1] - old[1])
            if (dx, dy) == (0, 0) or abs(dx) > 10 or abs(dy) > 10:
                continue
            vote = (action_id, dx, dy, color)
            self._move_votes[vote] += 1
            if self._move_votes[vote] >= 2:
                self._moves[action_id] = (dx, dy)
                self._avatar_color = color

    def _effect_stats(self, action: ActionKey, grid: Grid) -> list[list[float]]:
        stats = [self._action_stats[action[0]]]
        if action[1] is not None and action[2] is not None:
            x, y = action[1], action[2]
            stats.append(self._color_stats[grid[y][x]])
            stats.append(self._region_stats[(y // 8, x // 8)])
        return stats

    def _observe(self, current: str, grid: Grid, level: int, state: GameState) -> None:
        novel = current not in self._state_visits
        if self._last_state is not None and self._last_action is not None:
            edge = (self._last_state, self._last_action)
            changed = current != self._last_state or level != self._last_level
            level_delta = max(0, level - self._last_level)
            delta = frame_delta(
                _masked_grid(self._last_grid or []), _masked_grid(grid)
            )
            self._telemetry["transitions_observed"] += 1
            self._telemetry["changed_transitions"] += int(changed)
            self._telemetry["novel_transitions"] += int(changed and novel)
            self._telemetry["no_ops"] += int(not changed)
            self._telemetry["deadly_transitions"] += int(
                state is GameState.GAME_OVER
            )
            self._telemetry["levels_gained"] += level_delta
            self._delta_classes[delta["class"]] += 1
            self._transition_events.append(
                {
                    "step": self._telemetry["transitions_observed"],
                    "from_state": self._last_state,
                    "action": {
                        "id": self._last_action[0],
                        "x": self._last_action[1],
                        "y": self._last_action[2],
                    },
                    "to_state": current,
                    "delta": delta,
                    "novel": bool(changed and novel),
                    "level_delta": level_delta,
                    "status": state.name,
                }
            )
            previous_node = self._nodes.get(self._last_state)
            if previous_node is not None:
                previous_node.tested[self._last_action] = changed
                previous_node.edges[self._last_action] = current
            if not changed:
                self._no_ops.add(edge)
            useful = changed and novel
            for pair in self._effect_stats(self._last_action, self._last_grid or grid):
                pair[0] += 1.0 if useful else 0.0
                pair[1] += 1.0
            if self._last_grid is not None:
                self._learn_move(self._last_action, self._last_grid, grid)
            if level > self._last_level:
                self._progress_actions[self._last_action[0]] += level - self._last_level
                self._nodes.clear()
                self._state_visits.clear()
                self._plan.clear()
                self._no_ops.clear()
                self._deadly.clear()
                self._visited_targets.clear()
            elif state is GameState.GAME_OVER:
                self._deadly.add(edge)
                self._plan.clear()
        self._state_visits[current] += 1
        self._states_seen.add(current)

    def _available_ids(self, frame: FrameData) -> list[int]:
        available = [int(value) for value in frame.available_actions]
        if not available:
            available = [
                action.value for action in GameAction if action is not GameAction.RESET
            ]
        return [value for value in available if value != GameAction.RESET.value]

    def _candidate_keys(self, frame: FrameData, grid: Grid) -> list[ActionKey]:
        candidates: list[ActionKey] = []
        for action_id in self._available_ids(frame):
            if action_id == GameAction.ACTION6.value:
                candidates.extend(
                    (action_id, x, y)
                    for x, y in coordinate_proposals(grid, self._last_grid)
                )
            else:
                candidates.append((action_id, None, None))
        return candidates

    def _movement_bonus(self, candidate: ActionKey, grid: Grid) -> float:
        delta = self._moves.get(candidate[0])
        if delta is None or self._avatar_color is None:
            return 0.0
        avatar = self._centroid(grid, self._avatar_color)
        if avatar is None:
            return 0.0
        ax, ay, _ = avatar
        targets = [
            (area, color, x, y)
            for area, color, x, y, _, _ in _components(grid)
            if color != self._avatar_color
            and (color, x // 4, y // 4) not in self._visited_targets
        ]
        if not targets:
            return 0.0
        _, color, tx, ty = min(
            targets, key=lambda item: (item[0], abs(item[2] - ax) + abs(item[3] - ay))
        )
        if abs(tx - ax) + abs(ty - ay) <= 3:
            return 0.0
        dx, dy = delta
        before = abs(tx - ax) + abs(ty - ay)
        after = abs(tx - (ax + dx)) + abs(ty - (ay + dy))
        return 3.0 * (before - after)

    def _score(self, current: str, candidate: ActionKey, grid: Grid) -> float:
        edge = (current, candidate)
        if edge in self._deadly:
            return -10_000.0
        if edge in self._no_ops:
            return -1_000.0
        node = self._nodes[current]
        exploration = 12.0 if candidate not in node.tested else 1.0
        score = exploration + 8.0 * self._progress_actions[candidate[0]]
        for changes, tries in self._effect_stats(candidate, grid):
            score += changes / tries + 1.5 / math.sqrt(tries)
        score += self._movement_bonus(candidate, grid)
        return score + self._rng.random() * 0.001

    def _plan_to_frontier(self, start: str) -> bool:
        parents: dict[str, tuple[str, ActionKey] | None] = {start: None}
        queue = deque([start])
        target: str | None = None
        while queue:
            state = queue.popleft()
            node = self._nodes.get(state)
            if node is None:
                continue
            if state != start and any(key not in node.tested for key in node.candidates):
                target = state
                break
            for action, outcome in node.edges.items():
                if outcome not in parents and (state, action) not in self._deadly:
                    parents[outcome] = (state, action)
                    queue.append(outcome)
        if target is None:
            return False
        path: list[tuple[str, ActionKey]] = []
        cursor = target
        while parents[cursor] is not None:
            previous, action = parents[cursor]
            path.append((previous, action))
            cursor = previous
        self._plan = deque(reversed(path))
        self._telemetry["replans"] += 1
        return True

    def _materialize(self, key: ActionKey, why: str) -> GameAction:
        action = GameAction.from_id(key[0])
        if action is GameAction.ACTION6:
            action.set_data({"x": int(key[1] or 0), "y": int(key[2] or 0)})
        action.reasoning = {
            "controller": "settled-world-model-v2",
            "why": why,
            "known_states": len(self._nodes),
            "movement_model": dict(self._moves),
        }
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        started = perf_counter()
        grid = frame_grid(latest_frame)
        current = state_key(latest_frame)
        self._observe(current, grid, latest_frame.levels_completed, latest_frame.state)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            selected: ActionKey = (GameAction.RESET.value, None, None)
            action = self._materialize(selected, "start or recover the level")
        else:
            node = self._nodes.setdefault(current, Node())
            if not node.candidates:
                node.candidates = self._candidate_keys(latest_frame, grid)

            selected = None
            reason = "select the highest-value untested local transition"
            if self._plan:
                expected, planned = self._plan[0]
                if expected == current:
                    self._plan.popleft()
                    selected = planned
                    reason = "replay a known edge toward an unexplored frontier"
                else:
                    self._plan.clear()
                    self._telemetry["plan_invalidations"] += 1

            untested = [
                key
                for key in node.candidates
                if key not in node.tested and (current, key) not in self._deadly
            ]
            if selected is None and untested:
                selected = max(untested, key=lambda key: self._score(current, key, grid))
            if selected is None and self._plan_to_frontier(current):
                _, selected = self._plan.popleft()
                reason = "return to a known state with untested actions"
            if selected is None:
                viable = [
                    key for key in node.candidates if (current, key) not in self._deadly
                ]
                selected = max(
                    viable or node.candidates,
                    key=lambda key: self._score(current, key, grid),
                )
                reason = "reuse the most promising observed transition"
            action = self._materialize(selected, reason)

        elapsed_ms = (perf_counter() - started) * 1000.0
        self._telemetry["actions_selected"] += 1
        self._selection_reasons[action.reasoning["why"]] += 1
        self._decision_latency_ms_total += elapsed_ms
        self._decision_latency_ms_max = max(self._decision_latency_ms_max, elapsed_ms)
        action.reasoning["telemetry"] = self.telemetry_summary()
        self._last_state = current
        self._last_grid = grid
        self._last_level = latest_frame.levels_completed
        self._last_action = selected
        return action

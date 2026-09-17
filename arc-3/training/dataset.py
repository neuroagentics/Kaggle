"""JSONL trajectory dataset — feeds collect_trajectories.py output into training.

Reads the collector's JSONL and yields TrajectoryBatch objects matching the
contract in train_simulator.py. Handles:
  - Fixed trajectory window T via truncation / padding (padding masked out).
  - Grid padding to a common (H, W) with a validity mask (never treats pad as bg).
  - Label derivation from observed signals only (no fabricated labels):
      progress_labels : 1.0 on a step whose action produced a level increase, else 0.
      failure_labels  : 1.0 if the resulting state is GAME_OVER, else 0.
      avail_labels    : per-action 1/0 from the observed available_actions list.
  - Whole-game / mechanic-family splitting is the caller's responsibility (via
    game_id filtering); this loader never mixes a held-out game into training.

torch is imported lazily so the module imports without torch for schema checks.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Sequence

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

NUM_ACTIONS = 8
PAD_TOKEN = 0  # value written into padded cells (masked out; never used as bg)


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for the trajectory dataset.")


def load_records(path: Path, *, games: Sequence[str] | None = None) -> list[dict]:
    """Load JSONL trajectory records, optionally filtered to specific game_ids."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if games is not None and rec.get("game_id") not in games:
                continue
            records.append(rec)
    return records


def _pad_grid(grid: list[list[int]], H: int, W: int) -> tuple[list[list[int]], list[list[bool]]]:
    """Pad a grid to (H, W); return (padded_grid, validity_mask)."""
    gh = len(grid)
    gw = len(grid[0]) if gh else 0
    out = [[PAD_TOKEN] * W for _ in range(H)]
    mask = [[False] * W for _ in range(H)]
    for y in range(min(gh, H)):
        for x in range(min(gw, W)):
            out[y][x] = int(grid[y][x]) & 0xF  # clamp to 0..15
            mask[y][x] = True
    return out, mask


def record_to_windows(
    rec: dict, *, window: int
) -> list[list[dict]]:
    """Split a trajectory's steps into fixed-length windows of `window` steps.

    Each window is a list of step dicts (len == window after padding). Only steps
    with a real action (action_id is not None) can be the *source* of a
    transition; the terminal step (action_id None) is kept as the final target.
    """
    steps = rec["steps"]
    if len(steps) < 2:
        return []
    windows = []
    # Non-overlapping windows keep transitions independent across the batch.
    for start in range(0, len(steps) - 1, window):
        chunk = steps[start:start + window + 1]  # +1 so last step is a target
        if len(chunk) >= 2:
            windows.append(chunk)
    return windows


def build_batch(
    windows: list[list[dict]],
    *,
    game_ids: list[str],
    max_hw: int = 64,
) -> "TrajectoryBatch":
    """Assemble a list of step-windows into a padded TrajectoryBatch."""
    _require_torch()
    from training.train_simulator import TrajectoryBatch

    B = len(windows)
    T = max(len(w) for w in windows)
    # Common H, W = max over all grids in the batch (bounded by max_hw)
    H = W = 0
    for w in windows:
        for s in w:
            g = s["grid"]
            H = max(H, min(len(g), max_hw))
            W = max(W, min(len(g[0]) if g else 0, max_hw))
    H = max(1, H)
    W = max(1, W)

    grid_tokens = torch.zeros(B, T, H, W, dtype=torch.long)
    validity = torch.zeros(B, T, H, W, dtype=torch.bool)
    action_ids = torch.zeros(B, T, dtype=torch.long)
    action_x = torch.full((B, T), -1.0, dtype=torch.float32)
    action_y = torch.full((B, T), -1.0, dtype=torch.float32)
    coord_valid = torch.zeros(B, T, dtype=torch.float32)
    progress = torch.full((B, T), -1.0, dtype=torch.float32)
    failure = torch.full((B, T), -1.0, dtype=torch.float32)
    avail = torch.full((B, T, NUM_ACTIONS), -1.0, dtype=torch.float32)
    level_ids = torch.zeros(B, T, dtype=torch.long)

    for b, w in enumerate(windows):
        for t, step in enumerate(w):
            pg, pm = _pad_grid(step["grid"], H, W)
            grid_tokens[b, t] = torch.tensor(pg, dtype=torch.long)
            validity[b, t] = torch.tensor(pm, dtype=torch.bool)
            level_ids[b, t] = int(step.get("level", 0))

            aid = step.get("action_id")
            if aid is not None:
                action_ids[b, t] = int(aid)
                if aid == 6 and step.get("action_x") is not None:
                    action_x[b, t] = float(step["action_x"]) / 63.0
                    action_y[b, t] = float(step["action_y"]) / 63.0
                    coord_valid[b, t] = 1.0

            # Availability labels from observed available_actions.
            av = step.get("available_actions")
            if av is not None:
                row = [0.0] * NUM_ACTIONS
                for a in av:
                    if 0 <= int(a) < NUM_ACTIONS:
                        row[int(a)] = 1.0
                avail[b, t] = torch.tensor(row, dtype=torch.float32)

            # Progress / failure derived from the NEXT step's observation.
            if t + 1 < len(w):
                nxt = w[t + 1]
                nxt_level = int(nxt.get("level", step.get("level", 0)))
                cur_level = int(step.get("level", 0))
                progress[b, t + 1] = 1.0 if nxt_level > cur_level else 0.0
                st = str(nxt.get("state", "")).upper()
                failure[b, t + 1] = 1.0 if st == "GAME_OVER" else 0.0

    return TrajectoryBatch(
        grid_tokens=grid_tokens,
        validity_mask=validity,
        action_ids=action_ids,
        action_x=action_x,
        action_y=action_y,
        coord_valid=coord_valid,
        progress_labels=progress,
        failure_labels=failure,
        avail_labels=avail,
        game_ids=game_ids,
        level_ids=level_ids,
    )


class JsonlTrajectoryDataset:
    """Iterable dataset yielding TrajectoryBatch objects from a JSONL file.

    Split-aware: pass `games` to restrict to a set of game_ids (train vs holdout).
    Never mixes games outside the provided set.
    """

    def __init__(
        self,
        path: Path,
        *,
        window: int = 8,
        batch_size: int = 8,
        games: Sequence[str] | None = None,
        max_hw: int = 64,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.path = Path(path)
        self.window = window
        self.batch_size = batch_size
        self.games = list(games) if games is not None else None
        self.max_hw = max_hw
        self.shuffle = shuffle
        self._rng = random.Random(seed)

        records = load_records(self.path, games=self.games)
        # Flatten into (game_id, window) pairs.
        self._items: list[tuple[str, list[dict]]] = []
        for rec in records:
            gid = rec.get("game_id", "unknown")
            for w in record_to_windows(rec, window=self.window):
                self._items.append((gid, w))

    def num_windows(self) -> int:
        return len(self._items)

    def game_ids(self) -> set[str]:
        return {gid for gid, _ in self._items}

    def split(self) -> str:
        return "custom"

    def num_games(self) -> int:
        return len(self.game_ids())

    def __len__(self) -> int:
        return (len(self._items) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator["TrajectoryBatch"]:
        _require_torch()
        items = list(self._items)
        if self.shuffle:
            self._rng.shuffle(items)
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            if not chunk:
                continue
            gids = [g for g, _ in chunk]
            windows = [w for _, w in chunk]
            yield build_batch(windows, game_ids=gids, max_hw=self.max_hw)


class InMemoryTrajectoryDataset:
    """TrajectoryDataset over in-memory records (no JSONL file).

    Same contract and windowing as JsonlTrajectoryDataset, but takes already-loaded
    trajectory records (e.g. from training.controlled_worlds.generate_dataset).
    Used by the controlled-world learning harness so the real trainer consumes
    controlled worlds with zero special-casing.
    """

    def __init__(
        self,
        records: Sequence[dict],
        *,
        window: int = 8,
        batch_size: int = 8,
        games: Sequence[str] | None = None,
        max_hw: int = 64,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.window = window
        self.batch_size = batch_size
        self.games = list(games) if games is not None else None
        self.max_hw = max_hw
        self.shuffle = shuffle
        self._rng = random.Random(seed)

        self._items: list[tuple[str, list[dict]]] = []
        for rec in records:
            gid = rec.get("game_id", "unknown")
            if self.games is not None and gid not in self.games:
                continue
            for w in record_to_windows(rec, window=self.window):
                self._items.append((gid, w))

    def num_windows(self) -> int:
        return len(self._items)

    def game_ids(self) -> set[str]:
        return {gid for gid, _ in self._items}

    def split(self) -> str:
        return "custom"

    def num_games(self) -> int:
        return len(self.game_ids())

    def __len__(self) -> int:
        return (len(self._items) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator["TrajectoryBatch"]:
        _require_torch()
        items = list(self._items)
        if self.shuffle:
            self._rng.shuffle(items)
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            if not chunk:
                continue
            gids = [g for g, _ in chunk]
            windows = [w for _, w in chunk]
            yield build_batch(windows, game_ids=gids, max_hw=self.max_hw)

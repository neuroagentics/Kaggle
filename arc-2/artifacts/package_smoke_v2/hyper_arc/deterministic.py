"""Small, exact-replay transformation channel for ARC-AGI-2.

This module deliberately covers only low-description-length transformations.
It does not score partial matches: a hypothesis is returned only when it
replays every training output exactly.  That makes it a safe, fast ensemble
channel and a useful measured baseline for the recursive solver.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable


Grid = list[list[int]]
Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class ExactCandidate:
    name: str
    complexity: int
    transform: Transform

    def predict(self, grid: Grid) -> Grid:
        return self.transform(grid)


def _copy(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _rectangular(grid: Grid) -> bool:
    return bool(grid and grid[0] and all(len(row) == len(grid[0]) for row in grid))


def _rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def _rotate180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def _rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def _reflect_horizontal(grid: Grid) -> Grid:
    return [row[:] for row in grid[::-1]]


def _reflect_vertical(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def _anti_transpose(grid: Grid) -> Grid:
    return [list(row)[::-1] for row in zip(*grid[::-1])]


GEOMETRIES: tuple[tuple[str, Transform, int], ...] = (
    ("identity", _copy, 0),
    ("rotate90", _rotate90, 1),
    ("rotate180", _rotate180, 1),
    ("rotate270", _rotate270, 1),
    ("reflect_horizontal", _reflect_horizontal, 1),
    ("reflect_vertical", _reflect_vertical, 1),
    ("transpose", _transpose, 1),
    ("anti_transpose", _anti_transpose, 1),
)


def _crop(grid: Grid, background: int) -> Grid:
    cells = [
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value != background
    ]
    if not cells:
        return _copy(grid)
    r0 = min(row for row, _ in cells)
    r1 = max(row for row, _ in cells)
    c0 = min(col for _, col in cells)
    c1 = max(col for _, col in cells)
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


def _crop_mode(grid: Grid) -> Grid:
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    return _crop(grid, background)


def _upscale(grid: Grid, factor: int) -> Grid:
    result: Grid = []
    for row in grid:
        expanded = [value for value in row for _ in range(factor)]
        result.extend(expanded[:] for _ in range(factor))
    return result


def _tile(grid: Grid, repeats_h: int, repeats_w: int) -> Grid:
    wide = [row * repeats_w for row in grid]
    return [row[:] for _ in range(repeats_h) for row in wide]


def _compose(*operations: Transform) -> Transform:
    def apply(grid: Grid) -> Grid:
        result = _copy(grid)
        for operation in operations:
            result = operation(result)
        return result

    return apply


def _base_transforms() -> Iterable[tuple[str, Transform, int]]:
    prefixes: tuple[tuple[str, Transform, int], ...] = (
        ("", _copy, 0),
        ("crop0+", lambda grid: _crop(grid, 0), 1),
        ("crop_mode+", _crop_mode, 1),
    )
    for prefix_name, prefix, prefix_cost in prefixes:
        for geometry_name, geometry, geometry_cost in GEOMETRIES:
            stem = _compose(prefix, geometry)
            name = f"{prefix_name}{geometry_name}"
            cost = prefix_cost + geometry_cost
            yield name, stem, cost
            for factor in (2, 3, 4):
                yield (
                    f"{name}+upscale{factor}",
                    _compose(stem, lambda grid, k=factor: _upscale(grid, k)),
                    cost + 1,
                )
            for repeats_h in range(1, 5):
                for repeats_w in range(1, 5):
                    if repeats_h == repeats_w == 1:
                        continue
                    yield (
                        f"{name}+tile{repeats_h}x{repeats_w}",
                        _compose(
                            stem,
                            lambda grid, rh=repeats_h, rw=repeats_w: _tile(
                                grid, rh, rw
                            ),
                        ),
                        cost + 2,
                    )


def _infer_color_map(
    transformed: list[Grid], outputs: list[Grid]
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for source, target in zip(transformed, outputs):
        if not _rectangular(source) or not _rectangular(target):
            return None
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            return None
        for source_row, target_row in zip(source, target):
            for before, after in zip(source_row, target_row):
                known = mapping.get(before)
                if known is not None and known != after:
                    return None
                mapping[before] = after
    return mapping


def _with_color_map(transform: Transform, mapping: dict[int, int]) -> Transform:
    def apply(grid: Grid) -> Grid:
        return [
            [mapping.get(value, value) for value in row]
            for row in transform(grid)
        ]

    return apply


def exact_candidates(
    train_pairs: list[tuple[Grid, Grid]],
    test_inputs: list[Grid] | None = None,
) -> list[ExactCandidate]:
    """Return simple hypotheses that exactly replay all demonstrations.

    Candidates are behaviorally deduplicated over both demonstrations and the
    supplied test inputs, then ordered by minimum description length.
    """
    if not train_pairs or any(
        not _rectangular(source) or not _rectangular(target)
        for source, target in train_pairs
    ):
        return []

    inputs = [source for source, _ in train_pairs]
    outputs = [target for _, target in train_pairs]
    probes = inputs + list(test_inputs or [])
    candidates: list[ExactCandidate] = []

    for name, base, base_cost in _base_transforms():
        transformed = [base(grid) for grid in inputs]
        mapping = _infer_color_map(transformed, outputs)
        if mapping is None:
            continue
        changed = {source: target for source, target in mapping.items() if source != target}
        transform = _with_color_map(base, mapping)
        if any(transform(source) != target for source, target in train_pairs):
            continue
        suffix = "" if not changed else "+remap"
        candidates.append(
            ExactCandidate(
                name=f"{name}{suffix}",
                complexity=base_cost + len(changed),
                transform=transform,
            )
        )

    candidates.sort(key=lambda item: (item.complexity, len(item.name), item.name))
    unique: list[ExactCandidate] = []
    seen: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for candidate in candidates:
        behavior = tuple(
            tuple(tuple(row) for row in candidate.predict(grid)) for grid in probes
        )
        if behavior not in seen:
            seen.add(behavior)
            unique.append(candidate)
    return unique


def prediction_pair(candidates: list[ExactCandidate], grid: Grid) -> tuple[Grid, Grid]:
    """Return the two highest-ranked distinct predictions for one test grid."""
    predictions: list[Grid] = []
    for candidate in candidates:
        prediction = candidate.predict(grid)
        if prediction not in predictions:
            predictions.append(prediction)
        if len(predictions) == 2:
            break
    if not predictions:
        return _copy(grid), _copy(grid)
    if len(predictions) == 1:
        fallback = _copy(grid)
        predictions.append(fallback)
    return predictions[0], predictions[1]

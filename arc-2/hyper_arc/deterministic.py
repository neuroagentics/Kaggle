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


def _foreground_components(grid: Grid, *, same_color: bool) -> list[set[tuple[int, int]]]:
    if not _rectangular(grid):
        return []
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    pending = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value != background
    }
    components: list[set[tuple[int, int]]] = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor not in pending:
                    continue
                if same_color and grid[neighbor[0]][neighbor[1]] != grid[row][col]:
                    continue
                pending.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(component)
    return components


def _crop_cells(grid: Grid, cells: set[tuple[int, int]], *, masked: bool) -> Grid:
    if not cells:
        return []
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    top = min(row for row, _ in cells)
    bottom = max(row for row, _ in cells)
    left = min(col for _, col in cells)
    right = max(col for _, col in cells)
    return [
        [
            grid[row][col] if not masked or (row, col) in cells else background
            for col in range(left, right + 1)
        ]
        for row in range(top, bottom + 1)
    ]


def _crop_component(
    grid: Grid, *, largest: bool, same_color: bool, masked: bool
) -> Grid:
    components = _foreground_components(grid, same_color=same_color)
    if not components:
        return []
    target_size = (max if largest else min)(len(component) for component in components)
    selected = [component for component in components if len(component) == target_size]
    if len(selected) != 1:
        return []
    return _crop_cells(grid, selected[0], masked=masked)


def _crop_color_rank(grid: Grid, *, most_common: bool, masked: bool) -> Grid:
    if not _rectangular(grid):
        return []
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    counts = Counter(value for row in grid for value in row if value != background)
    if not counts:
        return []
    target_count = (max if most_common else min)(counts.values())
    selected_colors = [color for color, count in counts.items() if count == target_count]
    if len(selected_colors) != 1:
        return []
    color = selected_colors[0]
    cells = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == color
    }
    return _crop_cells(grid, cells, masked=masked)


def _upscale(grid: Grid, factor: int) -> Grid:
    return _upscale_rect(grid, factor, factor)


def _upscale_rect(grid: Grid, factor_h: int, factor_w: int) -> Grid:
    result: Grid = []
    for row in grid:
        expanded = [value for value in row for _ in range(factor_w)]
        result.extend(expanded[:] for _ in range(factor_h))
    return result


def _downscale_uniform(grid: Grid, factor_h: int, factor_w: int) -> Grid:
    if (
        not _rectangular(grid)
        or len(grid) % factor_h
        or len(grid[0]) % factor_w
    ):
        return []
    result: Grid = []
    for top in range(0, len(grid), factor_h):
        row = []
        for left in range(0, len(grid[0]), factor_w):
            block = {
                grid[r][c]
                for r in range(top, top + factor_h)
                for c in range(left, left + factor_w)
            }
            if len(block) != 1:
                return []
            row.append(next(iter(block)))
        result.append(row)
    return result


def _downscale_blocks(
    grid: Grid, factor_h: int, factor_w: int, selector: str
) -> Grid:
    """Reduce fixed-size blocks by a unique mode or minority color."""
    if (
        not _rectangular(grid)
        or len(grid) % factor_h
        or len(grid[0]) % factor_w
    ):
        return []
    result: Grid = []
    for top in range(0, len(grid), factor_h):
        row = []
        for left in range(0, len(grid[0]), factor_w):
            counts = Counter(
                grid[r][c]
                for r in range(top, top + factor_h)
                for c in range(left, left + factor_w)
            )
            target_count = (
                max(counts.values()) if selector == "mode" else min(counts.values())
            )
            selected = [color for color, count in counts.items() if count == target_count]
            if len(selected) != 1:
                return []
            row.append(selected[0])
        result.append(row)
    return result


def _tile(grid: Grid, repeats_h: int, repeats_w: int) -> Grid:
    wide = [row * repeats_w for row in grid]
    return [row[:] for _ in range(repeats_h) for row in wide]


def _hcat(left: Grid, right: Grid) -> Grid:
    if not _rectangular(left) or not _rectangular(right) or len(left) != len(right):
        return []
    return [left_row + right_row for left_row, right_row in zip(left, right)]


def _vcat(top: Grid, bottom: Grid) -> Grid:
    if (
        not _rectangular(top)
        or not _rectangular(bottom)
        or len(top[0]) != len(bottom[0])
    ):
        return []
    return _copy(top) + _copy(bottom)


def _mirror_quad(grid: Grid, origin: str) -> Grid:
    variants = {
        "top-left": (
            grid,
            _reflect_vertical(grid),
            _reflect_horizontal(grid),
            _rotate180(grid),
        ),
        "top-right": (
            _reflect_vertical(grid),
            grid,
            _rotate180(grid),
            _reflect_horizontal(grid),
        ),
        "bottom-left": (
            _reflect_horizontal(grid),
            _rotate180(grid),
            grid,
            _reflect_vertical(grid),
        ),
        "bottom-right": (
            _rotate180(grid),
            _reflect_horizontal(grid),
            _reflect_vertical(grid),
            grid,
        ),
    }
    top_left, top_right, bottom_left, bottom_right = variants[origin]
    return _vcat(_hcat(top_left, top_right), _hcat(bottom_left, bottom_right))


def _compress_identical_lines(grid: Grid) -> Grid:
    if not _rectangular(grid):
        return []
    rows = [row[:] for index, row in enumerate(grid) if index == 0 or row != grid[index - 1]]
    columns = [
        col
        for col in range(len(rows[0]))
        if col == 0 or any(row[col] != row[col - 1] for row in rows)
    ]
    return [[row[col] for col in columns] for row in rows]


def _split_on_background(grid: Grid, orientation: str) -> tuple[Grid, Grid] | None:
    if not _rectangular(grid):
        return None
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    if orientation == "horizontal":
        separators = [index for index, row in enumerate(grid) if all(v == background for v in row)]
        chunks = []
        start = 0
        for separator in separators + [len(grid)]:
            if separator > start:
                chunks.append([row[:] for row in grid[start:separator]])
            start = separator + 1
    else:
        separators = [
            col
            for col in range(len(grid[0]))
            if all(row[col] == background for row in grid)
        ]
        bounds = []
        start = 0
        for separator in separators + [len(grid[0])]:
            if separator > start:
                bounds.append((start, separator))
            start = separator + 1
        chunks = [[row[left:right] for row in grid] for left, right in bounds]
    if len(chunks) != 2:
        return None
    first, second = chunks
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        return None
    return first, second


def _split_panels(grid: Grid) -> list[Grid]:
    """Remove full background separators and return equal rectangular panels."""
    if not _rectangular(grid):
        return []
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    row_separators = {
        row for row, values in enumerate(grid) if all(value == background for value in values)
    }
    col_separators = {
        col
        for col in range(len(grid[0]))
        if all(row[col] == background for row in grid)
    }

    def spans(size: int, separators: set[int]) -> list[tuple[int, int]]:
        values = []
        start = 0
        for separator in sorted(separators) + [size]:
            if separator > start:
                values.append((start, separator))
            start = separator + 1
        return values

    row_spans = spans(len(grid), row_separators)
    col_spans = spans(len(grid[0]), col_separators)
    if len(row_spans) * len(col_spans) not in {2, 3, 4, 6, 9}:
        return []
    panels = [
        [row[left:right] for row in grid[top:bottom]]
        for top, bottom in row_spans
        for left, right in col_spans
    ]
    shapes = {(len(panel), len(panel[0])) for panel in panels}
    return panels if len(shapes) == 1 else []


def _panel_at(grid: Grid, index: int) -> Grid:
    panels = _split_panels(grid)
    if index >= len(panels):
        return []
    return _copy(panels[index])


def _color_statistic_cell(grid: Grid, statistic: str) -> Grid:
    """Select one uniquely distinguished non-background color as a 1x1 grid."""
    if not _rectangular(grid):
        return []
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    colors = sorted({value for row in grid for value in row if value != background})
    if not colors:
        return []
    counts = Counter(value for row in grid for value in row if value != background)
    components = _foreground_components(grid, same_color=True)
    component_counts = Counter(grid[next(iter(component))[0]][next(iter(component))[1]] for component in components)
    largest_components = {
        color: max(
            len(component)
            for component in components
            if grid[next(iter(component))[0]][next(iter(component))[1]] == color
        )
        for color in colors
    }
    values = {
        "least-cells": {color: counts[color] for color in colors},
        "most-cells": {color: -counts[color] for color in colors},
        "least-components": {color: component_counts[color] for color in colors},
        "most-components": {color: -component_counts[color] for color in colors},
        "smallest-largest-component": largest_components,
        "largest-component": {color: -largest_components[color] for color in colors},
    }[statistic]
    best = min(values.values())
    selected = [color for color, value in values.items() if value == best]
    return [[selected[0]]] if len(selected) == 1 else []


def _panel_boolean(grid: Grid, orientation: str, operation: str) -> Grid:
    panels = _split_on_background(grid, orientation)
    if panels is None:
        return []
    first, second = panels
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    result: Grid = []
    for row in range(len(first)):
        values = []
        for col in range(len(first[0])):
            a = first[row][col] != background
            b = second[row][col] != background
            matched = {
                "or": a or b,
                "and": a and b,
                "xor": a != b,
                "equal": a == b,
                "first_only": a and not b,
                "second_only": b and not a,
            }[operation]
            values.append(1 if matched else 0)
        result.append(values)
    return result


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
            for factor_h in range(1, 5):
                for factor_w in range(1, 5):
                    if factor_h == factor_w:
                        continue
                    yield (
                        f"{name}+upscale{factor_h}x{factor_w}",
                        _compose(
                            stem,
                            lambda grid, fh=factor_h, fw=factor_w: _upscale_rect(
                                grid, fh, fw
                            ),
                        ),
                        cost + 2,
                    )

    for geometry_name, geometry, geometry_cost in GEOMETRIES:
        for factor_h in range(1, 6):
            for factor_w in range(1, 6):
                if factor_h == factor_w == 1:
                    continue
                yield (
                    f"downscale{factor_h}x{factor_w}+{geometry_name}",
                    _compose(
                        lambda grid, fh=factor_h, fw=factor_w: _downscale_uniform(
                            grid, fh, fw
                        ),
                        geometry,
                    ),
                    geometry_cost + 2,
                )
        yield (
            f"compress-lines+{geometry_name}",
            _compose(_compress_identical_lines, geometry),
            geometry_cost + 2,
        )
        for selector in ("mode", "minority"):
            for factor_h in range(2, 11):
                for factor_w in range(2, 11):
                    yield (
                        f"block-{selector}{factor_h}x{factor_w}+{geometry_name}",
                        _compose(
                            lambda grid, fh=factor_h, fw=factor_w, choice=selector: _downscale_blocks(
                                grid, fh, fw, choice
                            ),
                            geometry,
                        ),
                        geometry_cost + 3,
                    )

    for panel_index in range(9):
        for geometry_name, geometry, geometry_cost in GEOMETRIES:
            yield (
                f"panel{panel_index}+{geometry_name}",
                _compose(
                    lambda grid, index=panel_index: _panel_at(grid, index), geometry
                ),
                geometry_cost + 2,
            )

    for statistic in (
        "least-cells",
        "most-cells",
        "least-components",
        "most-components",
        "smallest-largest-component",
        "largest-component",
    ):
        yield (
            f"select-color-{statistic}",
            lambda grid, name=statistic: _color_statistic_cell(grid, name),
            3,
        )

    for origin in ("top-left", "top-right", "bottom-left", "bottom-right"):
        yield (
            f"mirror-quad-{origin}",
            lambda grid, corner=origin: _mirror_quad(grid, corner),
            4,
        )

    identity = GEOMETRIES[0][1]
    for geometry_name, geometry, geometry_cost in GEOMETRIES:
        if geometry_name == "identity":
            continue
        yield (
            f"hcat-identity-{geometry_name}",
            lambda grid, op=geometry: _hcat(identity(grid), op(grid)),
            geometry_cost + 2,
        )
        yield (
            f"hcat-{geometry_name}-identity",
            lambda grid, op=geometry: _hcat(op(grid), identity(grid)),
            geometry_cost + 2,
        )
        yield (
            f"vcat-identity-{geometry_name}",
            lambda grid, op=geometry: _vcat(identity(grid), op(grid)),
            geometry_cost + 2,
        )
        yield (
            f"vcat-{geometry_name}-identity",
            lambda grid, op=geometry: _vcat(op(grid), identity(grid)),
            geometry_cost + 2,
        )

    for orientation in ("horizontal", "vertical"):
        for operation in ("or", "and", "xor", "equal", "first_only", "second_only"):
            yield (
                f"panel-{orientation}-{operation}",
                lambda grid, axis=orientation, op=operation: _panel_boolean(grid, axis, op),
                3,
            )

    extraction_transforms: list[tuple[str, Transform, int]] = []
    for same_color, component_kind in ((False, "foreground"), (True, "monochrome")):
        for largest, size_name in ((True, "largest"), (False, "smallest")):
            for masked in (False, True):
                mask_name = "masked" if masked else "raw"
                extraction_transforms.append(
                    (
                        f"crop-{size_name}-{component_kind}-{mask_name}",
                        lambda grid, big=largest, mono=same_color, mask=masked: _crop_component(
                        grid, largest=big, same_color=mono, masked=mask
                        ),
                        2 if not masked else 3,
                    )
                )
    for most_common, rank_name in ((True, "most-common"), (False, "rarest")):
        for masked in (False, True):
            mask_name = "masked" if masked else "raw"
            extraction_transforms.append(
                (
                    f"crop-{rank_name}-color-{mask_name}",
                    lambda grid, common=most_common, mask=masked: _crop_color_rank(
                        grid, most_common=common, masked=mask
                    ),
                    2 if not masked else 3,
                )
            )
    for extraction_name, extraction, extraction_cost in extraction_transforms:
        for geometry_name, geometry, geometry_cost in GEOMETRIES:
            stem = _compose(extraction, geometry)
            yield (
                f"{extraction_name}+{geometry_name}",
                stem,
                extraction_cost + geometry_cost,
            )
            for factor in (2, 3):
                yield (
                    f"{extraction_name}+{geometry_name}+upscale{factor}",
                    _compose(stem, lambda grid, k=factor: _upscale(grid, k)),
                    extraction_cost + geometry_cost + 1,
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

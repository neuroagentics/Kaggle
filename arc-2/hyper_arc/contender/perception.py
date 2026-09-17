"""Canonical pixel, object, relation, and demonstration-delta perception."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from typing import Any, Iterable, Mapping

from hyper_arc.contender.schemas import (
    ExampleDelta,
    Grid,
    GridPair,
    GridState,
    ObjectCorrespondence,
    ObjectState,
    Relation,
    TaskState,
)


class PerceptionError(ValueError):
    """Raised when an ARC grid cannot be represented canonically."""


def canonical_grid(grid: Iterable[Iterable[int]]) -> Grid:
    rows = tuple(tuple(row) for row in grid)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise PerceptionError("Grid must be non-empty and rectangular")
    if len(rows) > 30 or len(rows[0]) > 30:
        raise PerceptionError("ARC grids cannot exceed 30x30")
    if any(
        not isinstance(cell, int)
        or isinstance(cell, bool)
        or not 0 <= cell <= 9
        for row in rows
        for cell in row
    ):
        raise PerceptionError("ARC grid cells must be integer colors 0-9")
    return rows


def _background_candidates(grid: Grid) -> tuple[int, ...]:
    counts = Counter(cell for row in grid for cell in row)
    border = Counter(grid[0] + grid[-1])
    for row in grid[1:-1]:
        border[row[0]] += 1
        border[row[-1]] += 1
    def mask_signature(color: int) -> str:
        mask = tuple(tuple(cell == color for cell in row) for row in grid)
        return hashlib.sha256(
            json.dumps(mask, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    ranked = sorted(
        counts,
        key=lambda color: (
            -counts[color],
            -border[color],
            mask_signature(color),
            color,
        ),
    )
    best = ranked[0]
    best_score = counts[best], border[best]
    tied = [color for color in ranked if (counts[color], border[color]) == best_score]
    return tuple(tied or [best])


def _neighbors(
    row: int, col: int, height: int, width: int, connectivity: int
) -> Iterable[tuple[int, int]]:
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for delta_row, delta_col in steps:
        candidate = row + delta_row, col + delta_col
        if 0 <= candidate[0] < height and 0 <= candidate[1] < width:
            yield candidate


def _components(grid: Grid, background: int, connectivity: int) -> list[dict[str, Any]]:
    if connectivity not in (4, 8):
        raise PerceptionError("Connectivity must be 4 or 8")
    height, width = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    found: list[dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == background or (row, col) in visited:
                continue
            queue = deque([(row, col)])
            visited.add((row, col))
            pixels: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                pixels.append(current)
                for candidate in _neighbors(*current, height, width, connectivity):
                    if candidate not in visited and grid[candidate[0]][candidate[1]] == color:
                        visited.add(candidate)
                        queue.append(candidate)
            top = min(point[0] for point in pixels)
            bottom = max(point[0] for point in pixels)
            left = min(point[1] for point in pixels)
            right = max(point[1] for point in pixels)
            relative = tuple(sorted((r - top, c - left) for r, c in pixels))
            shape_payload = json.dumps(relative, separators=(",", ":")).encode("utf-8")
            found.append(
                {
                    "color": color,
                    "pixels": tuple(sorted(pixels)),
                    "bbox": (top, left, bottom, right),
                    "relative": relative,
                    "shape_signature": hashlib.sha256(shape_payload).hexdigest(),
                    "holes": _count_holes(relative, bottom - top + 1, right - left + 1),
                }
            )
    return sorted(
        found,
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
            item["shape_signature"],
            len(item["pixels"]),
        ),
    )


def _count_holes(
    relative_pixels: tuple[tuple[int, int], ...], height: int, width: int
) -> int:
    occupied = set(relative_pixels)
    empty = {(row, col) for row in range(height) for col in range(width)} - occupied
    outside: set[tuple[int, int]] = set()
    queue = deque(
        point
        for point in empty
        if point[0] in (0, height - 1) or point[1] in (0, width - 1)
    )
    outside.update(queue)
    while queue:
        point = queue.popleft()
        for neighbor in _neighbors(*point, height, width, 4):
            if neighbor in empty and neighbor not in outside:
                outside.add(neighbor)
                queue.append(neighbor)
    holes = 0
    remaining = empty - outside
    while remaining:
        holes += 1
        queue = deque([next(iter(remaining))])
        remaining.remove(queue[0])
        while queue:
            point = queue.popleft()
            for neighbor in _neighbors(*point, height, width, 4):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return holes


def _object_relations(objects: tuple[ObjectState, ...], grid_ref: str) -> tuple[Relation, ...]:
    """Build a sparse graph instead of materializing quadratic equivalence edges.

    Shape, color, and area equivalence are already explicit object attributes.
    Relations retain only adjacency in spatial order, pixel touching, and the
    smallest containing box. This bounds ordinary relation count linearly.
    """
    keys: set[tuple[str, str, str]] = set()

    def add(kind: str, source: str, target: str) -> None:
        keys.add((kind, source, target))

    horizontal = sorted(
        objects,
        key=lambda item: (
            item.bounding_box[1],
            item.bounding_box[3],
            item.bounding_box[0],
            item.object_id,
        ),
    )
    for first, second in zip(horizontal, horizontal[1:]):
        if first.bounding_box[3] < second.bounding_box[1]:
            add("left_of", first.object_id, second.object_id)

    vertical = sorted(
        objects,
        key=lambda item: (
            item.bounding_box[0],
            item.bounding_box[2],
            item.bounding_box[1],
            item.object_id,
        ),
    )
    for first, second in zip(vertical, vertical[1:]):
        if first.bounding_box[2] < second.bounding_box[0]:
            add("above", first.object_id, second.object_id)

    occupancy = {
        pixel: item.object_id for item in objects for pixel in item.pixels
    }
    for (row, col), source in occupancy.items():
        for neighbor in ((row + 1, col), (row, col + 1)):
            target = occupancy.get(neighbor)
            if target is not None and target != source:
                first, second = sorted((source, target))
                add("touching", first, second)

    for inner in objects:
        containers = [
            outer
            for outer in objects
            if outer.object_id != inner.object_id
            and _contains(outer.bounding_box, inner.bounding_box)
        ]
        if containers:
            immediate = min(
                containers,
                key=lambda item: (
                    (item.bounding_box[2] - item.bounding_box[0] + 1)
                    * (item.bounding_box[3] - item.bounding_box[1] + 1),
                    item.area,
                    item.object_id,
                ),
            )
            add("bbox_contains", immediate.object_id, inner.object_id)

    return tuple(
        Relation(kind, source, target, grid_ref)
        for kind, source, target in sorted(keys)
    )


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer != inner
        and outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _symmetries(grid: Grid) -> tuple[str, ...]:
    horizontal = tuple(reversed(grid))
    vertical = tuple(tuple(reversed(row)) for row in grid)
    rotated = tuple(tuple(reversed(row)) for row in reversed(grid))
    values = []
    if grid == horizontal:
        values.append("horizontal")
    if grid == vertical:
        values.append("vertical")
    if grid == rotated:
        values.append("rotational_180")
    return tuple(values)


def perceive_grid(
    grid: Iterable[Iterable[int]], grid_ref: str, connectivity: int = 4
) -> GridState:
    canonical = canonical_grid(grid)
    candidates = _background_candidates(canonical)
    background = candidates[0]
    components = _components(canonical, background, connectivity)
    color_roles: dict[int, str] = {}
    objects: list[ObjectState] = []
    for index, component in enumerate(components):
        color = component["color"]
        if color not in color_roles:
            color_roles[color] = f"c{len(color_roles)}"
        objects.append(
            ObjectState(
                object_id=f"{grid_ref}:o{index}",
                pixels=component["pixels"],
                colors=(color,),
                bounding_box=component["bbox"],
                connectivity=connectivity,
                area=len(component["pixels"]),
                holes=component["holes"],
                grid_ref=grid_ref,
                relative_pixels=component["relative"],
                shape_signature=component["shape_signature"],
                color_role=color_roles[color],
                attributes=_object_attributes(component),
            )
        )
    object_tuple = tuple(objects)
    relations = _object_relations(object_tuple, grid_ref)
    signature_payload = {
        "dimensions": [len(canonical), len(canonical[0])],
        "objects": [
            {
                "id": item.object_id.rsplit(":", 1)[-1],
                "shape": item.shape_signature,
                "area": item.area,
                "holes": item.holes,
                "color_role": item.color_role,
            }
            for item in object_tuple
        ],
        "relations": [
            [
                relation.kind,
                relation.source_id.rsplit(":", 1)[-1],
                relation.target_id.rsplit(":", 1)[-1],
            ]
            for relation in relations
        ],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return GridState(
        grid_ref=grid_ref,
        grid=canonical,
        height=len(canonical),
        width=len(canonical[0]),
        palette=tuple(sorted({cell for row in canonical for cell in row})),
        background_candidates=candidates,
        objects=object_tuple,
        relations=relations,
        symmetries=_symmetries(canonical),
        periodicity=_periodicity(canonical),
        separator_rows=tuple(
            index for index, row in enumerate(canonical) if len(set(row)) == 1
        ),
        separator_columns=tuple(
            index
            for index in range(len(canonical[0]))
            if len({row[index] for row in canonical}) == 1
        ),
        canonical_signature=signature,
    )


def _object_attributes(component: Mapping[str, Any]) -> dict[str, Any]:
    top, left, bottom, right = component["bbox"]
    height = bottom - top + 1
    width = right - left + 1
    area = len(component["pixels"])
    relative = component["relative"]
    relative_set = set(relative)
    mask = tuple(
        tuple((row, col) in relative_set for col in range(width))
        for row in range(height)
    )
    return {
        "height": height,
        "width": width,
        "filled_bbox": area == height * width,
        "is_line": height == 1 or width == 1,
        "is_square": height == width and area == height * width,
        "shape_symmetries": _symmetries(mask),
    }


def _periodicity(grid: Grid) -> tuple[tuple[str, int], ...]:
    height, width = len(grid), len(grid[0])
    periods: list[tuple[str, int]] = []
    for period in range(1, height):
        if height % period == 0 and all(
            grid[index] == grid[index % period] for index in range(height)
        ):
            periods.append(("rows", period))
    columns = tuple(tuple(row[index] for row in grid) for index in range(width))
    for period in range(1, width):
        if width % period == 0 and all(
            columns[index] == columns[index % period] for index in range(width)
        ):
            periods.append(("columns", period))
    return tuple(periods)


def _example_delta(index: int, source: GridState, target: GridState) -> ExampleDelta:
    unused_targets = {item.object_id: item for item in target.objects}
    correspondences: list[ObjectCorrespondence] = []
    removed: list[str] = []
    for source_object in source.objects:
        matches = [
            item
            for item in unused_targets.values()
            if item.shape_signature == source_object.shape_signature
        ]
        if not matches:
            removed.append(source_object.object_id)
            continue
        target_object = min(
            matches,
            key=lambda item: (
                item.color_role != source_object.color_role,
                abs(item.bounding_box[0] - source_object.bounding_box[0])
                + abs(item.bounding_box[1] - source_object.bounding_box[1]),
                item.object_id,
            ),
        )
        unused_targets.pop(target_object.object_id)
        correspondences.append(
            ObjectCorrespondence(
                source_id=source_object.object_id,
                target_id=target_object.object_id,
                evidence="same_shape",
                translation=(
                    target_object.bounding_box[0] - source_object.bounding_box[0],
                    target_object.bounding_box[1] - source_object.bounding_box[1],
                ),
                color_changed=source_object.colors != target_object.colors,
            )
        )
    return ExampleDelta(
        example_index=index,
        input_ref=source.grid_ref,
        output_ref=target.grid_ref,
        correspondences=tuple(correspondences),
        added_object_ids=tuple(sorted(unused_targets)),
        removed_object_ids=tuple(sorted(removed)),
        input_shape=(source.height, source.width),
        output_shape=(target.height, target.width),
    )


def perceive_task(
    task_id: str,
    task_data: Mapping[str, Any],
    *,
    connectivity: int = 4,
    provenance: Mapping[str, Any] | None = None,
) -> TaskState:
    train_pairs: list[GridPair] = []
    grid_states: list[GridState] = []
    deltas: list[ExampleDelta] = []
    for index, pair in enumerate(task_data.get("train", [])):
        source = perceive_grid(pair["input"], f"train:{index}:input", connectivity)
        target = perceive_grid(pair["output"], f"train:{index}:output", connectivity)
        train_pairs.append(GridPair(source.grid, target.grid))
        grid_states.extend((source, target))
        deltas.append(_example_delta(index, source, target))
    test_inputs: list[Grid] = []
    for index, pair in enumerate(task_data.get("test", [])):
        state = perceive_grid(pair["input"], f"test:{index}:input", connectivity)
        grid_states.append(state)
        test_inputs.append(state.grid)
    if not train_pairs:
        raise PerceptionError("Task must contain at least one training pair")
    objects = tuple(item for state in grid_states for item in state.objects)
    relations = tuple(item for state in grid_states for item in state.relations)
    palette = tuple(sorted({color for state in grid_states for color in state.palette}))
    return TaskState(
        task_id=task_id,
        train=tuple(train_pairs),
        test_inputs=tuple(test_inputs),
        objects=objects,
        relations=relations,
        palette=palette,
        grids=tuple(grid_states),
        deltas=tuple(deltas),
        provenance=dict(provenance or {}),
    )

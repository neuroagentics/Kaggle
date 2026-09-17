"""Deterministic relational grid operations used by closed typed plans."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from hyper_arc.contender.schemas import Grid


@dataclass(frozen=True)
class Component:
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return len(self.cells)


def _mutable(grid: Grid | Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _background(grid: Sequence[Sequence[int]]) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _components(
    grid: Sequence[Sequence[int]], background: int
) -> tuple[Component, ...]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    result: list[Component] = []
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == background or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and (next_row, next_col) not in seen
                        and grid[next_row][next_col] == color
                    ):
                        seen.add((next_row, next_col))
                        queue.append((next_row, next_col))
            rows = [item[0] for item in cells]
            cols = [item[1] for item in cells]
            result.append(
                Component(
                    color=color,
                    cells=tuple(sorted(cells)),
                    bbox=(min(rows), min(cols), max(rows), max(cols)),
                )
            )
    return tuple(result)


def _stamp(
    output: list[list[int]],
    template: Sequence[Sequence[int]],
    *,
    top: int,
    left: int,
    background: int,
) -> None:
    for row, values in enumerate(template):
        for col, value in enumerate(values):
            target_row, target_col = top + row, left + col
            if (
                value != background
                and 0 <= target_row < len(output)
                and 0 <= target_col < len(output[0])
            ):
                output[target_row][target_col] = value


def edge_marker_motif_propagation(grid: Grid) -> Grid:
    """Propagate an existing motif from aligned edge markers toward the edge."""
    source = _mutable(grid)
    height, width = len(source), len(source[0])
    background = _background(source)
    components = _components(source, background)
    marker_colors = []
    for color in sorted({item.color for item in components}):
        colored = [item for item in components if item.color == color]
        if colored and all(
            (item.bbox[0] == item.bbox[2] or item.bbox[1] == item.bbox[3])
            and (
                item.bbox[0] == 0
                or item.bbox[2] == height - 1
                or item.bbox[1] == 0
                or item.bbox[3] == width - 1
            )
            for item in colored
        ):
            marker_colors.append(color)
    if len(marker_colors) != 1:
        raise ValueError("Expected one edge-marker color")
    marker_color = marker_colors[0]
    markers = [item for item in components if item.color == marker_color]
    motifs = [
        item
        for item in components
        if item.color != marker_color
        and item.area >= 3
        and item.bbox[0] != item.bbox[2]
        and item.bbox[1] != item.bbox[3]
    ]
    if not markers or not motifs:
        raise ValueError("Marker propagation requires markers and motifs")
    output = _mutable(grid)
    for marker in markers:
        for row, col in marker.cells:
            output[row][col] = background
        mr0, mc0, mr1, mc1 = marker.bbox
        horizontal = mr0 == mr1
        if horizontal and mr0 not in {0, height - 1}:
            raise ValueError("Horizontal marker must touch a horizontal edge")
        if not horizontal and mc0 not in {0, width - 1}:
            raise ValueError("Vertical marker must touch a vertical edge")

        def alignment_score(item: Component) -> tuple[int, int]:
            r0, c0, r1, c1 = item.bbox
            if horizontal:
                cross = abs(c0 - mc0) + abs((c1 - c0) - (mc1 - mc0))
                distance = abs(r0 - mr0)
            else:
                cross = abs(r0 - mr0) + abs((r1 - r0) - (mr1 - mr0))
                distance = abs(c0 - mc0)
            return cross, distance

        anchor = min(motifs, key=alignment_score)
        r0, c0, r1, c1 = anchor.bbox
        template = [row[c0 : c1 + 1] for row in source[r0 : r1 + 1]]
        tile_height, tile_width = len(template), len(template[0])
        if horizontal:
            step = -tile_height if mr0 == 0 else tile_height
            top, left = r0 + step, c0
            while top + tile_height > 0 and top < height:
                _stamp(output, template, top=top, left=left, background=background)
                top += step
        else:
            step = -tile_width if mc0 == 0 else tile_width
            top, left = r0, c0 + step
            while left + tile_width > 0 and left < width:
                _stamp(output, template, top=top, left=left, background=background)
                left += step
    return tuple(tuple(row) for row in output)


def keyed_object_frames(grid: Grid) -> Grid:
    """Use isolated two-color keys to frame larger objects by input color."""
    source = _mutable(grid)
    height, width = len(source), len(source[0])
    background = _background(source)
    mappings: dict[int, int] = {}
    for row in range(height - 1):
        for col in range(width - 1):
            first, second = source[row][col], source[row][col + 1]
            if (
                first != background
                and second != background
                and first != second
                and source[row + 1][col] == first
                and source[row + 1][col + 1] == second
            ):
                surrounding = [
                    source[r][c]
                    for r in range(max(0, row - 1), min(height, row + 3))
                    for c in range(max(0, col - 1), min(width, col + 3))
                    if not (row <= r <= row + 1 and col <= c <= col + 1)
                ]
                if surrounding and all(value == background for value in surrounding):
                    known = mappings.get(first)
                    if known is not None and known != second:
                        raise ValueError("Conflicting frame key")
                    mappings[first] = second
    if not mappings:
        raise ValueError("No isolated frame keys found")
    output = _mutable(grid)
    for component in _components(source, background):
        frame_color = mappings.get(component.color)
        if frame_color is None or component.area <= 2:
            continue
        r0, c0, r1, c1 = component.bbox
        top = max(0, r0 - 1)
        left = max(0, c0 - 1)
        bottom = min(height - 1, r1 + 1)
        right = min(width - 1, c1 + 1)
        if top >= bottom or left >= right:
            continue
        queue = deque()
        exterior: set[tuple[int, int]] = set()
        for col in range(left, right + 1):
            queue.extend(((top, col), (bottom, col)))
        for row in range(top, bottom + 1):
            queue.extend(((row, left), (row, right)))
        while queue:
            row, col = queue.popleft()
            if (
                (row, col) in exterior
                or not (top <= row <= bottom and left <= col <= right)
                or source[row][col] != background
            ):
                continue
            exterior.add((row, col))
            queue.extend(
                ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            )
        for row, col in exterior:
            output[row][col] = frame_color
    return tuple(tuple(row) for row in output)


def normalized_frame_signal(
    grid: Grid,
    motif_shape: tuple[int, int] | None = None,
) -> tuple[Grid, int]:
    """Normalize a rounded rectangular frame around its interior signal."""
    source = _mutable(grid)
    height, width = len(source), len(source[0])
    corners = (source[0][0], source[0][-1], source[-1][0], source[-1][-1])
    background = corners[0] if len(set(corners)) == 1 else _background(source)
    candidates = []
    for color in sorted({value for row in source for value in row} - {background}):
        border = (
            all(source[0][col] == color for col in range(1, width - 1))
            and all(source[height - 1][col] == color for col in range(1, width - 1))
            and all(source[row][0] == color for row in range(1, height - 1))
            and all(source[row][width - 1] == color for row in range(1, height - 1))
        )
        if border:
            candidates.append(color)
    if len(candidates) != 1:
        raise ValueError("Expected one rounded frame color")
    frame_color = candidates[0]
    signal = [
        (row, col, source[row][col])
        for row in range(1, height - 1)
        for col in range(1, width - 1)
        if source[row][col] != background
    ]
    if not signal:
        raise ValueError("Frame contains no signal")
    r0 = min(item[0] for item in signal)
    c0 = min(item[1] for item in signal)
    r1 = max(item[0] for item in signal)
    c1 = max(item[1] for item in signal)
    signal_height, signal_width = r1 - r0 + 1, c1 - c0 + 1
    motif_height, motif_width = motif_shape or (signal_height + 2, signal_width + 2)
    if motif_height < signal_height + 2 or motif_width < signal_width + 2:
        raise ValueError("Signal does not fit the requested motif shape")
    motif = [[background for _ in range(motif_width)] for _ in range(motif_height)]
    for col in range(1, motif_width - 1):
        motif[0][col] = frame_color
        motif[-1][col] = frame_color
    for row in range(1, motif_height - 1):
        motif[row][0] = frame_color
        motif[row][-1] = frame_color
    row_offset = 1 + (motif_height - 2 - signal_height) // 2
    col_offset = 1 + (motif_width - 2 - signal_width) // 2
    for row, col, value in signal:
        motif[row - r0 + row_offset][col - c0 + col_offset] = value
    return tuple(tuple(row) for row in motif), background


def frame_signal_repeat(grid: Grid, arguments: Mapping[str, Any]) -> Grid:
    """Place a normalized frame signal at learned output anchors."""
    motif, background = normalized_frame_signal(
        grid,
        (int(arguments["motif_height"]), int(arguments["motif_width"])),
    )
    height = int(arguments["height"])
    width = int(arguments["width"])
    positions = arguments["positions"]
    if not 1 <= height <= 30 or not 1 <= width <= 30:
        raise ValueError("Output dimensions must be within ARC limits")
    output = [[background for _ in range(width)] for _ in range(height)]
    for position in positions:
        if not isinstance(position, Sequence) or len(position) != 2:
            raise ValueError("Each position must contain row and column")
        _stamp(
            output,
            motif,
            top=int(position[0]),
            left=int(position[1]),
            background=background,
        )
    return tuple(tuple(row) for row in output)

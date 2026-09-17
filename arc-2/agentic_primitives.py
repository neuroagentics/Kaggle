"""Trusted pure-Python ARC primitives available to model-authored programs."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any


def copy_grid(grid):
    return [list(row) for row in grid]


def color_counts(grid):
    return dict(Counter(value for row in grid for value in row))


def background_color(grid):
    counts = color_counts(grid)
    return max(counts, key=counts.get)


def make_grid(
    height=None,
    width=None,
    color=0,
    rows=None,
    columns=None,
    cols=None,
    fill=None,
    fill_color=None,
):
    if height is None:
        height = rows
    if width is None:
        width = columns if columns is not None else cols
    if fill_color is not None:
        color = fill_color
    elif fill is not None:
        color = fill
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("make_grid requires integer height/width")
    return [[color for _ in range(width)] for _ in range(height)]


def transpose_grid(grid):
    return [list(row) for row in zip(*grid)]


def rotate90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid):
    return [row[::-1] for row in grid[::-1]]


def rotate270(grid):
    return [list(row) for row in zip(*grid)][::-1]


def reflect_horizontal(grid):
    return copy_grid(grid[::-1])


def reflect_vertical(grid):
    return [row[::-1] for row in grid]


def replace_color(grid, source, target):
    return [[target if value == source else value for value in row] for row in grid]


def crop_bbox(grid, bbox):
    top, left, bottom, right = bbox
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def scale_grid(grid, factor):
    output = []
    for row in grid:
        expanded = []
        for value in row:
            expanded.extend([value] * factor)
        for _ in range(factor):
            output.append(list(expanded))
    return output


def tile_grid(grid, repeats_h, repeats_w):
    row_block = [row * repeats_w for row in grid]
    return [list(row) for _ in range(repeats_h) for row in row_block]


def paint(grid, pixels, color):
    output = copy_grid(grid)
    for row, col in pixels:
        if 0 <= row < len(output) and 0 <= col < len(output[0]):
            output[row][col] = color
    return output


def components(grid, background=None):
    """Return 4-connected same-color non-background components as dictionaries."""
    if background is None:
        background = background_color(grid)
    height, width = len(grid), len(grid[0])
    seen = set()
    result: list[dict[str, Any]] = []
    for start_row in range(height):
        for start_col in range(width):
            color = grid[start_row][start_col]
            if color == background or (start_row, start_col) in seen:
                continue
            queue = deque([(start_row, start_col)])
            seen.add((start_row, start_col))
            pixels = []
            while queue:
                row, col = queue.popleft()
                pixels.append((row, col))
                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and (next_row, next_col) not in seen
                        and grid[next_row][next_col] == color
                    ):
                        seen.add((next_row, next_col))
                        queue.append((next_row, next_col))
            pixels.sort()
            rows = [row for row, _ in pixels]
            cols = [col for _, col in pixels]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            result.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "bbox": bbox,
                    "area": len(pixels),
                    "height": bbox[2] - bbox[0] + 1,
                    "width": bbox[3] - bbox[1] + 1,
                }
            )
    result.sort(key=lambda item: (item["area"], item["bbox"], item["color"]))
    return result


def component_count(grid, background=None):
    return len(components(grid, background))


def symmetries(grid):
    result = []
    if grid == reflect_horizontal(grid):
        result.append("horizontal")
    if grid == reflect_vertical(grid):
        result.append("vertical")
    if grid == rotate180(grid):
        result.append("rotational_180")
    if len(grid) == len(grid[0]) and grid == transpose_grid(grid):
        result.append("diagonal")
    return result


def split_panels(grid, background=None):
    """Split a grid on full background rows and columns."""
    if background is None:
        background = background_color(grid)
    separator_rows = [
        index for index, row in enumerate(grid) if all(value == background for value in row)
    ]
    separator_cols = [
        index
        for index in range(len(grid[0]))
        if all(row[index] == background for row in grid)
    ]

    def spans(size, separators):
        boundaries = [-1] + separators + [size]
        return [
            (boundaries[index] + 1, boundaries[index + 1])
            for index in range(len(boundaries) - 1)
            if boundaries[index] + 1 < boundaries[index + 1]
        ]

    row_spans = spans(len(grid), separator_rows)
    col_spans = spans(len(grid[0]), separator_cols)
    return [
        [row[left:right] for row in grid[top:bottom]]
        for top, bottom in row_spans
        for left, right in col_spans
    ]


def unique_panel(grid, background=None):
    panels = split_panels(grid, background)
    for panel in panels:
        if panels.count(panel) == 1:
            return copy_grid(panel)
    return None


def largest_component(grid, background=None):
    items = components(grid, background)
    return items[-1] if items else None


def smallest_component(grid, background=None):
    items = components(grid, background)
    return items[0] if items else None

"""Isolated execution worker for model-authored ARC transformation functions."""

from __future__ import annotations

import json
import sys
import builtins
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentic_sandbox import SAFE_CALLS, validate_code
from agentic_primitives import (
    background_color,
    color_counts,
    component_count,
    components,
    copy_grid,
    crop_bbox,
    largest_component,
    make_grid,
    paint,
    reflect_horizontal,
    reflect_vertical,
    replace_color,
    rotate90,
    rotate180,
    rotate270,
    scale_grid,
    smallest_component,
    split_panels,
    symmetries,
    tile_grid,
    transpose_grid,
    unique_panel,
)


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in SAFE_CALLS
    if hasattr(builtins, name)
}
SAFE_BUILTINS.update(
    {
        name: value
        for name, value in globals().items()
        if name in SAFE_CALLS and callable(value)
    }
)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    grids = payload["grids"]
    validate_code(code)
    namespace = {"__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<agentic-transform>", "exec"), namespace, namespace)
    transform = namespace["transform"]
    outputs = [transform([list(row) for row in grid]) for grid in grids]
    sys.stdout.write(json.dumps(outputs, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

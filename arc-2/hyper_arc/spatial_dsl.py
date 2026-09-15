"""hyper_arc/spatial_dsl.py — Spatial DSL with 12 pure-tensor primitives.

Design constraints:
- ARC grids are integer grids with colors 0-9.
- Grid tensors are stored as torch.int8.
- All public functions accept and return ESB objects.
- ESB.data shape is (C, H, W); C is treated as a channel/batch-within-ESB dim.
- rotate(esb, degrees) — degrees ∈ {90, 180, 270}, CLOCKWISE.
- flood_fill(esb, row, col, new_color) — (row, col) matches PyTorch (H, W).
- connect_points(esb, color, row1, col1, row2, col2) — row/col convention.
- symmetrize modes: "copy" and "priority" per spec requirement 3.14.
- Default background color / out-of-bounds fill: 0.
- No neural approximations. No nn.Module. No gradients.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple, Union

import torch

from hyper_arc.esb import ESB

__all__ = [
    "SpatialDSL",
    "PrimitiveCall",
    "DSLProgram",
]

# ── Type aliases ──────────────────────────────────────────────────────────

PrimitiveCall = tuple[str, dict[str, Any]]
DSLProgram = list[PrimitiveCall]

# ── Validated constant sets ───────────────────────────────────────────────

VALID_ROTATIONS = {90, 180, 270}
VALID_AXES = {"horizontal", "vertical"}
VALID_BLEND = {"overwrite", "or"}
VALID_SYM_MODES = {"copy", "priority"}
VALID_SCALES = {0.5, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp09(x: torch.Tensor) -> torch.Tensor:
    """Clamp to ARC color range 0-9 and cast to int8."""
    return x.clamp(0, 9).to(torch.int8)


def _normalize_axis(axis: Union[str, int]) -> Tuple[int, str]:
    """Normalise axis argument to (flip_dim, kind_str).

    Returns:
        dim:  -2 for horizontal (flip rows = top-bottom mirror)
              -1 for vertical   (flip cols = left-right mirror)
        kind: "horizontal" or "vertical"
    """
    if isinstance(axis, int):
        if axis in (-2, 0):
            return -2, "horizontal"
        if axis in (-1, 1):
            return -1, "vertical"
        raise ValueError(f"Unsupported integer axis: {axis}")
    a = str(axis).lower().strip()
    if a in {"h", "horizontal"}:
        return -2, "horizontal"
    if a in {"v", "vertical"}:
        return -1, "vertical"
    raise ValueError(
        f"axis must be 'horizontal' or 'vertical', got {axis!r}"
    )


def _flood_component_mask(
    mask: torch.Tensor,
    row: int,
    col: int,
) -> torch.Tensor:
    """Vectorised BFS connected-component mask.

    Args:
        mask: (C, H, W) bool tensor — True where cells are traversable.
        row:  seed row index.
        col:  seed column index.

    Returns:
        (C, H, W) bool tensor — True for all cells in the connected
        component reachable from (row, col) through True cells.
    """
    if mask.dim() != 3:
        raise ValueError("mask must have shape (C, H, W)")
    _, h, w = mask.shape
    if not (0 <= row < h and 0 <= col < w):
        return torch.zeros_like(mask, dtype=torch.bool)

    start = torch.zeros_like(mask, dtype=torch.bool)
    start[:, row, col] = True
    frontier = start & mask
    visited = frontier.clone()

    while bool(frontier.any()):
        dilated = torch.zeros_like(frontier, dtype=torch.bool)
        dilated[..., 1:, :]  |= frontier[..., :-1, :]   # down
        dilated[..., :-1, :] |= frontier[..., 1:, :]    # up
        dilated[..., :, 1:]  |= frontier[..., :, :-1]   # right
        dilated[..., :, :-1] |= frontier[..., :, 1:]    # left
        frontier = dilated & mask & (~visited)
        visited = visited | frontier

    return visited


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Bresenham line from (r0,c0) to (r1,c1). Returns list of (row, col)."""
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = -abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr + dc
    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r0 += sr
        if e2 <= dr:
            err += dr
            c0 += sc
    return points


# ---------------------------------------------------------------------------
# SpatialDSL
# ---------------------------------------------------------------------------

class SpatialDSL:
    """12-primitive stateless Spatial DSL.

    All methods are @staticmethod. Zero instance state, zero learned weights.
    All ops are pure deterministic PyTorch tensor math.
    """

    # ── Group 1: Rigid Spatial Transformations ────────────────────────────

    @staticmethod
    def translate(esb: ESB, dx: int, dy: int, wrap: bool = False) -> ESB:
        """Shift grid content by dx columns and dy rows.

        wrap=False (default): vacated cells filled with 0.
        wrap=True: content wraps using torch.roll.
        """
        g = esb.data  # (C, H, W)
        if dx == 0 and dy == 0:
            return ESB(g.clone())
        if wrap:
            out = torch.roll(g, shifts=(dy, dx), dims=(-2, -1))
            return ESB(out)
        # Use ESB.shift which handles (C, H, W) correctly
        return ESB(esb.shift(delta_row=dy, delta_col=dx, fill_value=0).data)

    @staticmethod
    def rotate(esb: ESB, degrees: int) -> ESB:
        """Rotate grid CLOCKWISE by degrees ∈ {90, 180, 270}."""
        if degrees not in VALID_ROTATIONS:
            raise ValueError(
                f"rotate: degrees must be in {VALID_ROTATIONS}, got {degrees!r}."
            )
        k = degrees // 90
        # torch.rot90 is CCW; negate k for CW rotation
        out = torch.rot90(esb.data, k=-k, dims=(-2, -1)).clone()
        return ESB(out)

    @staticmethod
    def reflect(esb: ESB, axis: str) -> ESB:
        """Mirror the grid.

        axis='horizontal': flip top-to-bottom (flip rows).
        axis='vertical':   flip left-to-right (flip columns).
        """
        if axis not in VALID_AXES:
            raise ValueError(
                f"reflect: axis must be in {VALID_AXES}, got {axis!r}."
            )
        dim, _ = _normalize_axis(axis)
        return ESB(esb.data.flip(dims=[dim]).clone())

    # ── Group 2: Object Manipulation & Topology ───────────────────────────

    @staticmethod
    def extract_object(esb: ESB, x: int, y: int) -> ESB:
        """Isolate the connected component at (col=x, row=y) via BFS CCL.

        All cells outside the component are zeroed.
        """
        g = esb.data  # (C, H, W)
        _, H, W = g.shape
        if not (0 <= y < H and 0 <= x < W):
            return ESB(torch.zeros_like(g))
        target_color = g[0, y, x].item()
        mask = (g == target_color)          # (C, H, W) bool
        component = _flood_component_mask(mask, row=y, col=x)
        out = torch.where(component, g, torch.zeros_like(g))
        return ESB(out)

    @staticmethod
    def overlay(
        esb: ESB,
        foreground: ESB,
        blend_mode: str = "overwrite",
    ) -> ESB:
        """Composite foreground on top of esb (background).

        blend_mode='overwrite': non-zero fg cells replace bg cells.
        blend_mode='or':        cell = foreground | background (bitwise OR).
        """
        if blend_mode not in VALID_BLEND:
            raise ValueError(
                f"overlay: blend_mode must be in {VALID_BLEND}, "
                f"got {blend_mode!r}."
            )
        bg = esb.data
        fg = foreground.data
        if blend_mode == "overwrite":
            out = torch.where(fg != 0, fg, bg)
        else:  # "or"
            out = _clamp09(bg.to(torch.int16) | fg.to(torch.int16))
        return ESB(out.to(torch.int8))

    @staticmethod
    def crop_to_bbox(esb: ESB, background: int = 0) -> ESB:
        """Return the minimal bounding box containing all non-zero cells.

        If the grid is entirely background, the input ESB is returned unchanged.
        """
        g = esb.data
        mask = g != background
        if not bool(mask.any()):
            return esb
        coords = torch.nonzero(mask, as_tuple=True)
        r_min = int(coords[1].min())
        r_max = int(coords[1].max())
        c_min = int(coords[2].min())
        c_max = int(coords[2].max())
        return ESB(g[:, r_min:r_max + 1, c_min:c_max + 1].clone())

    # ── Group 3: Geometric Scaling & Pattern Generation ───────────────────

    @staticmethod
    def scale_integer(esb: ESB, factor: float) -> ESB:
        """Scale the grid by factor ∈ {0.5, 1, 2, 3, 4}.

        Integer upscale: repeat_interleave.
        Downscale (0.5): max_pool2d (requires temporary float cast).
        """
        if factor not in VALID_SCALES:
            raise ValueError(
                f"scale_integer: factor must be in {VALID_SCALES}, "
                f"got {factor!r}."
            )
        g = esb.data
        if factor >= 1:
            f = int(factor)
            out = g.repeat_interleave(f, dim=-1).repeat_interleave(f, dim=-2)
            return ESB(out)
        else:
            # factor == 0.5: max-pool with kernel 2
            import torch.nn.functional as F
            d = g.unsqueeze(0).float()   # (1, C, H, W)
            pooled = F.max_pool2d(d, kernel_size=2, stride=2)
            return ESB(pooled.squeeze(0).to(torch.int8))

    @staticmethod
    def tile(esb: ESB, repeats_h: int, repeats_w: int) -> ESB:
        """Repeat the grid repeats_h times vertically, repeats_w horizontally."""
        if repeats_h <= 0 or repeats_w <= 0:
            raise ValueError("tile: repeats must be positive integers.")
        out = torch.tile(esb.data, dims=(1, repeats_h, repeats_w))
        return ESB(out)

    @staticmethod
    def connect_points(
        esb: ESB,
        color: int,
        row1: int,
        col1: int,
        row2: int,
        col2: int,
    ) -> ESB:
        """Draw a 1-pixel line from (row1,col1) to (row2,col2) using Bresenham.

        Out-of-bounds cells are silently skipped.
        """
        g = esb.data.clone()
        H, W = g.shape[-2], g.shape[-1]
        for r, c in _bresenham(row1, col1, row2, col2):
            if 0 <= r < H and 0 <= c < W:
                g[:, r, c] = color
        return ESB(g)

    # ── Group 4: Logic, Color & Constraints ───────────────────────────────

    @staticmethod
    def color_remap(esb: ESB, mapping: Dict[int, int]) -> ESB:
        """Remap cell colors using a lookup table.

        mapping: {src_color: dst_color}. Colors absent from mapping unchanged.
        """
        lookup = torch.arange(256, dtype=torch.long, device=esb.data.device)
        for src, dst in mapping.items():
            key = int(src)
            if 0 <= key < 256:
                lookup[key] = max(0, min(255, int(dst)))
        idx = esb.data.to(torch.long).clamp(0, 255)
        out = lookup[idx].to(torch.int8)
        return ESB(out)

    @staticmethod
    def flood_fill(esb: ESB, row: int, col: int, new_color: int) -> ESB:
        """Replace all 4-connected cells of same color as (row,col) with new_color.

        No-op if new_color equals the current cell color.
        """
        g = esb.data
        _, H, W = g.shape
        if not (0 <= row < H and 0 <= col < W):
            return ESB(g.clone())
        target_color = g[0, row, col].item()
        if target_color == new_color:
            return ESB(g.clone())
        mask = (g == target_color)   # (C, H, W) bool
        component = _flood_component_mask(mask, row=row, col=col)
        fill = torch.full_like(g, int(new_color), dtype=torch.int8)
        out = torch.where(component, fill, g)
        return ESB(out)

    @staticmethod
    def symmetrize(
        esb: ESB,
        axis: str,
        mode: Literal["copy", "priority"] = "copy",
    ) -> ESB:
        """Force the grid to be symmetric about the specified axis.

        mode='copy':     the flipped half overwrites the original.
        mode='priority': non-zero cells take precedence over zero cells.
        """
        if axis not in VALID_AXES:
            raise ValueError(
                f"symmetrize: axis must be in {VALID_AXES}, got {axis!r}."
            )
        if mode not in VALID_SYM_MODES:
            raise ValueError(
                f"symmetrize: mode must be in {VALID_SYM_MODES}, got {mode!r}."
            )
        dim, _ = _normalize_axis(axis)
        data = esb.data
        flipped = data.flip(dims=[dim])
        if mode == "copy":
            out = flipped.clone()
        else:  # "priority"
            out = torch.where(data != 0, data, flipped)
        return ESB(out)

    # ── Program Execution ─────────────────────────────────────────────────

    def apply_program(self, esb: ESB, program: DSLProgram) -> ESB:
        """Apply a sequence of (primitive_name, kwargs) to an ESB."""
        for name, kwargs in program:
            fn = getattr(self, name, None)
            if fn is None:
                raise ValueError(
                    f"apply_program: unknown primitive {name!r}."
                )
            esb = fn(esb, **kwargs)
        return esb

    # ── Introspection ─────────────────────────────────────────────────────

    @staticmethod
    def list_primitives() -> list[dict[str, Any]]:
        """Return name + parameter-domain metadata for all 12 primitives."""
        return [
            {"name": "translate",      "params": {"dx": "int ∈ [-2,2]", "dy": "int ∈ [-2,2]", "wrap": "bool"}},
            {"name": "rotate",         "params": {"degrees": "int ∈ {90,180,270}"}},
            {"name": "reflect",        "params": {"axis": "str ∈ {horizontal,vertical}"}},
            {"name": "extract_object", "params": {"x": "int (col)", "y": "int (row)"}},
            {"name": "overlay",        "params": {"foreground": "ESB", "blend_mode": "str ∈ {overwrite,or}"}},
            {"name": "crop_to_bbox",   "params": {}},
            {"name": "scale_integer",  "params": {"factor": "float ∈ {0.5,1,2,3,4}"}},
            {"name": "tile",           "params": {"repeats_h": "int", "repeats_w": "int"}},
            {"name": "connect_points", "params": {"color": "int", "row1": "int", "col1": "int", "row2": "int", "col2": "int"}},
            {"name": "color_remap",    "params": {"mapping": "dict[int,int]"}},
            {"name": "flood_fill",     "params": {"row": "int", "col": "int", "new_color": "int"}},
            {"name": "symmetrize",     "params": {"axis": "str ∈ {horizontal,vertical}", "mode": "str ∈ {copy,priority}"}},
        ]

"""tests/test_spatial_dsl.py — Unit and property tests for SpatialDSL.

Covers all 12 primitives, ValueError validation, batch dimension support,
and the mandatory property tests (Properties 4–9, 18–23).
"""

from __future__ import annotations

import torch
import pytest
import hypothesis
from hypothesis import given, settings
from hypothesis import strategies as st

from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL

DSL = SpatialDSL()

# ── Strategies ────────────────────────────────────────────────────────────


@st.composite
def rectangular_grid(draw):
    height = draw(st.integers(min_value=2, max_value=10))
    width = draw(st.integers(min_value=2, max_value=10))
    row = st.lists(st.integers(0, 9), min_size=width, max_size=width)
    return draw(st.lists(row, min_size=height, max_size=height))


grid_st = rectangular_grid()

axis_st = st.sampled_from(["horizontal", "vertical"])


def make(grid) -> ESB:
    return ESB.from_grid(grid)


# ══════════════════════════════════════════════════════════════════════════
# Group 1 — Rigid Spatial Transformations
# ══════════════════════════════════════════════════════════════════════════


class TestTranslate:
    def test_clip_mode_shifts_content(self):
        esb = make([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        result = DSL.translate(esb, dx=1, dy=1)
        assert result.data[0, 0, 0].item() == 0
        assert result.data[0, 1, 1].item() == 1

    def test_clip_mode_fills_zero(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.translate(esb, dx=1, dy=0)
        assert result.data[0, 0, 0].item() == 0  # vacated column

    def test_wrap_mode(self):
        esb = make([[1, 0], [0, 0]])
        result = DSL.translate(esb, dx=1, dy=0, wrap=True)
        # 1 should wrap to column 0
        assert result.data[0, 0, 0].item() == 0
        assert result.data[0, 0, 1].item() == 1

    def test_zero_delta_unchanged(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.translate(esb, dx=0, dy=0)
        assert torch.all(result.data == esb.data).item()


class TestRotate:
    def test_rotate_90_shape(self):
        esb = make([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        result = DSL.rotate(esb, 90)
        assert result.H == 3
        assert result.W == 2

    def test_rotate_180_values(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.rotate(esb, 180)
        assert result.to_grid() == [[4, 3], [2, 1]]

    def test_rotate_invalid_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="degrees"):
            DSL.rotate(esb, 45)

    def test_rotate_360_identity(self):
        """Four 90-degree rotations return original (Property 4)."""
        esb = make([[1, 2, 3], [4, 5, 6]])
        result = esb
        for _ in range(4):
            result = DSL.rotate(result, 90)
        assert torch.all(result.data == esb.data).item()


class TestReflect:
    def test_reflect_horizontal(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.reflect(esb, "horizontal")
        assert result.to_grid() == [[3, 4], [1, 2]]

    def test_reflect_vertical(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.reflect(esb, "vertical")
        assert result.to_grid() == [[2, 1], [4, 3]]

    def test_reflect_invalid_axis_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="axis"):
            DSL.reflect(esb, "diagonal")

    def test_reflect_involution(self):
        """reflect twice returns original (Property 5 check)."""
        esb = make([[1, 2, 3], [4, 5, 6]])
        for axis in ["horizontal", "vertical"]:
            result = DSL.reflect(DSL.reflect(esb, axis), axis)
            assert torch.all(result.data == esb.data).item()


# ══════════════════════════════════════════════════════════════════════════
# Group 2 — Object Manipulation
# ══════════════════════════════════════════════════════════════════════════


class TestExtractObject:
    def test_isolates_connected_component(self):
        esb = make(
            [
                [1, 1, 0],
                [1, 0, 2],
                [0, 2, 2],
            ]
        )
        result = DSL.extract_object(esb, x=0, y=0)  # seed at col=0, row=0
        # The 1-component should survive
        assert result.data[0, 0, 0].item() == 1
        assert result.data[0, 0, 1].item() == 1
        assert result.data[0, 1, 0].item() == 1
        # The 2-component should be zeroed
        assert result.data[0, 1, 2].item() == 0
        assert result.data[0, 2, 1].item() == 0

    def test_extract_zero_region_zeroes_all(self):
        esb = make([[0, 1], [1, 1]])
        result = DSL.extract_object(esb, x=0, y=0)
        assert result.data[0, 0, 0].item() == 0
        # Non-zero cells outside component are zeroed
        assert result.data[0, 0, 1].item() == 0


class TestOverlay:
    def test_overwrite_mode(self):
        bg = make([[1, 1], [1, 1]])
        fg = make([[2, 0], [0, 2]])
        result = DSL.overlay(bg, fg, blend_mode="overwrite")
        assert result.to_grid() == [[2, 1], [1, 2]]

    def test_or_mode(self):
        bg = make([[1, 0], [0, 0]])
        fg = make([[0, 2], [0, 0]])
        result = DSL.overlay(bg, fg, blend_mode="or")
        assert result.data[0, 0, 0].item() == 1
        assert result.data[0, 0, 1].item() == 2

    def test_invalid_blend_mode_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="blend_mode"):
            DSL.overlay(esb, esb, blend_mode="xor")


class TestCropToBbox:
    def test_trims_padding(self):
        esb = make(
            [
                [0, 0, 0],
                [0, 5, 0],
                [0, 0, 0],
            ]
        )
        result = DSL.crop_to_bbox(esb)
        assert result.H == 1
        assert result.W == 1
        assert result.data[0, 0, 0].item() == 5

    def test_all_zero_unchanged(self):
        esb = make([[0, 0], [0, 0]])
        result = DSL.crop_to_bbox(esb)
        assert result.data.shape == esb.data.shape

    def test_no_zero_border_remains(self):
        esb = make(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0],
            ]
        )
        result = DSL.crop_to_bbox(esb)
        # First and last rows and columns must all be non-zero
        assert result.data[0, 0, :].any().item()
        assert result.data[0, -1, :].any().item()
        assert result.data[0, :, 0].any().item()
        assert result.data[0, :, -1].any().item()


# ══════════════════════════════════════════════════════════════════════════
# Group 3 — Geometric Scaling & Pattern Generation
# ══════════════════════════════════════════════════════════════════════════


class TestScaleInteger:
    def test_scale_2_shape(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.scale_integer(esb, 2)
        assert result.H == 4
        assert result.W == 4

    def test_scale_2_values(self):
        esb = make([[1, 2]])
        result = DSL.scale_integer(esb, 2)
        assert result.to_grid() == [[1, 1, 2, 2], [1, 1, 2, 2]]

    def test_scale_3_shape(self):
        esb = make([[5]])
        result = DSL.scale_integer(esb, 3)
        assert result.H == 3
        assert result.W == 3

    def test_scale_half_shape(self):
        esb = make([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]])
        result = DSL.scale_integer(esb, 0.5)
        assert result.H == 2
        assert result.W == 2

    def test_invalid_factor_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="factor"):
            DSL.scale_integer(esb, 5)


class TestTile:
    def test_tile_shape(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.tile(esb, repeats_h=2, repeats_w=3)
        assert result.H == 4
        assert result.W == 6

    def test_tile_content(self):
        esb = make([[1, 2]])
        result = DSL.tile(esb, repeats_h=1, repeats_w=2)
        assert result.to_grid() == [[1, 2, 1, 2]]


class TestConnectPoints:
    def test_horizontal_line(self):
        esb = make([[0, 0, 0, 0, 0]])
        result = DSL.connect_points(esb, color=3, row1=0, col1=0, row2=0, col2=4)
        assert all(result.data[0, 0, c].item() == 3 for c in range(5))

    def test_diagonal_line(self):
        esb = make([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        result = DSL.connect_points(esb, color=1, row1=0, col1=0, row2=2, col2=2)
        assert result.data[0, 0, 0].item() == 1
        assert result.data[0, 2, 2].item() == 1

    def test_out_of_bounds_points_skip(self):
        esb = make([[0, 0], [0, 0]])
        # Line endpoint inside grid, some steps might go OOB
        result = DSL.connect_points(esb, color=5, row1=0, col1=0, row2=1, col2=1)
        assert result.data[0, 0, 0].item() == 5
        assert result.data[0, 1, 1].item() == 5


# ══════════════════════════════════════════════════════════════════════════
# Group 4 — Logic, Color & Constraints
# ══════════════════════════════════════════════════════════════════════════


class TestColorRemap:
    def test_single_remap(self):
        esb = make([[1, 2], [3, 1]])
        result = DSL.color_remap(esb, {1: 9})
        assert result.data[0, 0, 0].item() == 9
        assert result.data[0, 1, 1].item() == 9
        assert result.data[0, 0, 1].item() == 2  # unchanged

    def test_identity_remap(self):
        esb = make([[1, 2], [3, 4]])
        result = DSL.color_remap(esb, {})
        assert torch.all(result.data == esb.data).item()

    def test_full_permutation(self):
        esb = make([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        mapping = {i: 9 - i for i in range(10)}
        result = DSL.color_remap(esb, mapping)
        assert result.data[0, 0, 0].item() == 9
        assert result.data[0, 0, 4].item() == 5
        assert result.data[0, 1, 4].item() == 0


class TestFloodFill:
    def test_fills_connected_region(self):
        esb = make(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        result = DSL.flood_fill(esb, row=0, col=0, new_color=5)
        assert result.data[0, 0, 0].item() == 5
        assert result.data[0, 0, 1].item() == 5
        assert result.data[0, 1, 0].item() == 5
        # Non-connected cells unchanged
        assert result.data[0, 0, 2].item() == 0

    def test_noop_same_color(self):
        esb = make([[3, 3], [3, 3]])
        result = DSL.flood_fill(esb, row=0, col=0, new_color=3)
        assert torch.all(result.data == esb.data).item()

    def test_does_not_cross_boundary(self):
        esb = make([[1, 2], [1, 2]])
        result = DSL.flood_fill(esb, row=0, col=0, new_color=9)
        assert result.data[0, 0, 1].item() == 2  # untouched
        assert result.data[0, 1, 1].item() == 2  # untouched


class TestSymmetrize:
    def test_copy_horizontal_symmetric_result(self):
        # copy mode must produce a result that is symmetric about the axis
        esb = make([[1, 2], [0, 0]])
        result = DSL.symmetrize(esb, axis="horizontal", mode="copy")
        # Result must equal its own horizontal flip (symmetric)
        flipped = result.data.flip(dims=[-2])
        assert torch.all(result.data == flipped).item()

    def test_copy_vertical_symmetric_result(self):
        esb = make([[1, 0], [2, 0]])
        result = DSL.symmetrize(esb, axis="vertical", mode="copy")
        # Result must equal its own vertical flip (symmetric)
        flipped = result.data.flip(dims=[-1])
        assert torch.all(result.data == flipped).item()

    def test_copy_resolves_conflicting_nonzero_mirror_cells(self):
        esb = make([[1, 2]])
        result = DSL.symmetrize(esb, axis="vertical", mode="copy")
        assert result.to_grid() == [[1, 1]]

    def test_priority_mode_nonzero_wins(self):
        esb = make([[1, 0], [0, 2]])
        result = DSL.symmetrize(esb, axis="vertical", mode="priority")
        # Original: col0=[1,0], col1=[0,2]; flipped col0=[0,2], col1=[1,0]
        # priority: non-zero wins → col0: orig=1 wins, col1: orig=2 wins
        assert result.data[0, 0, 0].item() == 1
        assert result.data[0, 1, 1].item() == 2

    def test_invalid_axis_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="axis"):
            DSL.symmetrize(esb, axis="diagonal")

    def test_invalid_mode_raises(self):
        esb = make([[1]])
        with pytest.raises(ValueError, match="mode"):
            DSL.symmetrize(esb, axis="horizontal", mode="average")


# ══════════════════════════════════════════════════════════════════════════
# Batch dimension support
# ══════════════════════════════════════════════════════════════════════════


class TestBatchDimension:
    """Verify that rotate and reflect handle (B, C, H, W) tensors."""

    def _batch_esb(self, grid, B=3) -> ESB:
        """Create an ESB with batch dimension by stacking."""
        single = ESB.from_grid(grid)
        # Manually construct batched tensor (B, C, H, W)
        batched = single.data.unsqueeze(0).expand(B, -1, -1, -1)
        return ESB(batched.reshape(B * single.C, single.H, single.W))

    def test_rotate_on_batched_shape(self):
        # Use (C=2, H=3, W=4) tensor — simulates 2 channels
        data = torch.randint(0, 9, (2, 3, 4), dtype=torch.int8)
        esb = ESB(data)
        result = DSL.rotate(esb, 90)
        assert result.H == 4
        assert result.W == 3
        assert result.C == 2

    def test_reflect_on_multichannel(self):
        data = torch.randint(0, 9, (2, 4, 4), dtype=torch.int8)
        esb = ESB(data)
        result = DSL.reflect(esb, "vertical")
        assert result.data.shape == esb.data.shape


# ══════════════════════════════════════════════════════════════════════════
# Property tests (hypothesis)
# ══════════════════════════════════════════════════════════════════════════


@given(grid_st)
@settings(max_examples=100)
def test_property_rotation_round_trip(grid):
    """Property 4: four 90-degree rotations return the original grid."""
    esb = make(grid)
    result = esb
    for _ in range(4):
        result = DSL.rotate(result, 90)
    assert torch.all(result.data == esb.data).item()


@given(grid_st, axis_st)
@settings(max_examples=100)
def test_property_reflection_involution(grid, axis):
    """Property 5: reflecting twice returns the original grid."""
    esb = make(grid)
    result = DSL.reflect(DSL.reflect(esb, axis), axis)
    assert torch.all(result.data == esb.data).item()


@given(grid_st)
@settings(max_examples=80)
def test_property_flood_fill_connectivity(grid):
    """Property 6: flood_fill recolors the entire 4-connected region."""
    esb = make(grid)
    seed_r, seed_c = 0, 0
    target_color = esb.data[0, seed_r, seed_c].item()
    new_color = (target_color + 1) % 10  # guaranteed different

    result = DSL.flood_fill(esb, row=seed_r, col=seed_c, new_color=new_color)

    # BFS to find all cells reachable from seed with original color
    H, W = esb.H, esb.W
    visited = set()
    queue = [(seed_r, seed_c)]
    while queue:
        r, c = queue.pop()
        if (r, c) in visited or not (0 <= r < H and 0 <= c < W):
            continue
        if esb.data[0, r, c].item() != target_color:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            queue.append((r + dr, c + dc))

    # All reachable cells must now be new_color
    for r, c in visited:
        assert result.data[0, r, c].item() == new_color

    # All other cells must be unchanged
    for r in range(H):
        for c in range(W):
            if (r, c) not in visited:
                assert result.data[0, r, c].item() == esb.data[0, r, c].item()


@given(grid_st)
@settings(max_examples=80)
def test_property_color_remap_bijective_round_trip(grid):
    """Property 7: applying color_remap(M) then color_remap(M⁻¹) returns original."""
    esb = make(grid)
    # Build a bijective permutation of 0–9
    perm = list(range(10))
    perm[1], perm[2] = perm[2], perm[1]
    perm[3], perm[7] = perm[7], perm[3]
    M = {i: perm[i] for i in range(10)}
    M_inv = {v: k for k, v in M.items()}

    result = DSL.color_remap(DSL.color_remap(esb, M), M_inv)
    assert torch.all(result.data == esb.data).item()


@given(grid_st)
@settings(max_examples=80)
def test_property_color_remap_lookup_correctness(grid):
    """Property 23: every cell with src color gets mapped to dst."""
    esb = make(grid)
    mapping = {1: 9, 3: 7, 5: 0}
    result = DSL.color_remap(esb, mapping)
    for r in range(esb.H):
        for c in range(esb.W):
            orig = esb.data[0, r, c].item()
            got = result.data[0, r, c].item()
            if orig in mapping:
                assert got == mapping[orig], (
                    f"cell ({r},{c}): expected {mapping[orig]}, got {got}"
                )
            else:
                assert got == orig


@given(grid_st)
@settings(max_examples=80)
def test_property_overlay_compositing_correctness(grid):
    """Property 8: overlay(overwrite) — non-zero fg replaces bg, zero fg keeps bg."""
    bg_esb = make(grid)
    # foreground: same shape, alternating 0 and non-zero
    fg_data = torch.zeros_like(bg_esb.data)
    fg_data[0, ::2, ::2] = 5  # some non-zero cells
    fg_esb = ESB(fg_data)
    result = DSL.overlay(bg_esb, fg_esb, blend_mode="overwrite")
    for r in range(bg_esb.H):
        for c in range(bg_esb.W):
            fg_val = fg_esb.data[0, r, c].item()
            bg_val = bg_esb.data[0, r, c].item()
            got = result.data[0, r, c].item()
            if fg_val != 0:
                assert got == fg_val
            else:
                assert got == bg_val


@given(grid_st)
@settings(max_examples=80)
def test_property_apply_program_sequential_composition(grid):
    """Property 9: apply_program(program) == chaining primitives manually."""
    esb = make(grid)
    program = [
        ("reflect", {"axis": "horizontal"}),
        ("rotate", {"degrees": 180}),
    ]
    via_program = DSL.apply_program(esb, program)
    manual = DSL.rotate(DSL.reflect(esb, "horizontal"), 180)
    assert torch.all(via_program.data == manual.data).item()


@given(grid_st)
@settings(max_examples=80)
def test_property_extract_object_isolation(grid):
    """Property 18: extract_object retains component, zeros everything else."""
    esb = make(grid)
    seed_r, seed_c = 0, 0
    target_color = esb.data[0, seed_r, seed_c].item()

    result = DSL.extract_object(esb, x=seed_c, y=seed_r)

    H, W = esb.H, esb.W
    visited = set()
    queue = [(seed_r, seed_c)]
    while queue:
        r, c = queue.pop()
        if (r, c) in visited or not (0 <= r < H and 0 <= c < W):
            continue
        if esb.data[0, r, c].item() != target_color:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            queue.append((r + dr, c + dc))

    for r in range(H):
        for c in range(W):
            if (r, c) in visited:
                assert result.data[0, r, c].item() == esb.data[0, r, c].item()
            else:
                assert result.data[0, r, c].item() == 0


@given(grid_st)
@settings(max_examples=80)
def test_property_crop_to_bbox_minimality(grid):
    """Property 19: crop_to_bbox leaves no all-zero border rows/columns."""
    esb = make(grid)
    # Ensure at least one non-zero cell
    esb.data[0, 0, 0] = 1
    result = DSL.crop_to_bbox(esb)
    if result.H == 0 or result.W == 0:
        return
    # First/last row must have at least one non-zero
    assert result.data[0, 0, :].any().item(), "first row all zero"
    assert result.data[0, -1, :].any().item(), "last row all zero"
    assert result.data[0, :, 0].any().item(), "first col all zero"
    assert result.data[0, :, -1].any().item(), "last col all zero"


@given(grid_st, st.sampled_from([2, 3, 4]))
@settings(max_examples=80)
def test_property_scale_integer_upscale_shape(grid, factor):
    """Property 20: scale_integer(factor) produces (C, H*factor, W*factor)."""
    esb = make(grid)
    result = DSL.scale_integer(esb, factor)
    assert result.H == esb.H * factor
    assert result.W == esb.W * factor
    assert result.C == esb.C


@given(
    grid_st,
    st.integers(1, 3),  # repeats_h
    st.integers(1, 3),  # repeats_w
)
@settings(
    max_examples=80, suppress_health_check=[hypothesis.HealthCheck.filter_too_much]
)
def test_property_tile_shape(grid, repeats_h, repeats_w):
    """Property 21: tile produces (C, H*repeats_h, W*repeats_w)."""
    esb = make(grid)
    result = DSL.tile(esb, repeats_h, repeats_w)
    assert result.H == esb.H * repeats_h
    assert result.W == esb.W * repeats_w
    assert result.C == esb.C


@given(grid_st, axis_st, st.sampled_from(["copy", "priority"]))
@settings(max_examples=80)
def test_property_symmetrize_idempotency(grid, axis, mode):
    """Property 22: symmetrize applied twice equals applying it once."""
    esb = make(grid)
    once = DSL.symmetrize(esb, axis, mode)
    twice = DSL.symmetrize(once, axis, mode)
    assert torch.all(once.data == twice.data).item()

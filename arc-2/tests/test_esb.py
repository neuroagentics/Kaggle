"""tests/test_esb.py — Unit and property-based tests for ESB.

Property tests use hypothesis. Unit tests cover each operation and
ValueError edge cases.
"""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from hyper_arc.esb import ESB

# ── Hypothesis strategies ──────────────────────────────────────────────────

# ARC grids: height and width 1–30, colors 0–9
grid_strategy = st.lists(
    st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=30),
    min_size=1,
    max_size=30,
).filter(lambda g: all(len(row) == len(g[0]) for row in g))  # rectangular


def make_esb(grid: list[list[int]]) -> ESB:
    return ESB.from_grid(grid)


# ── Unit tests: from_grid / to_grid ───────────────────────────────────────


def test_from_grid_shape_and_dtype():
    grid = [[1, 2, 3], [4, 5, 6]]
    esb = ESB.from_grid(grid)
    assert esb.data.shape == (1, 2, 3)
    assert esb.data.dtype == torch.int8
    assert esb.C == 1
    assert esb.H == 2
    assert esb.W == 3


def test_to_grid_round_trip():
    grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    esb = ESB.from_grid(grid)
    assert esb.to_grid() == grid


# ── Unit tests: pad ────────────────────────────────────────────────────────


def test_pad_shape():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    padded = esb.pad(top=1, bottom=2, left=1, right=3)
    assert padded.H == 2 + 1 + 2  # 5
    assert padded.W == 2 + 1 + 3  # 6


def test_pad_fill_value():
    esb = ESB.from_grid([[5]])
    padded = esb.pad(top=1, bottom=1, left=1, right=1, fill_value=9)
    assert padded.data[0, 0, 0].item() == 9  # corner
    assert padded.data[0, 1, 1].item() == 5  # original cell


def test_pad_zero_padding_no_change():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    padded = esb.pad(0, 0, 0, 0)
    assert torch.all(padded.data == esb.data).item()


def test_pad_raises_on_negative_result():
    esb = ESB.from_grid([[1]])
    with pytest.raises(ValueError, match="pad would produce"):
        esb.pad(top=0, bottom=-2, left=0, right=0)


# ── Unit tests: crop ───────────────────────────────────────────────────────


def test_crop_shape():
    esb = ESB.from_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cropped = esb.crop(0, 2, 1, 3)
    assert cropped.H == 2
    assert cropped.W == 2


def test_crop_values():
    esb = ESB.from_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cropped = esb.crop(1, 3, 1, 3)
    assert cropped.to_grid() == [[5, 6], [8, 9]]


def test_crop_raises_on_zero_height():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="crop region"):
        esb.crop(1, 1, 0, 2)  # row_end == row_start → height 0


def test_crop_raises_on_zero_width():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="crop region"):
        esb.crop(0, 2, 1, 1)  # col_end == col_start → width 0


# ── Unit tests: mask ───────────────────────────────────────────────────────


def test_mask_zeros_false_cells():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    condition = torch.tensor([[True, False], [False, True]])
    masked = esb.mask(condition)
    assert masked.data[0, 0, 0].item() == 1
    assert masked.data[0, 0, 1].item() == 0
    assert masked.data[0, 1, 0].item() == 0
    assert masked.data[0, 1, 1].item() == 4


def test_mask_all_true_unchanged():
    esb = ESB.from_grid([[5, 6], [7, 8]])
    condition = torch.ones(2, 2, dtype=torch.bool)
    masked = esb.mask(condition)
    assert torch.all(masked.data == esb.data).item()


def test_mask_all_false_zeros():
    esb = ESB.from_grid([[5, 6], [7, 8]])
    condition = torch.zeros(2, 2, dtype=torch.bool)
    masked = esb.mask(condition)
    assert torch.all(masked.data == 0).item()


# ── Unit tests: shift ──────────────────────────────────────────────────────


def test_shift_preserves_shape():
    esb = ESB.from_grid([[1, 2, 3], [4, 5, 6]])
    shifted = esb.shift(1, 1)
    assert shifted.data.shape == esb.data.shape


def test_shift_down_right():
    esb = ESB.from_grid([[1, 0], [0, 0]])
    shifted = esb.shift(delta_row=1, delta_col=1)
    # The 1 should move to (1,1)
    assert shifted.data[0, 0, 0].item() == 0
    assert shifted.data[0, 1, 1].item() == 1


def test_shift_fill_value():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    shifted = esb.shift(delta_row=1, delta_col=0, fill_value=7)
    assert shifted.data[0, 0, 0].item() == 7
    assert shifted.data[0, 0, 1].item() == 7


def test_shift_zero_delta_unchanged():
    esb = ESB.from_grid([[1, 2], [3, 4]])
    shifted = esb.shift(0, 0)
    assert torch.all(shifted.data == esb.data).item()


# ── Property tests ─────────────────────────────────────────────────────────


@given(grid_strategy)
@settings(max_examples=100)
def test_property_from_grid_shape_invariant(grid: list[list[int]]):
    """Property 1 (partial): from_grid always produces (1, H, W) int8 tensor."""
    esb = ESB.from_grid(grid)
    H = len(grid)
    W = len(grid[0])
    assert esb.data.shape == (1, H, W)
    assert esb.data.dtype == torch.int8


@given(
    grid_strategy,
    st.integers(0, 5),  # top
    st.integers(0, 5),  # bottom
    st.integers(0, 5),  # left
    st.integers(0, 5),  # right
)
@settings(max_examples=100)
def test_property_pad_shape_invariant(
    grid: list[list[int]],
    top: int,
    bottom: int,
    left: int,
    right: int,
):
    """Property 1 (partial): pad produces (C, H+top+bottom, W+left+right)."""
    esb = ESB.from_grid(grid)
    padded = esb.pad(top, bottom, left, right)
    assert padded.H == esb.H + top + bottom
    assert padded.W == esb.W + left + right
    assert padded.C == esb.C


@given(
    grid_strategy,
    st.integers(-10, 10),  # delta_row
    st.integers(-10, 10),  # delta_col
)
@settings(max_examples=100)
def test_property_shift_preserves_shape(
    grid: list[list[int]],
    delta_row: int,
    delta_col: int,
):
    """Property 1 (partial): shift always preserves (C, H, W) shape."""
    esb = ESB.from_grid(grid)
    shifted = esb.shift(delta_row, delta_col)
    assert shifted.data.shape == esb.data.shape


@given(
    grid_strategy,
    st.integers(0, 10),   # row_start offset from 0
    st.integers(1, 10),   # crop height
    st.integers(0, 10),   # col_start offset from 0
    st.integers(1, 10),   # crop width
)
@settings(max_examples=100)
def test_property_crop_shape(
    grid: list[list[int]],
    row_off: int,
    crop_h: int,
    col_off: int,
    crop_w: int,
):
    """Property 2: crop produces (C, crop_h, crop_w) when in bounds."""
    esb = ESB.from_grid(grid)
    row_start = row_off % esb.H
    row_end = min(row_start + crop_h, esb.H)
    col_start = col_off % esb.W
    col_end = min(col_start + crop_w, esb.W)
    if row_end <= row_start or col_end <= col_start:
        return  # skip degenerate cases
    cropped = esb.crop(row_start, row_end, col_start, col_end)
    assert cropped.H == row_end - row_start
    assert cropped.W == col_end - col_start
    assert cropped.C == esb.C


@given(grid_strategy)
@settings(max_examples=100)
def test_property_mask_zeros_false_cells(grid: list[list[int]]):
    """Property 3: mask zeros out cells where condition is False."""
    esb = ESB.from_grid(grid)
    # Use a checkerboard condition
    condition = torch.zeros(esb.H, esb.W, dtype=torch.bool)
    condition[::2, ::2] = True
    masked = esb.mask(condition)
    # All False-condition cells must be 0
    false_cells = masked.data[:, ~condition]
    assert torch.all(false_cells == 0).item()
    # All True-condition cells must match original
    true_cells_orig = esb.data[:, condition]
    true_cells_masked = masked.data[:, condition]
    assert torch.all(true_cells_orig == true_cells_masked).item()

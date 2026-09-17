"""hyper_arc/esb.py — Eidetic Spatial Buffer.

Immutable tensor wrapper for ARC grids. All operations return new ESB
instances; the original is never mutated.

Grid tensors use torch.int8 (values 0-9 fit in 8 bits) to minimise VRAM
on free-tier GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ESB:
    """Eidetic Spatial Buffer — immutable tensor wrapper for ARC grids.

    Shape convention: ``(C, H, W)`` where C is the number of channels
    (typically 1 for single-channel integer grids), H is height, W is width.
    dtype is always ``torch.int8``.
    """

    data: Tensor  # shape (C, H, W), dtype torch.int8

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_grid(cls, grid: list[list[int]]) -> "ESB":
        """Build a single-channel ESB from a 2-D ARC grid (list of lists)."""
        t = torch.tensor(grid, dtype=torch.int8).unsqueeze(0)  # (1, H, W)
        return cls(data=t)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def C(self) -> int:
        return self.data.shape[0]

    @property
    def H(self) -> int:
        return self.data.shape[1]

    @property
    def W(self) -> int:
        return self.data.shape[2]

    # ── Core Operations ────────────────────────────────────────────────────

    def pad(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
        fill_value: int = 0,
    ) -> "ESB":
        """Return a new ESB padded by the given number of cells on each side.

        Raises ValueError if the resulting height or width would be < 1.
        """
        new_h = self.H + top + bottom
        new_w = self.W + left + right
        if new_h < 1 or new_w < 1:
            raise ValueError(
                f"pad would produce shape ({new_h}, {new_w}); "
                "both dimensions must be >= 1."
            )
        out = torch.full(
            (self.C, new_h, new_w),
            fill_value,
            dtype=torch.int8,
        )
        out[:, top : top + self.H, left : left + self.W] = self.data
        return ESB(out)

    def crop(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> "ESB":
        """Return a new ESB containing the specified sub-region.

        Raises ValueError if the resulting height or width would be < 1.
        """
        h = row_end - row_start
        w = col_end - col_start
        if h < 1 or w < 1:
            raise ValueError(
                f"crop region has size ({h}, {w}); "
                "both dimensions must be >= 1."
            )
        return ESB(self.data[:, row_start:row_end, col_start:col_end].clone())

    def mask(self, condition: Tensor) -> "ESB":
        """Return a new ESB where cells failing *condition* are zeroed out.

        Args:
            condition: Boolean tensor of shape ``(H, W)`` or broadcastable.
                       Cells where condition is True retain their value;
                       cells where condition is False are set to 0.
        """
        out = self.data.clone()
        out[:, ~condition] = 0
        return ESB(out)

    def shift(
        self,
        delta_row: int,
        delta_col: int,
        fill_value: int = 0,
    ) -> "ESB":
        """Return a new ESB with content translated by (delta_row, delta_col).

        The output shape is identical to the input shape. Vacated cells are
        filled with *fill_value* (default 0, the ARC background color).
        """
        out = torch.full_like(self.data, fill_value)
        src_r0 = max(0, -delta_row)
        src_r1 = self.H - max(0, delta_row)
        dst_r0 = max(0, delta_row)
        dst_r1 = self.H - max(0, -delta_row)
        src_c0 = max(0, -delta_col)
        src_c1 = self.W - max(0, delta_col)
        dst_c0 = max(0, delta_col)
        dst_c1 = self.W - max(0, -delta_col)
        if src_r0 < src_r1 and src_c0 < src_c1:
            out[:, dst_r0:dst_r1, dst_c0:dst_c1] = (
                self.data[:, src_r0:src_r1, src_c0:src_c1]
            )
        return ESB(out)

    # ── Export ─────────────────────────────────────────────────────────────

    def to_grid(self) -> list[list[int]]:
        """Export channel 0 as a 2-D list of Python ints."""
        return self.data[0].tolist()

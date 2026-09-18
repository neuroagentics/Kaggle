"""Tests for the generalization gate: a general program passes, a program that
baked a demo literal (fixed color) fails the color-permutation invariance gate.
"""

from __future__ import annotations

from hyper_arc.contender.generalization import generalization_gate
from hyper_arc.contender.schemas import GridPair, ProgramAST, ValueType


def _input():
    return ProgramAST(op="input", output_type=ValueType.GRID)


def _pairs(rows):
    return tuple(GridPair(input=i, output=o) for i, o in rows)


def test_color_invariant_program_passes_the_gate():
    # rotate180 encodes a genuinely general rule: it commutes with any color
    # permutation and every dihedral op, so it must pass.
    program = ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": 180},
        children=(_input(),),
    )
    pairs = _pairs(
        [
            (((1, 2), (3, 4)), ((4, 3), (2, 1))),
            (((5, 0), (0, 6)), ((6, 0), (0, 5))),
        ]
    )
    report = generalization_gate(program, pairs)
    assert report.leave_one_out_exact is True
    assert report.generalizes is True
    assert "color_permutation" not in report.failed_augmentations


def test_baked_color_literal_program_fails_the_gate():
    # A program that remaps every cell to the literal color 4 fits demos whose
    # output is all-4, but it is NOT color-invariant: under a palette relabel the
    # transformed output is no longer all-4, so program(T(input)) != T(output).
    program = ProgramAST(
        op="color_remap",
        output_type=ValueType.GRID,
        arguments={"mapping": {0: 4, 1: 4, 2: 4, 3: 4, 5: 4, 6: 4}},
        children=(_input(),),
    )
    pairs = _pairs(
        [
            (((1, 2), (3, 0)), ((4, 4), (4, 4))),
            (((5, 6), (0, 1)), ((4, 4), (4, 4))),
        ]
    )
    report = generalization_gate(program, pairs)
    # It does fit every demo exactly (LODO true)...
    assert report.leave_one_out_exact is True
    # ...but it memorized the literal target color, so the gate rejects it.
    assert report.generalizes is False
    assert "color_permutation" in report.failed_augmentations


def test_gate_handles_empty_pairs():
    program = ProgramAST(op="input", output_type=ValueType.GRID)
    report = generalization_gate(program, ())
    assert report.generalizes is False


def test_gate_passes_color_invariant_when_palette_too_small_for_permutation():
    # Single-color grids can't form a non-identity permutation; the color gate
    # is then not applicable and must not spuriously fail a valid program.
    program = ProgramAST(op="input", output_type=ValueType.GRID)  # identity
    pairs = _pairs([(((0, 0), (0, 0)), ((0, 0), (0, 0)))])
    report = generalization_gate(program, pairs)
    assert report.leave_one_out_exact is True
    assert report.generalizes is True

"""Advisory invariance diagnostics are not unseen-task generalization tests."""

from __future__ import annotations

from hyper_arc.contender.generalization import augmentation_consistency
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
    report = augmentation_consistency(program, pairs)
    assert report.demonstration_exact is True
    assert report.passes_checks is True
    assert "color_permutation" not in report.failed_augmentations


def test_color_literal_is_not_rejected_without_a_rule_specific_invariance_assumption():
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
    report = augmentation_consistency(program, pairs)
    # Ordinary demo replay is true; no leave-one-out induction was performed.
    assert report.demonstration_exact is True
    # A fixed target color can be the correct rule, not necessarily memorization.
    assert report.passes_checks is True  # valid color-specific rules remain eligible
    assert augmentation_consistency(program, pairs, require_color_invariance=True).passes_checks is False
    assert "color_permutation" in report.failed_augmentations


def test_gate_handles_empty_pairs():
    program = ProgramAST(op="input", output_type=ValueType.GRID)
    report = augmentation_consistency(program, ())
    assert report.passes_checks is False


def test_gate_passes_color_invariant_when_palette_too_small_for_permutation():
    # Single-color grids can't form a non-identity permutation; the color gate
    # is then not applicable and must not spuriously fail a valid program.
    program = ProgramAST(op="input", output_type=ValueType.GRID)  # identity
    pairs = _pairs([(((0, 0), (0, 0)), ((0, 0), (0, 0)))])
    report = augmentation_consistency(program, pairs)
    assert report.demonstration_exact is True
    assert report.passes_checks is True

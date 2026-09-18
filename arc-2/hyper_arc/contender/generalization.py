"""Generalization gate: distinguish a demo-fitting program that encodes the
GENERAL rule from one that merely memorized demo specifics.

Motivation (measured): every channel accepts a program on the single criterion
"reproduces all training demos exactly." With 2-5 demos and thousands of
candidates, spurious programs fit the demos by accident and are trusted, then
fail on the unseen test input. This is the mechanism behind the 57/160 (tuned
split) -> ~2/120 (unseen eval) collapse.

This module adds two channel-agnostic, exact-only consistency checks that a
program must survive to be *trusted* for the test input. Neither re-fits the
program; both reuse the existing TypedExecutor:

1. Augmentation-consistency (the strong signal). Apply a rule-preserving
   transform T to a demonstration pair and require program(T(input)) == T(output).
   A program that encodes the general rule COMMUTES with T; one that baked demo
   literals (a fixed color, offset, coordinate, or attribute value) breaks,
   because the literal no longer matches the transformed grid.

2. Leave-one-demo-out replay. Re-verify the program reproduces each demo output
   individually. Cheap sanity check that the accepted program is genuinely exact
   on every demo (guards against acceptance bugs), and the natural hook if a
   channel later re-fits on N-1 demos.

The gate returns a consistency score in [0, 1] and a boolean `generalizes`
verdict, so callers can use it as a hard filter or a ranking signal. Colors are
treated as categorical labels throughout — augmentations never do arithmetic on
color values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from hyper_arc.contender.executor import ExecutionError, TypedExecutor
from hyper_arc.contender.schemas import Grid, GridPair, ProgramAST


# A rule-preserving augmentation maps a whole grid (input or output) the same
# way, so a correct rule's output commutes with it.
Augmentation = Callable[[Grid], Grid]


def _rotate90(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def _rotate180(grid: Grid) -> Grid:
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def _rotate270(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))[::-1]


def _reflect_h(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in grid[::-1])


def _reflect_v(grid: Grid) -> Grid:
    return tuple(tuple(row[::-1]) for row in grid)


def _transpose(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))


# Dihedral group (excluding identity, which is the plain replay already checked).
_DIHEDRAL: tuple[tuple[str, Augmentation], ...] = (
    ("rotate90", _rotate90),
    ("rotate180", _rotate180),
    ("rotate270", _rotate270),
    ("reflect_h", _reflect_h),
    ("reflect_v", _reflect_v),
    ("transpose", _transpose),
)


def _color_permutation(pairs: Sequence[GridPair], seed: int) -> Augmentation | None:
    """A deterministic non-identity permutation of the non-background colors
    present across the demos. Returns None if fewer than two colors are movable.
    """
    palette = sorted({v for pair in pairs for grid in (pair.input, pair.output) for row in grid for v in row})
    if len(palette) < 2:
        return None
    # Rotate the palette by (seed mod len-1)+1 so it is a derangement of at
    # least the moved colors; 0 (background) is included so mappings stay total.
    shift = (seed % (len(palette) - 1)) + 1
    mapping = {c: palette[(i + shift) % len(palette)] for i, c in enumerate(palette)}
    if all(k == v for k, v in mapping.items()):
        return None

    def apply(grid: Grid) -> Grid:
        return tuple(tuple(mapping[v] for v in row) for row in grid)

    return apply


@dataclass(frozen=True)
class GateReport:
    generalizes: bool
    consistency: float          # fraction of augmentation checks survived
    checks_run: int
    checks_passed: int
    leave_one_out_exact: bool
    failed_augmentations: tuple[str, ...]


def _runs_exact(
    executor: TypedExecutor, program: ProgramAST, input_grid: Grid, expected: Grid
) -> bool:
    try:
        result = executor.execute(program, input_grid, capture_trace=False)
    except (ExecutionError, ValueError, IndexError, RecursionError):
        return False
    value = result.value
    # Executor may return an ESB or nested lists; normalize to tuple-of-tuples.
    if hasattr(value, "to_grid"):
        value = value.to_grid()
    try:
        normalized = tuple(tuple(int(v) for v in row) for row in value)
    except (TypeError, ValueError):
        return False
    return normalized == tuple(tuple(row) for row in expected)


def generalization_gate(
    program: ProgramAST,
    train_pairs: Sequence[GridPair],
    *,
    executor: TypedExecutor | None = None,
    require_color_invariance: bool = True,
) -> GateReport:
    """Score whether a demo-fitting program encodes the general rule.

    Discriminator design (important): not every ARC rule commutes with every
    augmentation. A directional rule ("move down by 3") legitimately breaks
    under rotation, so dihedral checks are reported as a soft signal only. The
    HARD gate is COLOR-PERMUTATION invariance: almost all ARC rules treat colors
    as categorical labels, so a general rule commutes with a relabeling of the
    palette, whereas a program that baked a literal color ("paint blue", "select
    the blue object") does NOT. Color-literal overfitting is the most common
    memorization mode in the generators, so this is the high-signal check.

    A program PASSES (generalizes=True) when it is exact on every demo (LODO) and
    — when require_color_invariance is set and a permutation is applicable — it is
    also exact under a non-identity color permutation of every demo pair. The
    dihedral checks populate `consistency`/`failed_augmentations` as advisory
    ranking signal but do not by themselves fail the gate.
    """
    executor = executor or TypedExecutor()
    pairs = tuple(
        GridPair(
            input=tuple(tuple(row) for row in pair.input),
            output=tuple(tuple(row) for row in pair.output),
        )
        for pair in train_pairs
    )
    if not pairs:
        return GateReport(False, 0.0, 0, 0, False, ())

    # 1) Leave-one-demo-out replay: the program must be exact on each demo.
    leave_one_out_exact = all(
        _runs_exact(executor, program, pair.input, pair.output) for pair in pairs
    )

    # 2) Dihedral consistency — ADVISORY only (directional rules may break it).
    checks_run = 0
    checks_passed = 0
    failed: list[str] = []
    for name, transform in _DIHEDRAL:
        if all(
            _runs_exact(executor, program, transform(p.input), transform(p.output))
            for p in pairs
        ):
            checks_passed += 1
        else:
            failed.append(name)
        checks_run += 1
    consistency = checks_passed / checks_run if checks_run else 0.0

    # 3) Color-permutation invariance — the HARD gate. Skipped only when the
    #    palette is too small to form a non-identity permutation.
    color_invariant = True
    color_applicable = False
    perm = _color_permutation(pairs, seed=len(pairs))
    if perm is not None:
        color_applicable = True
        color_invariant = all(
            _runs_exact(executor, program, perm(p.input), perm(p.output))
            for p in pairs
        )
        if not color_invariant:
            failed.append("color_permutation")

    generalizes = leave_one_out_exact and (
        color_invariant if (require_color_invariance and color_applicable) else True
    )
    return GateReport(
        generalizes=generalizes,
        consistency=consistency,
        checks_run=checks_run,
        checks_passed=checks_passed,
        leave_one_out_exact=leave_one_out_exact,
        failed_augmentations=tuple(failed),
    )

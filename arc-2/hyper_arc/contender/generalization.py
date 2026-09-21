"""Research-only augmentation diagnostics. Not used for production acceptance."""

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
class ConsistencyReport:
    passes_checks: bool
    consistency: float          # fraction of augmentation checks survived
    checks_run: int
    checks_passed: int
    demonstration_exact: bool
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


def augmentation_consistency(
    program: ProgramAST,
    train_pairs: Sequence[GridPair],
    *,
    executor: TypedExecutor | None = None,
    require_color_invariance: bool = False,
) -> ConsistencyReport:
    """Advisory metamorphic checks, not evidence of unseen-task generalization.

    Only opt into color filtering when a rule-specific invariance is justified.
    Replaying demonstrations is not leave-one-out induction: no example is
    withheld from program generation. Neither result certifies hidden outputs.
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
        return ConsistencyReport(False, 0.0, 0, 0, False, ())

    # 1) Ordinary demonstration replay (no withheld-example claim).
    demonstration_exact = all(
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

    # 3) Optional color consistency, advisory by default. Skipped when the
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

    passes_checks = demonstration_exact and (
        color_invariant if (require_color_invariance and color_applicable) else True
    )
    return ConsistencyReport(
        passes_checks=passes_checks,
        consistency=consistency,
        checks_run=checks_run,
        checks_passed=checks_passed,
        demonstration_exact=demonstration_exact,
        failed_augmentations=tuple(failed),
    )

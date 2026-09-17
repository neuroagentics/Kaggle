"""Bounded residual diagnosis and recursive repair for typed grid programs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from hyper_arc.contender.executor import ExecutionError, TypedExecutor, replay_program
from hyper_arc.contender.perception import canonical_grid, perceive_grid
from hyper_arc.contender.schemas import (
    Grid,
    GridPair,
    Hypothesis,
    ProgramAST,
    Residual,
    ValueType,
    stable_digest,
)


@dataclass(frozen=True)
class ResidualDiagnosis:
    """Structured explanation of one non-exact demonstration replay."""

    residual: Residual
    cause: str
    mismatch_count: int


@dataclass(frozen=True)
class RepairSpec:
    """One typed wrapper and its aggregate demonstration error."""

    name: str
    op: str
    arguments: Mapping[str, object]
    error: int


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _grid_error(prediction: Grid, expected: Grid) -> int:
    """Cell mismatch plus a bounded penalty for incompatible canvases."""
    ph, pw = _shape(prediction)
    eh, ew = _shape(expected)
    overlap = sum(
        prediction[row][col] != expected[row][col]
        for row in range(min(ph, eh))
        for col in range(min(pw, ew))
    )
    return overlap + abs(ph * pw - eh * ew) + abs(ph - eh) + abs(pw - ew)


def grid_error(prediction: Grid, expected: Grid) -> int:
    """Public error metric used by the recursive world-model search."""
    return _grid_error(prediction, expected)


def _object_difference(first: Grid, second: Grid, prefix: str) -> tuple[str, ...]:
    first_state = perceive_grid(first, f"residual:{prefix}:first")
    second_state = perceive_grid(second, f"residual:{prefix}:second")
    available = Counter(item.shape_signature for item in first_state.objects)
    missing: list[str] = []
    for item in second_state.objects:
        if available[item.shape_signature]:
            available[item.shape_signature] -= 1
        else:
            missing.append(item.shape_signature)
    return tuple(sorted(missing))


def diagnose_residual(
    prediction: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    *,
    example_index: int,
) -> ResidualDiagnosis:
    """Classify a failed replay without treating partial accuracy as acceptance."""
    actual = canonical_grid(prediction)
    target = canonical_grid(expected)
    actual_shape, expected_shape = _shape(actual), _shape(target)
    mismatches: tuple[tuple[int, int], ...] = ()
    if actual_shape == expected_shape:
        mismatches = tuple(
            (row, col)
            for row in range(expected_shape[0])
            for col in range(expected_shape[1])
            if actual[row][col] != target[row][col]
        )
    missing = _object_difference(actual, target, f"{example_index}:missing")
    extra = _object_difference(target, actual, f"{example_index}:extra")
    if actual_shape != expected_shape:
        cause = "wrong_canvas"
    elif not mismatches:
        cause = "exact"
    elif missing and not extra:
        cause = "missing_content"
    elif extra and not missing:
        cause = "extra_content"
    elif _consistent_color_map((actual,), (target,)) is not None:
        cause = "wrong_color_binding"
    else:
        cause = "local_or_structural"
    return ResidualDiagnosis(
        residual=Residual(
            example_index=example_index,
            mismatched_cells=mismatches,
            missing_objects=missing,
            extra_objects=extra,
            expected_shape=expected_shape,
            actual_shape=actual_shape,
        ),
        cause=cause,
        mismatch_count=_grid_error(actual, target),
    )


def _consistent_color_map(
    predictions: Sequence[Grid], targets: Sequence[Grid]
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for prediction, target in zip(predictions, targets):
        if _shape(prediction) != _shape(target):
            return None
        for before_row, after_row in zip(prediction, target):
            for before, after in zip(before_row, after_row):
                if before in mapping and mapping[before] != after:
                    return None
                mapping[before] = after
    changed = {before: after for before, after in mapping.items() if before != after}
    return changed or None


def _wrap(program: ProgramAST, spec: RepairSpec) -> ProgramAST:
    return ProgramAST(
        op=spec.op,
        output_type=ValueType.GRID,
        arguments=dict(spec.arguments),
        children=(program,),
    )


def wrap_program(program: ProgramAST, spec: RepairSpec) -> ProgramAST:
    """Apply one typed residual repair to a program."""
    return _wrap(program, spec)


def _repair_specs(
    predictions: Sequence[Grid], targets: Sequence[Grid], current_error: int
) -> tuple[RepairSpec, ...]:
    specs: list[RepairSpec] = []
    mapping = _consistent_color_map(predictions, targets)
    if mapping:
        specs.append(RepairSpec("color-remap", "color_remap", {"mapping": mapping}, 0))
    fixed: tuple[tuple[str, str, Mapping[str, object]], ...] = (
        ("crop-mode", "crop_mode", {}),
        ("rotate-90", "rotate", {"degrees": 90}),
        ("rotate-180", "rotate", {"degrees": 180}),
        ("rotate-270", "rotate", {"degrees": 270}),
        ("reflect-horizontal", "reflect", {"axis": "horizontal"}),
        ("reflect-vertical", "reflect", {"axis": "vertical"}),
        ("transpose", "transpose", {}),
        ("anti-transpose", "anti_transpose", {}),
    )
    executor = TypedExecutor()
    source = ProgramAST(op="input", output_type=ValueType.GRID)
    for name, op, arguments in fixed:
        wrapper = ProgramAST(
            op=op,
            output_type=ValueType.GRID,
            arguments=arguments,
            children=(source,),
        )
        try:
            transformed = tuple(
                executor.execute(wrapper, prediction, capture_trace=False).value
                for prediction in predictions
            )
        except ExecutionError:
            continue
        error = sum(
            _grid_error(item, target) for item, target in zip(transformed, targets)
        )
        if error < current_error:
            specs.append(RepairSpec(name, op, arguments, error))
    return tuple(sorted(specs, key=lambda item: (item.error, item.name)))


def propose_repairs(
    predictions: Sequence[Grid], targets: Sequence[Grid], current_error: int
) -> tuple[RepairSpec, ...]:
    """Return only wrappers that strictly reduce aggregate demonstration error."""
    return _repair_specs(predictions, targets, current_error)


def repair_hypotheses(
    seeds: Iterable[tuple[str, ProgramAST]],
    train_pairs: Sequence[GridPair],
    *,
    task_id: str,
    max_depth: int = 2,
    max_candidates: int = 256,
) -> list[Hypothesis]:
    """Recursively wrap improving seeds and return exact-replay hypotheses only."""
    pairs = tuple(train_pairs)
    if not pairs:
        raise ExecutionError("Residual repair requires at least one training pair")
    executor = TypedExecutor()
    queue = deque((name, program, 0, None) for name, program in seeds)
    seen: set[str] = set()
    seen_behaviors: set[tuple[Grid, ...]] = set()
    exact: list[Hypothesis] = []
    executed = 0
    while queue and executed < max_candidates:
        name, program, depth, parent_id = queue.popleft()
        if program.digest in seen:
            continue
        seen.add(program.digest)
        executed += 1
        try:
            predictions = tuple(
                executor.execute(program, pair.input, capture_trace=False).value
                for pair in pairs
            )
        except ExecutionError:
            continue
        if predictions in seen_behaviors:
            continue
        seen_behaviors.add(predictions)
        targets = tuple(pair.output for pair in pairs)
        error = sum(
            _grid_error(item, target) for item, target in zip(predictions, targets)
        )
        if error == 0:
            traces, residuals, fits = replay_program(
                executor, program, pairs, task_id=task_id
            )
            if fits:
                hypothesis_id = stable_digest((task_id, name, program.digest))
                exact.append(
                    Hypothesis(
                        hypothesis_id=hypothesis_id,
                        program=program,
                        traces=traces,
                        residuals=residuals,
                        exact_replay=True,
                        complexity=program.complexity,
                        confidence=1.0,
                        channel="residual-repair",
                        parent_id=parent_id,
                        provenance={"seed": name, "repair_depth": depth},
                    )
                )
            continue
        if depth >= max_depth:
            continue
        for spec in _repair_specs(predictions, targets, error):
            child = _wrap(program, spec)
            queue.append(
                (
                    f"{name}+{spec.name}",
                    child,
                    depth + 1,
                    stable_digest((task_id, name, program.digest)),
                )
            )
    exact.sort(key=lambda item: (item.complexity, item.hypothesis_id))
    return exact


def exact_object_repair_hypotheses(
    task_data: Mapping[str, Any],
    *,
    task_id: str = "object-repair",
    max_depth: int = 2,
    max_candidates: int = 256,
) -> list[Hypothesis]:
    """Repair generated object programs without exposing non-exact candidates."""
    from hyper_arc.contender.object_programs import generate_object_programs

    pairs = tuple(
        GridPair(
            input=canonical_grid(pair["input"]),
            output=canonical_grid(pair["output"]),
        )
        for pair in task_data.get("train", [])
    )
    seeds = tuple(
        (name, program)
        for name, program, _penalty in generate_object_programs(task_data)
    )
    return repair_hypotheses(
        seeds,
        pairs,
        task_id=task_id,
        max_depth=max_depth,
        max_candidates=max_candidates,
    )

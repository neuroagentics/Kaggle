"""Closed relational plan generation, binding, and exact replay."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping

from hyper_arc.contender.executor import ExecutionError, TypedExecutor, input_node
from hyper_arc.contender.relational_primitives import normalized_frame_signal
from hyper_arc.contender.schemas import Grid, GridPair, ProgramAST, ValueType


@dataclass(frozen=True)
class RelationalCandidate:
    kind: str
    program: ProgramAST

    @cached_property
    def complexity(self) -> int:
        return self.program.complexity

    def predict(self, grid: list[list[int]] | Grid) -> list[list[int]]:
        result = TypedExecutor().execute(self.program, grid, capture_trace=False).value
        return [list(row) for row in result]


def _frame_repeat_arguments(task_data: Mapping[str, Any]) -> dict[str, Any] | None:
    first = task_data.get("train", [None])[0]
    if not isinstance(first, Mapping):
        return None
    try:
        motif, _background = normalized_frame_signal(
            tuple(tuple(row) for row in first["input"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    output = first["output"]
    motif_height, motif_width = len(motif), len(motif[0])
    positions = []
    for row in range(len(output) - motif_height + 1):
        for col in range(len(output[0]) - motif_width + 1):
            patch = tuple(
                tuple(values[col : col + motif_width])
                for values in output[row : row + motif_height]
            )
            if patch == motif:
                positions.append([row, col])
    if not positions:
        return None
    return {
        "height": len(output),
        "width": len(output[0]),
        "motif_height": motif_height,
        "motif_width": motif_width,
        "positions": positions,
    }


def generate_relational_programs(
    task_data: Mapping[str, Any],
) -> tuple[tuple[str, ProgramAST], ...]:
    source = input_node()
    programs = [
        (
            "edge-marker-motif-propagation",
            ProgramAST(
                op="edge_marker_motif_propagation",
                output_type=ValueType.GRID,
                children=(source,),
            ),
        ),
        (
            "keyed-object-frames",
            ProgramAST(
                op="keyed_object_frames",
                output_type=ValueType.GRID,
                children=(source,),
            ),
        ),
    ]
    arguments = _frame_repeat_arguments(task_data)
    if arguments is not None:
        programs.append(
            (
                "frame-signal-repeat",
                ProgramAST(
                    op="frame_signal_repeat",
                    output_type=ValueType.GRID,
                    arguments=arguments,
                    children=(source,),
                ),
            )
        )
    return tuple(programs)


def exact_relational_candidates(
    task_data: Mapping[str, Any],
) -> list[RelationalCandidate]:
    pairs = tuple(
        GridPair(
            input=tuple(tuple(row) for row in pair["input"]),
            output=tuple(tuple(row) for row in pair["output"]),
        )
        for pair in task_data.get("train", [])
    )
    if not pairs:
        return []
    executor = TypedExecutor()
    test_inputs = [pair["input"] for pair in task_data.get("test", [])]
    exact = []
    seen: set[tuple[Grid, ...]] = set()
    for kind, program in generate_relational_programs(task_data):
        try:
            if any(
                executor.execute(program, pair.input, capture_trace=False).value
                != pair.output
                for pair in pairs
            ):
                continue
            behavior = tuple(
                executor.execute(program, grid, capture_trace=False).value
                for grid in test_inputs
            )
        except (ExecutionError, ValueError):
            continue
        if behavior in seen:
            continue
        seen.add(behavior)
        exact.append(RelationalCandidate(kind=kind, program=program))
    exact.sort(key=lambda item: (item.complexity, item.kind))
    return exact

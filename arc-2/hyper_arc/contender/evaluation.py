"""Strict exact-match evaluation for ARC-AGI-2 pass@1 and pass@2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class EvaluationError(ValueError):
    """Raised when predictions cannot be scored without ambiguity."""


def output_size_family(task: Mapping[str, Any]) -> str:
    """Return a deterministic coarse family based only on demonstration shapes."""
    relations: list[str] = []
    for pair in task.get("train", []):
        before = pair.get("input", [])
        after = pair.get("output", [])
        if not _is_grid(before) or not _is_grid(after):
            raise EvaluationError("Cannot classify malformed training grids")
        before_shape = (len(before), len(before[0]))
        after_shape = (len(after), len(after[0]))
        if before_shape == after_shape:
            relations.append("same-size")
        elif after_shape[0] <= before_shape[0] and after_shape[1] <= before_shape[1]:
            relations.append("contract")
        elif after_shape[0] >= before_shape[0] and after_shape[1] >= before_shape[1]:
            relations.append("expand")
        elif before_shape[0] * before_shape[1] == after_shape[0] * after_shape[1]:
            relations.append("reshape")
        else:
            relations.append("mixed-axis")
    if not relations:
        return "no-demonstrations"
    return relations[0] if len(set(relations)) == 1 else "mixed-demonstrations"


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    return all(
        len(row) == width
        and all(
            isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9
            for cell in row
        )
        for row in value
    )


@dataclass(frozen=True)
class OutputExactResult:
    output_index: int
    attempt_1_exact: bool
    attempt_2_exact: bool
    pass_at_2: bool
    attempt_1_channel: str | None = None
    attempt_2_channel: str | None = None


@dataclass(frozen=True)
class TaskExactResult:
    task_id: str
    output_count: int
    pass_at_1_outputs: int
    pass_at_2_outputs: int
    task_pass_at_1: bool
    task_pass_at_2: bool
    outputs: tuple[OutputExactResult, ...]


@dataclass(frozen=True)
class ExactReport:
    task_count: int
    output_count: int
    pass_at_1_tasks: int
    pass_at_2_tasks: int
    pass_at_1_outputs: int
    pass_at_2_outputs: int
    task_pass_at_1: float
    task_pass_at_2: float
    output_pass_at_1: float
    output_pass_at_2: float
    tasks: tuple[TaskExactResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_submission(
    submission: Mapping[str, Any],
    solutions: Mapping[str, Any],
    channel_attribution: Mapping[str, Any] | None = None,
) -> ExactReport:
    """Score a complete two-attempt submission using exact grid equality.

    The task metric requires every test output in a task to be solved. The
    output metric is also returned because it exposes progress hidden by the
    all-or-nothing task metric.
    """
    if set(submission) != set(solutions):
        missing = sorted(set(solutions) - set(submission))
        extra = sorted(set(submission) - set(solutions))
        raise EvaluationError(
            f"Task IDs mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    task_results: list[TaskExactResult] = []
    for task_id in sorted(solutions):
        expected_outputs = solutions[task_id]
        attempts = submission[task_id]
        if not isinstance(expected_outputs, list) or not isinstance(attempts, list):
            raise EvaluationError(f"Task {task_id}: outputs must be lists")
        if len(attempts) != len(expected_outputs):
            raise EvaluationError(
                f"Task {task_id}: expected {len(expected_outputs)} outputs, "
                f"found {len(attempts)}"
            )

        output_results: list[OutputExactResult] = []
        for index, (entry, expected) in enumerate(zip(attempts, expected_outputs)):
            if not _is_grid(expected):
                raise EvaluationError(
                    f"Task {task_id} output {index}: invalid solution"
                )
            if not isinstance(entry, dict) or set(entry) != {
                "attempt_1",
                "attempt_2",
            }:
                raise EvaluationError(
                    f"Task {task_id} output {index}: expected ordered attempt fields"
                )
            first = entry["attempt_1"]
            second = entry["attempt_2"]
            if not _is_grid(first) or not _is_grid(second):
                raise EvaluationError(
                    f"Task {task_id} output {index}: malformed prediction grid"
                )
            first_exact = first == expected
            second_exact = second == expected
            channels = None
            if channel_attribution is not None:
                task_channels = channel_attribution.get(task_id)
                if not isinstance(task_channels, list) or len(task_channels) != len(
                    expected_outputs
                ):
                    raise EvaluationError(
                        f"Task {task_id}: channel attribution count mismatch"
                    )
                channels = task_channels[index]
                if not isinstance(channels, dict) or set(channels) != {
                    "attempt_1",
                    "attempt_2",
                }:
                    raise EvaluationError(
                        f"Task {task_id} output {index}: invalid channel attribution"
                    )
            output_results.append(
                OutputExactResult(
                    output_index=index,
                    attempt_1_exact=first_exact,
                    attempt_2_exact=second_exact,
                    pass_at_2=first_exact or second_exact,
                    attempt_1_channel=channels["attempt_1"] if channels else None,
                    attempt_2_channel=channels["attempt_2"] if channels else None,
                )
            )

        pass_1 = [result.attempt_1_exact for result in output_results]
        pass_2 = [result.pass_at_2 for result in output_results]

        task_results.append(
            TaskExactResult(
                task_id=task_id,
                output_count=len(expected_outputs),
                pass_at_1_outputs=sum(pass_1),
                pass_at_2_outputs=sum(pass_2),
                task_pass_at_1=all(pass_1),
                task_pass_at_2=all(pass_2),
                outputs=tuple(output_results),
            )
        )

    task_count = len(task_results)
    output_count = sum(result.output_count for result in task_results)
    pass_at_1_tasks = sum(result.task_pass_at_1 for result in task_results)
    pass_at_2_tasks = sum(result.task_pass_at_2 for result in task_results)
    pass_at_1_outputs = sum(result.pass_at_1_outputs for result in task_results)
    pass_at_2_outputs = sum(result.pass_at_2_outputs for result in task_results)
    return ExactReport(
        task_count=task_count,
        output_count=output_count,
        pass_at_1_tasks=pass_at_1_tasks,
        pass_at_2_tasks=pass_at_2_tasks,
        pass_at_1_outputs=pass_at_1_outputs,
        pass_at_2_outputs=pass_at_2_outputs,
        task_pass_at_1=pass_at_1_tasks / task_count if task_count else 0.0,
        task_pass_at_2=pass_at_2_tasks / task_count if task_count else 0.0,
        output_pass_at_1=pass_at_1_outputs / output_count if output_count else 0.0,
        output_pass_at_2=pass_at_2_outputs / output_count if output_count else 0.0,
        tasks=tuple(task_results),
    )


def build_ledger_entry(
    report: ExactReport,
    *,
    run_id: str,
    solver_id: str,
    dataset_id: str,
    channels: list[str],
    duration_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
    telemetry: Mapping[str, Any] | None = None,
    task_families: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one append-only, JSON-serializable experiment record."""
    channel_output_wins: dict[str, int] = {}
    channel_task_contributions: dict[str, int] = {}
    for task in report.tasks:
        contributing: set[str] = set()
        for output in task.outputs:
            winners = set()
            if output.attempt_1_exact and output.attempt_1_channel:
                winners.add(output.attempt_1_channel)
            if output.attempt_2_exact and output.attempt_2_channel:
                winners.add(output.attempt_2_channel)
            for winner in winners:
                channel_output_wins[winner] = channel_output_wins.get(winner, 0) + 1
                contributing.add(winner)
        if task.task_pass_at_2:
            for channel in contributing:
                channel_task_contributions[channel] = (
                    channel_task_contributions.get(channel, 0) + 1
                )
    family_metrics: dict[str, dict[str, int | float]] = {}
    if task_families is not None:
        missing = sorted({task.task_id for task in report.tasks} - set(task_families))
        if missing:
            raise EvaluationError(f"Missing task family labels: {missing[:5]}")
        for task in report.tasks:
            family = str(task_families[task.task_id])
            row = family_metrics.setdefault(
                family,
                {
                    "task_count": 0,
                    "output_count": 0,
                    "pass_at_1_tasks": 0,
                    "pass_at_2_tasks": 0,
                    "pass_at_1_outputs": 0,
                    "pass_at_2_outputs": 0,
                },
            )
            row["task_count"] += 1
            row["output_count"] += task.output_count
            row["pass_at_1_tasks"] += int(task.task_pass_at_1)
            row["pass_at_2_tasks"] += int(task.task_pass_at_2)
            row["pass_at_1_outputs"] += task.pass_at_1_outputs
            row["pass_at_2_outputs"] += task.pass_at_2_outputs
        for row in family_metrics.values():
            task_count = int(row["task_count"])
            output_count = int(row["output_count"])
            row["task_pass_at_1"] = row["pass_at_1_tasks"] / task_count
            row["task_pass_at_2"] = row["pass_at_2_tasks"] / task_count
            row["output_pass_at_1"] = row["pass_at_1_outputs"] / output_count
            row["output_pass_at_2"] = row["pass_at_2_outputs"] / output_count
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "solver_id": solver_id,
        "dataset_id": dataset_id,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "telemetry": dict(telemetry) if telemetry is not None else None,
        "metrics": {
            key: value for key, value in report.to_dict().items() if key != "tasks"
        },
        "channel_output_wins": channel_output_wins,
        "channel_task_contributions": channel_task_contributions,
        "family_metrics": family_metrics,
        "solved_task_ids": [
            task.task_id for task in report.tasks if task.task_pass_at_2
        ],
    }


def append_ledger(path: str | Path, entry: Mapping[str, Any]) -> None:
    """Append exactly one canonical JSON line to an experiment ledger."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
        handle.write("\n")

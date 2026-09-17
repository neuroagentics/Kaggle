"""Evaluate typed object-program policies on the nested development fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.evaluation import output_size_family
from hyper_arc.contender.executor import TypedExecutor
from hyper_arc.contender.object_programs import exact_object_candidates
from hyper_arc.contender.repair import exact_object_repair_hypotheses
from hyper_arc.contender.telemetry import ResourceMonitor
from hyper_arc.deterministic import exact_candidates, prediction_pair
from hyper_arc.verified_adapter import solve_verified


def _attempts(prediction_groups, input_grid):
    distinct = []
    for prediction in prediction_groups:
        if prediction not in distinct:
            distinct.append(prediction)
        if len(distinct) == 2:
            break
    while len(distinct) < 2:
        distinct.append([row[:] for row in input_grid])
    return distinct[0], distinct[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        choices=("validation", "holdout"),
        default="validation",
        help="Frozen partition to score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N validation tasks for diagnostics.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print non-sensitive progress after each N tasks (0 disables).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    builder_ids, validation_ids = ids[:640], ids[640:]
    digest = hashlib.sha256(
        json.dumps(
            {"builder": builder_ids, "validation": validation_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = json.loads(Path("config/arc2_split_v1.json").read_text(encoding="utf-8"))
    expected_digest = manifest["iteration_split"]["split_sha256"]
    if digest != expected_digest:
        raise SystemExit(
            f"Nested split mismatch: expected {expected_digest}, got {digest}"
        )
    partition_ids = (
        validation_ids if arguments.partition == "validation" else list(holdout)
    )
    selected_ids = partition_ids[: arguments.limit]
    solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(
            encoding="utf-8"
        )
    )
    solved = {
        name: set()
        for name in (
            "baseline",
            "object_first",
            "stable_object",
            "production",
            "production_stable_object",
            "repair_first",
            "production_repair",
        )
    }
    object_fit_tasks = 0
    object_candidate_count = 0
    repair_fit_tasks = 0
    repair_candidate_count = 0
    revised_candidate_count = 0
    task_families = {}
    started = time.perf_counter()
    with ResourceMonitor() as resource_monitor:
        for ordinal, task_id in enumerate(selected_ids, start=1):
            task = (development if arguments.partition == "validation" else holdout)[
                task_id
            ]
            train = [(pair["input"], pair["output"]) for pair in task["train"]]
            tests = [pair["input"] for pair in task["test"]]
            expected = solutions[task_id]
            task_families[task_id] = output_size_family(task)
            baseline = exact_candidates(train, tests)
            object_candidates = exact_object_candidates(task)
            repair_hypotheses = exact_object_repair_hypotheses(
                task, task_id=f"{arguments.partition}:object-repair"
            )
            verified_predictions, fitting_rules = solve_verified(task)
            object_fit_tasks += bool(object_candidates)
            object_candidate_count += len(object_candidates)
            repair_fit_tasks += bool(repair_hypotheses)
            repair_candidate_count += len(repair_hypotheses)
            revised_candidate_count += sum(
                int(hypothesis.provenance.get("repair_depth", 0)) > 0
                for hypothesis in repair_hypotheses
            )
            repair_executor = TypedExecutor()
            policies = {
                "baseline": [],
                "object_first": [],
                "stable_object": [],
                "production": [],
                "production_stable_object": [],
                "repair_first": [],
                "production_repair": [],
            }
            for index, grid in enumerate(tests):
                baseline_pair = prediction_pair(baseline, grid)
                object_predictions = [
                    candidate.predict(grid) for candidate in object_candidates
                ]
                repair_predictions = [
                    [
                        list(row)
                        for row in repair_executor.execute(
                            hypothesis.program, grid, capture_trace=False
                        ).value
                    ]
                    for hypothesis in repair_hypotheses
                ]
                baseline_predictions = [
                    candidate.predict(grid) for candidate in baseline
                ]
                policies["baseline"].append(baseline_pair)
                policies["object_first"].append(
                    _attempts(object_predictions + baseline_predictions, grid)
                )
                stable_groups = (
                    [baseline_pair[0]] + object_predictions + [baseline_pair[1]]
                )
                policies["stable_object"].append(_attempts(stable_groups, grid))
                production_predictions = (
                    list(verified_predictions[index]) + baseline_predictions
                    if fitting_rules
                    else baseline_predictions
                )
                production_pair = _attempts(production_predictions, grid)
                policies["production"].append(production_pair)
                policies["production_stable_object"].append(
                    _attempts(
                        [production_pair[0]]
                        + object_predictions
                        + [production_pair[1]],
                        grid,
                    )
                )
                policies["repair_first"].append(
                    _attempts(repair_predictions + baseline_predictions, grid)
                )
                policies["production_repair"].append(
                    _attempts(
                        [production_pair[0]]
                        + repair_predictions
                        + [production_pair[1]],
                        grid,
                    )
                )
            for name, attempts_by_output in policies.items():
                if attempts_by_output and all(
                    target in attempts
                    for target, attempts in zip(expected, attempts_by_output)
                ):
                    solved[name].add(task_id)
            if arguments.progress_every and ordinal % arguments.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "completed": ordinal,
                            "evaluated": len(selected_ids),
                            "object_fit_tasks": object_fit_tasks,
                            "object_candidate_count": object_candidate_count,
                            "repair_candidate_count": repair_candidate_count,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    telemetry = resource_monitor.result()
    family_metrics = {}
    for family in sorted(set(task_families.values())):
        members = {
            task_id
            for task_id, task_family in task_families.items()
            if task_family == family
        }
        family_metrics[family] = {
            "task_count": len(members),
            **{
                f"{name}_pass_at_2": len(task_ids & members)
                for name, task_ids in solved.items()
            },
        }
    report = {
        "partition": arguments.partition,
        "partition_sha256": (
            digest
            if arguments.partition == "validation"
            else manifest["training_split"]["split_sha256"]
        ),
        "builder_tasks": len(builder_ids),
        "validation_tasks": len(validation_ids),
        "holdout_tasks": len(holdout),
        "evaluated_tasks": len(selected_ids),
        "diagnostic_limit": arguments.limit,
        "object_fit_tasks": object_fit_tasks,
        "object_candidate_count": object_candidate_count,
        "repair_fit_tasks": repair_fit_tasks,
        "repair_candidate_count": repair_candidate_count,
        "revised_candidate_count": revised_candidate_count,
        "baseline_pass_at_2": len(solved["baseline"]),
        "object_first_pass_at_2": len(solved["object_first"]),
        "stable_object_pass_at_2": len(solved["stable_object"]),
        "object_first_unique": len(solved["object_first"] - solved["baseline"]),
        "object_first_regressions": len(solved["baseline"] - solved["object_first"]),
        "stable_object_unique": len(solved["stable_object"] - solved["baseline"]),
        "stable_object_regressions": len(solved["baseline"] - solved["stable_object"]),
        "production_pass_at_2": len(solved["production"]),
        "production_stable_object_pass_at_2": len(solved["production_stable_object"]),
        "production_stable_object_unique": len(
            solved["production_stable_object"] - solved["production"]
        ),
        "production_stable_object_regressions": len(
            solved["production"] - solved["production_stable_object"]
        ),
        "repair_first_pass_at_2": len(solved["repair_first"]),
        "repair_first_unique_vs_object": len(
            solved["repair_first"] - solved["object_first"]
        ),
        "production_repair_pass_at_2": len(solved["production_repair"]),
        "production_repair_unique": len(
            solved["production_repair"] - solved["production"]
        ),
        "production_repair_regressions": len(
            solved["production"] - solved["production_repair"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "telemetry": telemetry.to_dict(),
        "family_metrics": family_metrics,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

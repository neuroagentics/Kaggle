"""Evaluate relational-plan v2 against the production diagnostic ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.evaluation import output_size_family
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.contender.telemetry import ResourceMonitor
from hyper_arc.deterministic import exact_candidates
from hyper_arc.verified_adapter import solve_verified


DESIGN_CASES = {"c62e2108", "5adee1b2", "db118e2a"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        choices=("validation", "holdout", "public-evaluation"),
        default="validation",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _attempts(predictions, input_grid):
    distinct = []
    for prediction in predictions:
        if prediction not in distinct:
            distinct.append(prediction)
        if len(distinct) == 2:
            break
    while len(distinct) < 2:
        distinct.append([row[:] for row in input_grid])
    return distinct


def _load_partition(name: str):
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    manifest = json.loads(Path("config/arc2_split_v1.json").read_text(encoding="utf-8"))
    if name == "validation":
        tasks = {task_id: development[task_id] for task_id in list(development)[640:]}
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
        digest = manifest["iteration_split"]["split_sha256"]
    elif name == "holdout":
        tasks = holdout
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
        digest = manifest["training_split"]["split_sha256"]
    else:
        challenge_path = Path("data/arc-agi-2/arc-agi_evaluation_challenges.json")
        tasks = json.loads(challenge_path.read_text(encoding="utf-8"))
        solutions_path = Path("data/arc-agi-2/arc-agi_evaluation_solutions.json")
        digest = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    return tasks, json.loads(solutions_path.read_text(encoding="utf-8")), digest


def main() -> int:
    arguments = _arguments()
    tasks, solutions, digest = _load_partition(arguments.partition)
    selected = list(tasks)[: arguments.limit]
    solved = {name: set() for name in ("production", "relational", "combined")}
    output_count = 0
    distinct_attempts = 0
    relational_fit_tasks = 0
    relational_fit_ids = set()
    relational_candidates = 0
    family_by_task = {}
    with ResourceMonitor() as monitor:
        for ordinal, task_id in enumerate(selected, start=1):
            task = tasks[task_id]
            tests = [pair["input"] for pair in task["test"]]
            train = [(pair["input"], pair["output"]) for pair in task["train"]]
            expected = solutions[task_id]
            deterministic = exact_candidates(train, tests)
            verified, fitting = solve_verified(task)
            relational = exact_relational_candidates(task)
            relational_fit_tasks += bool(relational)
            if relational:
                relational_fit_ids.add(task_id)
            relational_candidates += len(relational)
            family_by_task[task_id] = output_size_family(task)
            policies = {name: [] for name in solved}
            for index, grid in enumerate(tests):
                baseline_predictions = [
                    candidate.predict(grid) for candidate in deterministic
                ]
                ranked = (
                    list(verified[index]) + baseline_predictions
                    if fitting
                    else baseline_predictions
                )
                production = _attempts(ranked, grid)
                relation_predictions = [
                    candidate.predict(grid) for candidate in relational
                ]
                relation_pair = _attempts(relation_predictions, grid)
                combined = _attempts(
                    [production[0], *relation_predictions, production[1]], grid
                )
                policies["production"].append(production)
                policies["relational"].append(relation_pair)
                policies["combined"].append(combined)
                output_count += 1
                distinct_attempts += combined[0] != combined[1]
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
                            "tasks": len(selected),
                            "relational_fit_tasks": relational_fit_tasks,
                            "combined_pass_at_2": len(solved["combined"]),
                        }
                    ),
                    flush=True,
                )
    telemetry = monitor.result()
    non_design = set(selected) - DESIGN_CASES
    family_metrics = {}
    for family in sorted(set(family_by_task.values())):
        members = {
            task_id
            for task_id, task_family in family_by_task.items()
            if task_family == family
        }
        family_metrics[family] = {
            "tasks": len(members),
            **{
                f"{name}_pass_at_2": len(task_ids & members)
                for name, task_ids in solved.items()
            },
        }
    report = {
        "partition": arguments.partition,
        "partition_sha256": digest,
        "tasks": len(selected),
        "outputs": output_count,
        "design_cases_present": sorted(set(selected) & DESIGN_CASES),
        "relational_fit_tasks": relational_fit_tasks,
        "relational_candidate_count": relational_candidates,
        "production_pass_at_2": len(solved["production"]),
        "relational_pass_at_2": len(solved["relational"]),
        "combined_pass_at_2": len(solved["combined"]),
        "combined_unique": sorted(solved["combined"] - solved["production"]),
        "combined_regressions": sorted(solved["production"] - solved["combined"]),
        "non_design_tasks": len(non_design),
        "non_design_relational_fits": len(non_design & relational_fit_ids),
        "non_design_combined_unique": sorted(
            (solved["combined"] - solved["production"]) & non_design
        ),
        "combined_distinct_attempt_outputs": distinct_attempts,
        "combined_attempt_diversity_rate": (
            distinct_attempts / output_count if output_count else 0.0
        ),
        "family_metrics": family_metrics,
        "telemetry": telemetry.to_dict(),
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

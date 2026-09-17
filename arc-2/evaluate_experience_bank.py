"""Production-aware and owned-channel ablation for experience memory v2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.experience_bank import ExperienceBank, experience_candidates
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.deterministic import exact_candidates, prediction_pair
from hyper_arc.verified_adapter import solve_verified


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        choices=("validation", "holdout", "public-evaluation"),
        default="validation",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--bank", type=Path, default=Path("hyper_arc/experience_bank_v2.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _pair(predictions, fallback):
    unique = []
    for prediction in predictions:
        if prediction not in unique:
            unique.append(prediction)
        if len(unique) == 2:
            break
    while len(unique) < 2:
        unique.append([row[:] for row in fallback])
    return unique


def _solved(expected, attempts):
    return bool(attempts) and all(output in pair for output, pair in zip(expected, attempts))


def main() -> int:
    arguments = _arguments()
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    if arguments.partition == "validation":
        tasks = development
        selected = ids[640:]
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
    elif arguments.partition == "holdout":
        tasks = holdout
        selected = list(holdout)
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
    else:
        tasks = json.loads(
            Path("data/arc-agi-2/arc-agi_evaluation_challenges.json").read_text(
                encoding="utf-8"
            )
        )
        selected = list(tasks)
        solutions_path = Path("data/arc-agi-2/arc-agi_evaluation_solutions.json")
    selected = selected[: arguments.limit]
    solutions = json.loads(
        solutions_path.read_text(encoding="utf-8")
    )
    bank = ExperienceBank.load(arguments.bank)
    solved = {name: set() for name in ("owned_baseline", "memory", "production", "production_memory")}
    fit_tasks = 0
    candidate_count = 0
    memory_second_distinct = 0
    memory_provenance = {}
    started = time.perf_counter()
    for ordinal, task_id in enumerate(selected, start=1):
        task = tasks[task_id]
        train = [(pair["input"], pair["output"]) for pair in task["train"]]
        tests = [pair["input"] for pair in task["test"]]
        expected = solutions[task_id]
        deterministic = exact_candidates(train, tests)
        relational = exact_relational_candidates(task)
        memory = experience_candidates(bank, task)
        memory_provenance[task_id] = [
            {
                "name": candidate.name,
                "family": candidate.family,
                "source": candidate.source,
                "confidence": candidate.confidence,
                "complexity": candidate.complexity,
            }
            for candidate in memory
        ]
        verified, rules = solve_verified(task)
        fit_tasks += bool(memory)
        candidate_count += len(memory)
        policies = {name: [] for name in solved}
        for index, input_grid in enumerate(tests):
            baseline = list(prediction_pair(deterministic, input_grid))
            relational_predictions = [candidate.predict(input_grid) for candidate in relational]
            owned = _pair([baseline[0], *relational_predictions, baseline[1]], input_grid)
            memory_predictions = [candidate.predict(index) for candidate in memory]
            memory_pair = _pair([owned[0], *memory_predictions, owned[1]], input_grid)
            production_predictions = list(verified[index]) if rules else []
            production = _pair([*production_predictions, *baseline], input_grid)
            production_memory = _pair(
                [production[0], *memory_predictions, *relational_predictions, production[1]],
                input_grid,
            )
            policies["owned_baseline"].append(owned)
            policies["memory"].append(memory_pair)
            policies["production"].append(production)
            policies["production_memory"].append(production_memory)
            memory_second_distinct += memory_pair[1] != memory_pair[0]
        for name, attempts in policies.items():
            if _solved(expected, attempts):
                solved[name].add(task_id)
        if arguments.progress_every and ordinal % arguments.progress_every == 0:
            print(json.dumps({"completed": ordinal, "total": len(selected), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    report = {
        "schema_version": 2,
        "bank_id": bank.bank_id,
        "partition": arguments.partition,
        "evaluated_tasks": len(selected),
        "memory_fit_tasks": fit_tasks,
        "memory_candidate_count": candidate_count,
        "memory_second_distinct": memory_second_distinct,
        **{f"{name}_pass_at_2": len(task_ids) for name, task_ids in solved.items()},
        "memory_unique_vs_owned": sorted(solved["memory"] - solved["owned_baseline"]),
        "memory_regressions_vs_owned": sorted(solved["owned_baseline"] - solved["memory"]),
        "production_memory_unique": sorted(solved["production_memory"] - solved["production"]),
        "production_memory_regressions": sorted(solved["production"] - solved["production_memory"]),
        "memory_unique_details": {
            task_id: memory_provenance[task_id]
            for task_id in sorted(solved["memory"] - solved["owned_baseline"])
        },
        "production_memory_unique_details": {
            task_id: memory_provenance[task_id]
            for task_id in sorted(solved["production_memory"] - solved["production"])
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

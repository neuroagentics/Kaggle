"""Measure exact full-task effects of procedural-memory candidate ranking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.deterministic import exact_candidates


def _pair(candidates, input_grid):
    predictions = []
    names = []
    for candidate in candidates:
        prediction = candidate.predict(input_grid)
        if prediction not in predictions:
            predictions.append(prediction)
            names.append(candidate.name)
        if len(predictions) == 2:
            break
    while len(predictions) < 2:
        predictions.append([row[:] for row in input_grid])
        names.append("input-fallback")
    return predictions, names


def _solved(task, expected, candidates):
    names = []
    for pair, output in zip(task["test"], expected):
        predictions, selected_names = _pair(candidates, pair["input"])
        if output not in predictions:
            return False, []
        names.extend(selected_names)
    return True, sorted(set(names))


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition", choices=("validation", "holdout", "public-evaluation"), default="validation"
    )
    parser.add_argument(
        "--bank", type=Path, default=Path("hyper_arc/procedural_memory_v1.json")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    if arguments.partition == "validation":
        tasks = {task_id: development[task_id] for task_id in list(development)[640:]}
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
    elif arguments.partition == "holdout":
        tasks = holdout
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
    else:
        tasks = json.loads(
            Path("data/arc-agi-2/arc-agi_evaluation_challenges.json").read_text(
                encoding="utf-8"
            )
        )
        solutions_path = Path("data/arc-agi-2/arc-agi_evaluation_solutions.json")
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    bank = ProceduralMemoryBank.load(arguments.bank)
    default_solved = {}
    memory_solved = {}
    oracle_solved = {}
    started = time.perf_counter()
    for task_id, task in tasks.items():
        train = [(pair["input"], pair["output"]) for pair in task["train"]]
        tests = [pair["input"] for pair in task["test"]]
        candidates = exact_candidates(train, tests)
        exact, names = _solved(task, solutions[task_id], candidates)
        if exact:
            default_solved[task_id] = names
        ranked = bank.rank(candidates, task)
        exact, names = _solved(task, solutions[task_id], ranked)
        if exact:
            memory_solved[task_id] = names
        winners = [
            candidate.name
            for candidate in candidates
            if all(
                candidate.predict(pair["input"]) == output
                for pair, output in zip(task["test"], solutions[task_id])
            )
        ]
        if winners:
            oracle_solved[task_id] = winners
    report = {
        "schema_version": 1,
        "partition": arguments.partition,
        "evaluated_tasks": len(tasks),
        "bank_id": bank.bank_id,
        "default_pass_at_2": len(default_solved),
        "memory_ranked_pass_at_2": len(memory_solved),
        "oracle_any_exact_candidate": len(oracle_solved),
        "memory_unique": sorted(set(memory_solved) - set(default_solved)),
        "memory_regressions": sorted(set(default_solved) - set(memory_solved)),
        "default_solved": default_solved,
        "memory_solved": memory_solved,
        "oracle_solved": oracle_solved,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

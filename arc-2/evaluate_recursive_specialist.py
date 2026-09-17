"""Exact validation gate for a trained recursive grid specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.recursive_specialist import (
    RecursiveGridSpecialist,
    SpecialistConfig,
)
from hyper_arc.deterministic import exact_candidates
from hyper_arc.verified_adapter import solve_verified


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=40)
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


def _production_attempts(task):
    tests = [pair["input"] for pair in task["test"]]
    train = [(pair["input"], pair["output"]) for pair in task["train"]]
    deterministic = exact_candidates(train, tests)
    verified_predictions, fitting_rules = solve_verified(task)
    attempts = []
    for index, grid in enumerate(tests):
        baseline = [candidate.predict(grid) for candidate in deterministic]
        ranked = (
            list(verified_predictions[index]) + baseline
            if fitting_rules
            else baseline
        )
        attempts.append(_attempts(ranked, grid))
    return attempts


def main() -> int:
    arguments = _arguments()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    config = SpecialistConfig(**metadata["config"])
    model = RecursiveGridSpecialist(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    validation_ids = ids[640:]
    digest = hashlib.sha256(
        json.dumps(
            {"builder": ids[:640], "validation": validation_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(
            encoding="utf-8"
        )
    )
    eligible = 0
    specialist_solved = 0
    production_solved = 0
    ensemble_solved = 0
    unique = 0
    regressions = 0
    started = time.perf_counter()
    for ordinal, task_id in enumerate(validation_ids, start=1):
        task = development[task_id]
        demonstrations = [
            (pair["input"], pair["output"]) for pair in task["train"]
        ]
        leave_one_out_exact = all(
            model.predict(
                [pair for index, pair in enumerate(demonstrations) if index != held_out],
                demonstrations[held_out][0],
            )
            == demonstrations[held_out][1]
            for held_out in range(len(demonstrations))
        )
        eligible += leave_one_out_exact
        production = _production_attempts(task)
        targets = solutions[task_id]
        production_exact = all(
            target in attempts for target, attempts in zip(targets, production)
        )
        production_solved += production_exact
        specialist_predictions = (
            [
                model.predict(demonstrations, pair["input"])
                for pair in task["test"]
            ]
            if leave_one_out_exact
            else []
        )
        specialist_exact = bool(specialist_predictions) and all(
            prediction == target
            for prediction, target in zip(specialist_predictions, targets)
        )
        specialist_solved += specialist_exact
        ensemble = [
            _attempts(
                [production[index][0], specialist_predictions[index], production[index][1]],
                task["test"][index]["input"],
            )
            if leave_one_out_exact
            else production[index]
            for index in range(len(task["test"]))
        ]
        ensemble_exact = all(
            target in attempts for target, attempts in zip(targets, ensemble)
        )
        ensemble_solved += ensemble_exact
        unique += ensemble_exact and not production_exact
        regressions += production_exact and not ensemble_exact
        if arguments.progress_every and ordinal % arguments.progress_every == 0:
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "eligible": eligible,
                        "specialist_solved": specialist_solved,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    report = {
        "partition": "nested-validation",
        "partition_sha256": digest,
        "checkpoint": str(arguments.checkpoint),
        "tasks": len(validation_ids),
        "leave_one_out_eligible_tasks": eligible,
        "specialist_pass_at_2": specialist_solved,
        "production_pass_at_2": production_solved,
        "ensemble_pass_at_2": ensemble_solved,
        "ensemble_unique": unique,
        "ensemble_regressions": regressions,
        "elapsed_seconds": time.perf_counter() - started,
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

"""Evaluate the exact deterministic channel on public ARC-AGI-2 solutions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hyper_arc.deterministic import exact_candidates, prediction_pair


def _fold(task_id: str) -> str:
    bucket = hashlib.sha256(task_id.encode("utf-8")).digest()[0] % 5
    return "validation" if bucket == 0 else "development"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenges",
        default="data/arc-agi-2/arc-agi_evaluation_challenges.json",
    )
    parser.add_argument(
        "--solutions",
        default="data/arc-agi-2/arc-agi_evaluation_solutions.json",
    )
    args = parser.parse_args()
    challenges = json.loads(Path(args.challenges).read_text(encoding="utf-8"))
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))

    results = {
        fold: {"tasks": 0, "demo_exact": 0, "pass1": 0, "pass2": 0}
        for fold in ("development", "validation", "all")
    }
    solved: list[str] = []
    for task_id, task in challenges.items():
        train = [(pair["input"], pair["output"]) for pair in task["train"]]
        tests = [pair["input"] for pair in task["test"]]
        candidates = exact_candidates(train, tests)
        actual = solutions[task_id]
        predictions = [prediction_pair(candidates, grid) for grid in tests]
        pass1 = bool(predictions) and all(
            first == expected for (first, _), expected in zip(predictions, actual)
        )
        pass2 = bool(predictions) and all(
            expected in (first, second)
            for (first, second), expected in zip(predictions, actual)
        )
        if pass2:
            solved.append(task_id)
        for fold in (_fold(task_id), "all"):
            row = results[fold]
            row["tasks"] += 1
            row["demo_exact"] += bool(candidates)
            row["pass1"] += pass1
            row["pass2"] += pass2

    for fold, row in results.items():
        tasks = row["tasks"]
        print(
            f"{fold:11} tasks={tasks:3} demo_exact={row['demo_exact']:3} "
            f"pass1={row['pass1']:3} ({row['pass1'] / tasks:.1%}) "
            f"pass2={row['pass2']:3} ({row['pass2'] / tasks:.1%})"
        )
    print("pass2_task_ids=" + ",".join(solved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

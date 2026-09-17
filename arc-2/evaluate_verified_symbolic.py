"""Evaluate the pinned external verified-symbolic ensemble on aggregate JSON.

Reference source:
https://github.com/tanmaybisen31/arc-agi-solver
Pinned commit: e151937e34c8b34f953833a0dab75797fc737ba4 (MIT)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REFERENCE = Path(__file__).parent / "reference" / "verified-symbolic"
if not REFERENCE.is_dir():
    raise SystemExit(
        "Missing reference/verified-symbolic; clone the pinned external solver first."
    )
sys.path.insert(0, str(REFERENCE))

from harness import benchmark, print_report  # noqa: E402
from registry import ALL  # noqa: E402


def load_aggregate(challenges_path: Path, solutions_path: Path) -> dict:
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    tasks = {}
    for task_id, task in challenges.items():
        train = [
            (
                np.asarray(pair["input"], dtype=int),
                np.asarray(pair["output"], dtype=int),
            )
            for pair in task["train"]
        ]
        test = [
            (np.asarray(pair["input"], dtype=int), np.asarray(output, dtype=int))
            for pair, output in zip(task["test"], solutions[task_id])
        ]
        tasks[task_id] = {"train": train, "test": test}
    return tasks


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
    tasks = load_aggregate(Path(args.challenges), Path(args.solutions))
    stats = benchmark(tasks, ALL)
    print(f"detectors={len(ALL)}")
    print_report("verified-symbolic", stats)
    print("solved_task_ids=" + ",".join(stats["solved_names"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

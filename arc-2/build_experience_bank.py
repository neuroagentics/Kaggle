"""Build the production Hyper-ARC experience bank from the frozen builder fold."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.experience_bank import mine_experience_bank


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyper_arc/experience_bank_v2.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    manifest = json.loads(Path("config/arc2_split_v1.json").read_text(encoding="utf-8"))
    builder_ids = list(development)[:640]
    if arguments.limit is not None:
        builder_ids = builder_ids[: arguments.limit]
    solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    bank = mine_experience_bank(
        {task_id: development[task_id] for task_id in builder_ids},
        {task_id: solutions[task_id] for task_id in builder_ids},
        split_sha256=manifest["iteration_split"]["split_sha256"],
        training_dataset_sha256=manifest["datasets"]["training_challenges"]["sha256"],
    )
    bank.save(arguments.output)
    outcomes = Counter(
        (record.family, record.demonstration_exact, record.test_exact)
        for record in bank.records
    )
    print(
        json.dumps(
            {
                "bank_id": bank.bank_id,
                "records": len(bank.records),
                "builder_tasks": len(builder_ids),
                "output": str(arguments.output),
                "elapsed_seconds": time.perf_counter() - started,
                "outcomes": {
                    f"{family}:demo={demo}:test={test}": count
                    for (family, demo, test), count in sorted(outcomes.items())
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Mine the Hyper-ARC procedural-memory bank from the frozen builder fold."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.procedural_memory import mine_procedural_memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyper_arc/procedural_memory_v1.json"),
    )
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    manifest = json.loads(Path("config/arc2_split_v1.json").read_text(encoding="utf-8"))
    builder_ids = list(development)[:640]
    if arguments.limit is not None:
        builder_ids = builder_ids[: arguments.limit]
    all_solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    bank = mine_procedural_memory(
        {task_id: development[task_id] for task_id in builder_ids},
        {task_id: all_solutions[task_id] for task_id in builder_ids},
        split_sha256=manifest["iteration_split"]["split_sha256"],
        training_dataset_sha256=manifest["datasets"]["training_challenges"]["sha256"],
    )
    bank.save(arguments.output)
    family_successes = Counter()
    for record in bank.records:
        family_successes[record.family] += record.exact_successes
    print(
        json.dumps(
            {
                "bank_id": bank.bank_id,
                "builder_tasks": len(builder_ids),
                "procedures": len(bank.records),
                "exact_success_episodes": sum(
                    record.exact_successes for record in bank.records
                ),
                "transfer_failure_episodes": sum(
                    record.transfer_failures for record in bank.records
                ),
                "family_successes": dict(sorted(family_successes.items())),
                "output": str(arguments.output),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

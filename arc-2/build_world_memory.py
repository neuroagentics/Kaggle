"""Mine a frozen executable world-memory index from the builder fold only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.experience_bank import task_fingerprint_v2
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyper_arc/world_memory_v1.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/world_memory_build_v1.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-programs-per-task", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    development, _holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    builder_ids = list(development)[:640]
    selected = builder_ids[: arguments.limit]
    memory = HyperbolicWorldMemory()
    model = RecursiveReasoningWorldModel(
        config=WorldModelConfig(
            max_candidates=arguments.max_candidates,
            retrieve_memory=False,
            learn_memory=False,
        )
    )
    fit_tasks = 0
    recursive_fit_tasks = 0
    source_counts: dict[str, int] = {}
    started = time.perf_counter()
    for ordinal, task_id in enumerate(selected, start=1):
        task = development[task_id]
        result = model.solve(task_id, task)
        fit_tasks += bool(result.hypotheses)
        fingerprint = task_fingerprint_v2(task)
        selected_programs = result.hypotheses[: arguments.max_programs_per_task]
        recursive_fit_tasks += any(
            hypothesis.provenance.get("repair_depth", 0) > 0
            for hypothesis in selected_programs
        )
        for hypothesis in selected_programs:
            source = str(hypothesis.provenance.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            memory.remember(
                source_task_id=task_id,
                family=source,
                fingerprint=fingerprint,
                program=hypothesis.program,
                demonstration_exact=True,
                test_exact=False,
                provenance={
                    "source": "builder-fold-exact-demonstration",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "repair_depth": hypothesis.provenance.get("repair_depth", 0),
                    "invariants": ["exact-demonstration-replay"],
                },
            )
        if arguments.progress_every and ordinal % arguments.progress_every == 0:
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "total": len(selected),
                        "fit_tasks": fit_tasks,
                        "memory_programs": len(memory.items),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    memory.save(arguments.output)
    report = {
        "schema_version": 1,
        "builder_tasks": len(selected),
        "builder_fold_total": len(builder_ids),
        "diagnostic_limit": arguments.limit,
        "fit_tasks": fit_tasks,
        "recursive_fit_tasks": recursive_fit_tasks,
        "memory_programs": len(memory.items),
        "source_counts": dict(sorted(source_counts.items())),
        "memory_path": str(arguments.output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

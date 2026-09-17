"""Fail-closed runtime sentinel for an extracted Hyper-ARC bundle."""

from __future__ import annotations

import json
from pathlib import Path

from hyper_arc.contender.experience_bank import ExperienceBank
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.hpm import GlobalMemoryBank
from main import solve_task, validate_submission


ROOT = Path(__file__).resolve().parent


def main() -> int:
    seed_path = ROOT / "hyper_arc" / "seed_bank.json"
    experience_path = ROOT / "hyper_arc" / "experience_bank_v2.json"
    world_path = ROOT / "hyper_arc" / "world_memory_v1.json"

    seed_memory = GlobalMemoryBank(k=5)
    seed_memory.load(seed_path)
    if not len(seed_memory):
        raise RuntimeError("Required legacy seed memory is empty")

    experience = ExperienceBank.load(experience_path)
    if not experience.records:
        raise RuntimeError("Required experience memory is empty")

    world_memory = HyperbolicWorldMemory.load(world_path)
    if not world_memory.items:
        raise RuntimeError("Required world-memory artifact is empty")

    fixture = {
        "train": [
            {"input": [[1, 0]], "output": [[1, 0]]},
            {"input": [[2, 0]], "output": [[2, 0]]},
        ],
        "test": [{"input": [[3, 0]]}],
    }
    attempts, score, exact, _ = solve_task(
        "bundle-sentinel",
        fixture,
        seed_memory,
        global_prior_weight=0.2,
        timeout_sec=1.0,
        max_iterations=1,
        experience_bank=experience,
        world_model=None,
    )
    validate_submission({"bundle-sentinel": attempts}, {"bundle-sentinel": fixture})
    if not exact or score != 1.0:
        raise RuntimeError("Deterministic exact-replay sentinel failed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "seed_records": len(seed_memory),
                "experience_records": len(experience.records),
                "world_records": len(world_memory.items),
                "deterministic_exact": exact,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

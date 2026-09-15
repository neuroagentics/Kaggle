"""build_global_memory.py — Offline seed bank generator.

This is a STANDALONE OFFLINE UTILITY. It is NOT a dependency of main.py.
Run it separately on Kaggle Notebooks (or locally) after acquiring training
data to generate hyper_arc/seed_bank.json.

Usage:
    python build_global_memory.py \\
        --data-dir ./data/arc-agi-2 \\
        --n-tasks  50 \\
        --output   hyper_arc/seed_bank.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

from hyper_arc.esb import ESB
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSEngine

SHALLOW_BUDGET = 500   # iterations per task during seed bank construction


def log(msg: str) -> None:
    print(msg, flush=True)


def load_tasks(data_dir: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for fpath in sorted(data_dir.glob("**/*.json")):
        try:
            tasks[fpath.stem] = json.loads(fpath.read_text())
        except Exception:
            pass
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the GlobalMemoryBank seed bank from ARC training data."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/arc-agi-2",
        help="Path to ARC training task JSON files.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=50,
        help="Number of tasks to sample (default: 50).",
    )
    parser.add_argument(
        "--output",
        default="hyper_arc/seed_bank.json",
        help="Output path for the seed bank JSON.",
    )
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)
    n_tasks     = args.n_tasks

    if not data_dir.exists():
        log(f"[ERROR] data-dir not found: {data_dir}")
        return 1

    tasks = load_tasks(data_dir)
    if not tasks:
        log(f"[ERROR] No task JSON files found in {data_dir}")
        return 1

    # Sample a random subset
    task_ids = list(tasks.keys())
    random.shuffle(task_ids)
    sampled  = task_ids[:n_tasks]

    log(f"[INFO] {len(tasks)} tasks found. Sampling {len(sampled)} for seed bank.")

    global_bank = GlobalMemoryBank(k=5)
    attempted = 0
    solved    = 0

    for task_id in sampled:
        task = tasks[task_id]
        attempted += 1
        try:
            local_memory = LocalTaskBuffer(k=5)
            # Shallow MCTS with reduced budget
            from hyper_arc import mcts as mcts_module
            orig_max = mcts_module.MAX_ITERATIONS
            mcts_module.MAX_ITERATIONS = SHALLOW_BUDGET

            engine = MCTSEngine(
                global_memory=global_bank,
                local_memory=local_memory,
                global_prior_weight=0.0,   # no seed bank yet
            )
            train_pairs = [
                (ESB.from_grid(p["input"]), ESB.from_grid(p["output"]))
                for p in task.get("train", [])
            ]
            program = engine.solve(train_pairs)

            mcts_module.MAX_ITERATIONS = orig_max

            if program:
                global_bank.add(program)
                solved += 1
                log(f"[SOLVED] {task_id}  program_len={len(program)}  "
                    f"({solved} embedded so far)")
            else:
                log(f"[UNSOLVED] {task_id}")

        except Exception as exc:
            log(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc(file=sys.stdout)

    log(f"\n[SUMMARY] attempted={attempted}  solved={solved}  "
        f"programs_embedded={len(global_bank)}")

    if len(global_bank) == 0:
        log("[WARN] No programs solved. Seed bank will be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_bank.save(output_path)
    log(f"[INFO] Seed bank written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

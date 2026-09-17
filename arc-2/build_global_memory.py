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
import sys
import traceback
from pathlib import Path

from hyper_arc.esb import ESB
from hyper_arc.deterministic import exact_candidates
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSEngine
from hyper_arc.spatial_dsl import DSLProgram, SpatialDSL

SHALLOW_BUDGET = 500  # default iterations per task during seed bank construction


def log(msg: str) -> None:
    print(msg, flush=True)


def load_tasks(data_dir: Path) -> dict[str, dict]:
    """Load aggregate ARC training challenges or individual task files."""
    aggregate = sorted(data_dir.glob("**/arc-agi_training_challenges.json"))
    if aggregate:
        data = json.loads(aggregate[0].read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected task mapping in {aggregate[0]}")
        return data

    tasks: dict[str, dict] = {}
    for fpath in sorted(data_dir.glob("**/*.json")):
        try:
            task = json.loads(fpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(task, dict) and isinstance(task.get("train"), list):
            tasks[fpath.stem] = task
    return tasks


def deterministic_program(name: str) -> DSLProgram | None:
    """Translate a safe subset of exact-candidate names into the legacy DSL."""
    if name.endswith("+remap") or name.startswith("crop_mode+"):
        return None
    parts = name.split("+")
    program: DSLProgram = []
    for part in parts:
        if part == "crop0":
            program.append(("crop_to_bbox", {"background": 0}))
        elif part == "identity":
            continue
        elif part.startswith("rotate"):
            program.append(("rotate", {"degrees": int(part.removeprefix("rotate"))}))
        elif part == "reflect_horizontal":
            program.append(("reflect", {"axis": "horizontal"}))
        elif part == "reflect_vertical":
            program.append(("reflect", {"axis": "vertical"}))
        elif part.startswith("upscale"):
            program.append(
                ("scale_integer", {"factor": int(part.removeprefix("upscale"))})
            )
        elif part.startswith("tile"):
            height, width = part.removeprefix("tile").split("x", maxsplit=1)
            program.append(
                ("tile", {"repeats_h": int(height), "repeats_w": int(width)})
            )
        else:
            return None
    return program


def replays_exactly(program: DSLProgram, task: dict) -> bool:
    dsl = SpatialDSL()
    pairs = task.get("train", [])
    return bool(pairs) and all(
        dsl.apply_program(ESB.from_grid(pair["input"]), program).to_grid()
        == pair["output"]
        for pair in pairs
    )


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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic task sampling/search seed (default: 0).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=SHALLOW_BUDGET,
        help=f"MCTS iterations per task (default: {SHALLOW_BUDGET}).",
    )
    parser.add_argument(
        "--strategy",
        choices=("deterministic", "hybrid", "mcts"),
        default="hybrid",
        help="Seed-mining strategy (default: hybrid).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    n_tasks = args.n_tasks

    if not data_dir.exists():
        log(f"[ERROR] data-dir not found: {data_dir}")
        return 1

    tasks = load_tasks(data_dir)
    if not tasks:
        log(f"[ERROR] No task JSON files found in {data_dir}")
        return 1

    # Sample a deterministic subset
    task_ids = list(tasks.keys())
    import random

    rng = random.Random(args.seed)
    rng.shuffle(task_ids)
    sampled = task_ids[:n_tasks]

    log(f"[INFO] {len(tasks)} tasks found. Sampling {len(sampled)} for seed bank.")

    global_bank = GlobalMemoryBank(k=5)
    attempted = 0
    solved = 0
    seen_programs: set[str] = set()

    for task_id in sampled:
        task = tasks[task_id]
        attempted += 1
        try:
            if args.strategy in {"deterministic", "hybrid"}:
                raw_pairs = [
                    (pair["input"], pair["output"])
                    for pair in task.get("train", [])
                ]
                for candidate in exact_candidates(raw_pairs):
                    program = deterministic_program(candidate.name)
                    if program is None or not replays_exactly(program, task):
                        continue
                    digest = json.dumps(program, sort_keys=True, separators=(",", ":"))
                    if digest not in seen_programs:
                        global_bank.add(program)
                        seen_programs.add(digest)
                    solved += 1
                    log(
                        f"[SOLVED] {task_id} channel=deterministic "
                        f"program={candidate.name} ({len(global_bank)} unique seeds)"
                    )
                    break
                else:
                    program = None
                if program is not None:
                    continue

            if args.strategy == "deterministic":
                log(f"[UNSOLVED] {task_id} channel=deterministic")
                continue

            local_memory = LocalTaskBuffer(k=5)
            engine = MCTSEngine(
                global_memory=global_bank,
                local_memory=local_memory,
                global_prior_weight=0.2 if len(global_bank) else 0.0,
                max_iterations=args.max_iterations,
                random_seed=args.seed + attempted,
            )
            train_pairs = [
                (ESB.from_grid(p["input"]), ESB.from_grid(p["output"]))
                for p in task.get("train", [])
            ]
            program = engine.solve(train_pairs)
            score, exact = engine.score_program(program, train_pairs)

            if exact:
                digest = json.dumps(program, sort_keys=True, separators=(",", ":"))
                if digest not in seen_programs:
                    global_bank.add(program)
                    seen_programs.add(digest)
                solved += 1
                log(
                    f"[SOLVED] {task_id}  score={score:.4f}  program_len={len(program)}  "
                    f"({solved} embedded so far)"
                )
            else:
                log(f"[UNSOLVED] {task_id}  best_score={score:.4f}")

        except Exception as exc:
            log(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc(file=sys.stdout)

    log(
        f"\n[SUMMARY] attempted={attempted}  solved={solved}  "
        f"programs_embedded={len(global_bank)}"
    )

    if len(global_bank) == 0:
        log("[WARN] No programs solved. Seed bank will be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_bank.save(output_path)
    log(f"[INFO] Seed bank written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

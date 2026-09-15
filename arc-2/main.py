"""main.py — Hyper-ARC end-to-end solver.

Kaggle usage:
    !python main.py
    # or with explicit paths:
    !python main.py --data-dir /kaggle/input/arc-prize-2025 \\
                    --output /kaggle/working/submission.json

MVP note:
    seed_bank.json is optional. If missing, GlobalMemoryBank starts empty
    and global_prior_weight is set to 0.0 — MCTS relies 100% on
    LocalTaskBuffer priors and uniform random fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from hyper_arc.esb import ESB
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSEngine, DEFAULT_GLOBAL_PRIOR_WEIGHT
from hyper_arc.spatial_dsl import SpatialDSL

# ── Kaggle environment detection ──────────────────────────────────────────

IS_KAGGLE = Path("/kaggle/working/").exists()

DEFAULT_DATA_DIR   = "/kaggle/input/arc-prize-2025" if IS_KAGGLE else "./data/arc-agi-2"
DEFAULT_OUTPUT     = "/kaggle/working/submission.json" if IS_KAGGLE else "./submission.json"
DEFAULT_CHECKPOINT = "/kaggle/working/checkpoint.pt"   if IS_KAGGLE else "./checkpoint.pt"
DEFAULT_SEED_BANK  = "hyper_arc/seed_bank.json"

_DSL = SpatialDSL()


# ── Logging ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Print with immediate flush for live Kaggle Notebook cell output."""
    print(msg, flush=True)


# ── Data loading ──────────────────────────────────────────────────────────

def load_tasks(
    data_dir: Path,
    seed_bank_path: Path,
) -> dict[str, dict]:
    """Glob all *.json task files, skipping the seed bank."""
    tasks: dict[str, dict] = {}
    for fpath in sorted(data_dir.glob("**/*.json")):
        if fpath.resolve() == seed_bank_path.resolve():
            continue
        try:
            tasks[fpath.stem] = json.loads(fpath.read_text())
        except Exception as exc:
            log(f"[WARN] Could not parse {fpath}: {exc}")
    return tasks


# ── Checkpoint helpers ────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    submission:    dict[str, list],
    completed_ids: set[str],
    global_memory: GlobalMemoryBank,
) -> None:
    torch.save(
        {
            "submission":    submission,
            "completed_ids": completed_ids,
            "global_memory": global_memory.state_dict_entries(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    global_memory: GlobalMemoryBank,
) -> tuple[dict[str, list], set[str]]:
    """Load checkpoint; restore GlobalMemoryBank in-place.

    Returns ({}, set()) on any error (starts fresh).
    """
    try:
        ckpt = torch.load(path, weights_only=False)
        global_memory.restore_from_state_dict(ckpt.get("global_memory", []))
        completed = set(ckpt.get("completed_ids", set()))
        log(f"[RESUME] Loaded checkpoint: {len(completed)} tasks already done.")
        return ckpt.get("submission", {}), completed
    except Exception as exc:
        log(f"[WARN] Checkpoint load failed ({exc}); starting fresh.")
        global_memory._entries.clear()
        return {}, set()


# ── Seed bank loading (MVP graceful fallback) ─────────────────────────────

def load_seed_bank(
    path: Path,
    global_memory: GlobalMemoryBank,
) -> float:
    """Try to load seed bank; return the global_prior_weight to use.

    Returns 0.2 on success, 0.0 on missing/error (MVP mode).
    """
    try:
        global_memory.load_seed_bank(path)
        log(f"[INFO] Loaded seed bank: {len(global_memory)} programs from {path}")
        return DEFAULT_GLOBAL_PRIOR_WEIGHT
    except FileNotFoundError:
        log(
            "[WARN] seed_bank.json not found — GlobalMemoryBank empty, "
            "using uniform prior (global_prior_weight=0.0)"
        )
        return 0.0
    except Exception as exc:
        log(
            f"[WARN] seed_bank.json failed to load ({exc}) — "
            "GlobalMemoryBank empty, using uniform prior"
        )
        return 0.0


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Hyper-ARC solver")
    parser.add_argument("--data-dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed-bank",  default=DEFAULT_SEED_BANK)
    args = parser.parse_args()

    data_dir        = Path(args.data_dir)
    output_path     = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    seed_bank_path  = Path(args.seed_bank)

    log(
        f"[INFO] data_dir={data_dir}  output={output_path}  "
        f"checkpoint={checkpoint_path}  kaggle={IS_KAGGLE}"
    )

    # ── Initialise GlobalMemoryBank ───────────────────────────────────────
    global_memory = GlobalMemoryBank(k=5)

    # ── Resume from checkpoint (restores global_memory entries too) ───────
    if checkpoint_path.exists():
        submission, completed_ids = load_checkpoint(checkpoint_path, global_memory)
    else:
        submission:    dict[str, list] = {}
        completed_ids: set[str]        = set()

    # ── Load seed bank only if global_memory is still empty ───────────────
    global_prior_weight: float
    if not global_memory._entries:
        global_prior_weight = load_seed_bank(seed_bank_path, global_memory)
    else:
        # Restored from checkpoint — keep using the same weight as before
        global_prior_weight = DEFAULT_GLOBAL_PRIOR_WEIGHT if global_memory._entries else 0.0

    # ── Load tasks ────────────────────────────────────────────────────────
    tasks = load_tasks(data_dir, seed_bank_path)
    total = len(tasks)
    log(f"[INFO] Found {total} tasks. Already completed: {len(completed_ids)}.")

    # ── Main task loop ────────────────────────────────────────────────────
    for n, (task_id, task) in enumerate(tasks.items(), 1):
        if task_id in completed_ids:
            log(f"[SKIP] {task_id} ({n}/{total})")
            continue

        try:
            local_memory = LocalTaskBuffer(k=5)
            engine = MCTSEngine(
                global_memory=global_memory,
                local_memory=local_memory,
                global_prior_weight=global_prior_weight,
            )

            train_pairs = [
                (ESB.from_grid(p["input"]), ESB.from_grid(p["output"]))
                for p in task.get("train", [])
            ]
            program = engine.solve(train_pairs)
            exact   = bool(program)

            predictions: list = []
            for test_pair in task.get("test", []):
                inp  = ESB.from_grid(test_pair["input"])
                pred = _DSL.apply_program(inp, program) if program else inp
                predictions.append(pred.to_grid())

            submission[task_id] = predictions
            completed_ids.add(task_id)
            log(
                f"[TASK] {task_id}  exact_match={exact}  "
                f"program_len={len(program)}  ({n}/{total})"
            )

        except Exception as exc:
            log(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc(file=sys.stdout)
            submission[task_id] = []
            completed_ids.add(task_id)

        # Checkpoint after every task
        save_checkpoint(checkpoint_path, submission, completed_ids, global_memory)

    # ── Write final submission ────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, indent=2))
    log(f"[INFO] Wrote {output_path}  ({len(submission)} tasks)")

    # ── Clean up checkpoint ───────────────────────────────────────────────
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log(f"[INFO] Deleted checkpoint {checkpoint_path} (clean completion)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

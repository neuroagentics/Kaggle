"""export_winning_path.py — Export MCTS winning program to annotated JSON.

Runs the solver on a single ARC task and exports:
  - The winning DSL program (step-by-step operations)
  - The grid state after each step (for Blender animation)
  - Loop/redundancy warnings (e.g., rotate applied 4+ times in a row)

Usage:
    python export_winning_path.py --task-file ./data/arc-agi-2/some_task.json
    python export_winning_path.py --task-file ./data/arc-agi-2/some_task.json \\
        --output winning_path.json --seed-bank hyper_arc/seed_bank.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from hyper_arc.esb import ESB
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSEngine, DEFAULT_GLOBAL_PRIOR_WEIGHT
from hyper_arc.spatial_dsl import SpatialDSL

_DSL = SpatialDSL()


# ── Loop / redundancy detection ───────────────────────────────────────────

# How many consecutive identical primitive calls triggers a warning
LOOP_THRESHOLD = 3

# Pairs of operations that cancel each other out
CANCEL_PAIRS: list[tuple[str, str]] = [
    ("rotate",  "rotate"),    # 90+270 or 180+180
    ("reflect", "reflect"),   # same axis twice
    ("translate", "translate"),  # (dx,dy) + (-dx,-dy)
]


def _degrees_sum(steps: list[dict]) -> dict[str, int]:
    """Sum rotation degrees per consecutive run of rotate steps."""
    totals: dict[str, int] = {}
    i = 0
    while i < len(steps):
        if steps[i]["primitive"] == "rotate":
            run_start = i
            total = 0
            while i < len(steps) and steps[i]["primitive"] == "rotate":
                total += steps[i]["kwargs"].get("degrees", 0)
                i += 1
            key = f"rotate_run_{run_start}"
            totals[key] = total % 360
        else:
            i += 1
    return totals


def detect_warnings(steps: list[dict]) -> list[str]:
    """Analyse a program for illogical patterns; return human-readable warnings."""
    warnings: list[str] = []

    if not steps:
        return warnings

    # ── Count per-primitive usage ─────────────────────────────────────────
    counts = Counter(s["primitive"] for s in steps)
    for prim, count in counts.items():
        if prim == "rotate" and count >= 4:
            warnings.append(
                f"LOOP: 'rotate' called {count} times — "
                f"4×90° is identity; program may be cycling."
            )
        if prim in ("reflect", "symmetrize") and count >= 2:
            warnings.append(
                f"REDUNDANCY: '{prim}' called {count} times — "
                f"applying it twice on the same axis is identity."
            )

    # ── Check for net-zero rotation runs ─────────────────────────────────
    rotation_sums = _degrees_sum(steps)
    for key, net in rotation_sums.items():
        if net == 0:
            warnings.append(
                f"NET-ZERO ROTATION in {key}: "
                f"consecutive rotations sum to 0° (identity). "
                f"Remove these steps."
            )

    # ── Check for consecutive identical primitives ────────────────────────
    run_prim = steps[0]["primitive"]
    run_len  = 1
    for i in range(1, len(steps)):
        if steps[i]["primitive"] == run_prim:
            run_len += 1
            if run_len >= LOOP_THRESHOLD:
                warnings.append(
                    f"CONSECUTIVE LOOP: '{run_prim}' repeated "
                    f"{run_len} times starting at step {i - run_len + 1}."
                )
        else:
            run_prim = steps[i]["primitive"]
            run_len  = 1

    # ── Check for translate that immediately cancels ──────────────────────
    for i in range(len(steps) - 1):
        a, b = steps[i], steps[i + 1]
        if a["primitive"] == "translate" and b["primitive"] == "translate":
            adx = a["kwargs"].get("dx", 0)
            ady = a["kwargs"].get("dy", 0)
            bdx = b["kwargs"].get("dx", 0)
            bdy = b["kwargs"].get("dy", 0)
            if adx + bdx == 0 and ady + bdy == 0:
                warnings.append(
                    f"CANCEL: translate at step {i} "
                    f"({adx},{ady}) cancelled by step {i+1} "
                    f"({bdx},{bdy}) — net displacement zero."
                )

    return warnings


# ── Grid state capture ────────────────────────────────────────────────────

def _esb_to_serialisable(esb: ESB) -> list[list[int]]:
    return esb.to_grid()


def trace_program(
    start: ESB,
    program: list[tuple[str, dict]],
) -> list[dict]:
    """Execute program step by step, capturing grid state after each op."""
    steps: list[dict] = []
    state = start
    for i, (name, kwargs) in enumerate(program):
        # Serialise kwargs (ESB values aren't JSON-serialisable)
        safe_kwargs = {
            k: (v if not isinstance(v, ESB) else "<ESB>")
            for k, v in kwargs.items()
        }
        try:
            fn = getattr(_DSL, name)
            # overlay foreground must reference current state
            if name == "overlay":
                kwargs = {**kwargs, "foreground": state}
            new_state = fn(state, **kwargs)
            steps.append({
                "step":       i,
                "primitive":  name,
                "kwargs":     safe_kwargs,
                "grid_after": _esb_to_serialisable(new_state),
                "error":      None,
            })
            state = new_state
        except Exception as exc:
            steps.append({
                "step":       i,
                "primitive":  name,
                "kwargs":     safe_kwargs,
                "grid_after": _esb_to_serialisable(state),  # unchanged
                "error":      str(exc),
            })
    return steps


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export MCTS winning path to annotated JSON."
    )
    parser.add_argument(
        "--task-file",
        required=True,
        help="Path to a single ARC task JSON file.",
    )
    parser.add_argument(
        "--output",
        default="winning_path.json",
        help="Output JSON path (default: winning_path.json).",
    )
    parser.add_argument(
        "--seed-bank",
        default="hyper_arc/seed_bank.json",
        help="Optional seed bank path.",
    )
    args = parser.parse_args()

    task_path = Path(args.task_file)
    if not task_path.exists():
        print(f"[ERROR] Task file not found: {task_path}", flush=True)
        return 1

    task = json.loads(task_path.read_text())

    # ── Load memory ───────────────────────────────────────────────────────
    global_memory = GlobalMemoryBank(k=5)
    global_prior_weight = 0.0
    seed_path = Path(args.seed_bank)
    if seed_path.exists():
        try:
            global_memory.load_seed_bank(seed_path)
            global_prior_weight = DEFAULT_GLOBAL_PRIOR_WEIGHT
            print(f"[INFO] Seed bank loaded: {len(global_memory)} programs", flush=True)
        except Exception as exc:
            print(f"[WARN] Seed bank load failed ({exc}), using empty bank.", flush=True)

    local_memory = LocalTaskBuffer(k=5)
    engine = MCTSEngine(
        global_memory=global_memory,
        local_memory=local_memory,
        global_prior_weight=global_prior_weight,
    )

    # ── Solve ─────────────────────────────────────────────────────────────
    train_pairs = [
        (ESB.from_grid(p["input"]), ESB.from_grid(p["output"]))
        for p in task.get("train", [])
    ]
    print(f"[INFO] Solving '{task_path.stem}' with "
          f"{len(train_pairs)} training pairs ...", flush=True)

    program = engine.solve(train_pairs)
    print(f"[INFO] Program found: {len(program)} steps.", flush=True)

    if not program:
        print("[WARN] No program found — exporting empty path.", flush=True)

    # ── Trace grid states ─────────────────────────────────────────────────
    input_esb = train_pairs[0][0] if train_pairs else ESB.from_grid([[0]])
    steps = trace_program(input_esb, program)

    # ── Loop / redundancy warnings ────────────────────────────────────────
    warnings = detect_warnings(steps)
    if warnings:
        print(f"\n[AUDIT] {len(warnings)} warning(s) detected:", flush=True)
        for w in warnings:
            print(f"  ⚠  {w}", flush=True)
    else:
        print("[AUDIT] No loops or redundancies detected.", flush=True)

    # ── Build output document ─────────────────────────────────────────────
    output = {
        "task_id":       task_path.stem,
        "program_length": len(program),
        "warnings":      warnings,
        "initial_grid":  _esb_to_serialisable(input_esb),
        "steps":         steps,
        # Blender-friendly flat list of (primitive, kwargs) for keyframing
        "blender_keyframes": [
            {"frame": i + 1, "primitive": s["primitive"], "kwargs": s["kwargs"]}
            for i, s in enumerate(steps)
        ],
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"[INFO] Exported to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

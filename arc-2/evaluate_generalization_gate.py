"""Research-only augmentation diagnostic on already inspected regression sets.

Reports exact demo/test behavior without treating arbitrary color relabeling as
proof of the intended rule. No program is re-fitted on N-1 demonstrations, so
this is not leave-one-out induction. This script is not a deployment gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_arc.contender.executor import ExecutionError, TypedExecutor
from hyper_arc.contender.object_programs import generate_object_programs
from hyper_arc.contender.generalization import augmentation_consistency
from hyper_arc.contender.schemas import GridPair
from hyper_arc.contender.data_protocol import load_verified_split


def _grid(rows):
    return tuple(tuple(int(v) for v in row) for row in rows)


def _run(executor, program, grid):
    try:
        value = executor.execute(program, grid, capture_trace=False).value
    except (ExecutionError, ValueError, IndexError, RecursionError):
        return None
    if hasattr(value, "to_grid"):
        value = value.to_grid()
    try:
        return tuple(tuple(int(v) for v in row) for row in value)
    except (TypeError, ValueError):
        return None


def evaluate(challenges: dict, solutions: dict, *, max_programs: int = 4000) -> dict:
    executor = TypedExecutor()
    fit_programs = 0
    fit_test_correct = 0
    gate_pass = 0
    gate_pass_test_correct = 0
    true_solves_lost = 0
    tasks_pass2_raw = 0
    tasks_pass2_gated = 0
    tasks_with_fit = 0

    for tid, task in challenges.items():
        train = task.get("train", [])
        tests = task.get("test", [])
        if not train or not tests:
            continue
        pairs = tuple(GridPair(input=_grid(p["input"]), output=_grid(p["output"])) for p in train)
        targets = [_grid(t) for t in solutions[tid]]

        programs = generate_object_programs(task)
        if len(programs) > max_programs:
            programs = programs[:max_programs]

        # Collect programs that fit ALL demos.
        fitters = []  # (program, test_correct, gate_pass)
        for _name, program, _penalty in programs:
            demo_ok = all(_run(executor, program, p.input) == p.output for p in pairs)
            if not demo_ok:
                continue
            # test_correct: pass@1-style, the program's own prediction on each test input
            predictions = tuple(_run(executor, program, _grid(t["input"])) for t in tests)
            test_correct = list(predictions) == targets
            report = augmentation_consistency(program, pairs, executor=executor)
            fitters.append((predictions, test_correct, report.passes_checks))

        if not fitters:
            continue
        tasks_with_fit += 1
        for _prog, test_correct, passed in fitters:
            fit_programs += 1
            fit_test_correct += int(test_correct)
            if passed:
                gate_pass += 1
                gate_pass_test_correct += int(test_correct)
            elif test_correct:
                true_solves_lost += 1

        # Distinct behavior, not distinct program text. Each test input can
        # succeed via a different member of the two-attempt ensemble.
        def pass2(order):
            seen = set()
            selected = []
            for predictions, _correct, _passed in order:
                if predictions in seen:
                    continue
                seen.add(predictions)
                selected.append(predictions)
                if len(selected) == 2:
                    break
            return bool(selected) and all(any(p[i] == target for p in selected)
                                          for i, target in enumerate(targets))

        raw_order = fitters
        gated_order = sorted(fitters, key=lambda f: (not f[2],))  # gate-passers first
        tasks_pass2_raw += int(pass2(raw_order))
        tasks_pass2_gated += int(pass2(gated_order))

    def pct(n, d):
        return f"{100*n/d:.1f}%" if d else "n/a"

    return {
        "scope": "augmentation-diagnostic-not-generalization-proof",
        "total_tasks": len(challenges),
        "color_filter_enabled": False,
        "tasks_with_demo_fit": tasks_with_fit,
        "demo_fitting_programs": fit_programs,
        "raw_precision(test_correct/fit)": pct(fit_test_correct, fit_programs),
        "gate_passed_programs": gate_pass,
        "gated_precision(test_correct/gate_pass)": pct(gate_pass_test_correct, gate_pass),
        "true_solves_lost_to_gate": true_solves_lost,
        "task_pass2_raw": tasks_pass2_raw,
        "task_pass2_gated": tasks_pass2_gated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()

    dev, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    sol = json.loads(Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text())
    ids = list(dev)
    validation = {t: dev[t] for t in ids[640:800]}
    val_sol = {t: sol[t] for t in validation}

    report = {"validation": evaluate(validation, val_sol)}

    pub_c = Path("data/arc-agi-2/arc-agi_evaluation_challenges.json")
    pub_s = Path("data/arc-agi-2/arc-agi_evaluation_solutions.json")
    if pub_c.is_file() and pub_s.is_file():
        report["public_eval"] = evaluate(
            json.loads(pub_c.read_text()), json.loads(pub_s.read_text())
        )

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

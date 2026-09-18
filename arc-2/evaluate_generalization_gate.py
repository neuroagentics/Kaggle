"""Measure whether the generalization gate cuts spurious demo-fits without
losing real solves.

For each task, the object-program channel yields programs that reproduce every
training demo. For each such program we know (from the held-out solutions) two
facts:
  - demo_fit: it reproduced all train demos (always true here, by construction)
  - test_correct: its test prediction actually matches the true test output

The gate is a classifier over demo-fitting programs. We report, per split:
  - programs that fit demos (the raw acceptance signal)
  - how many of those are actually test-correct  (precision of raw signal)
  - how many the gate PASSES, and precision among gate-passed
  - true solves lost to the gate (test-correct programs the gate rejected)
  - task-level pass@2 before vs after gating (rank gate-passers first)

A good gate raises precision and task-level retention on the sealed holdout
while losing few/no real solves. Uses labels ONLY to score, never to gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_arc.contender.executor import ExecutionError, TypedExecutor
from hyper_arc.contender.object_programs import generate_object_programs
from hyper_arc.contender.generalization import generalization_gate
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
            test_correct = all(
                _run(executor, program, _grid(t["input"])) == targets[i]
                for i, t in enumerate(tests)
            )
            report = generalization_gate(program, pairs, executor=executor)
            fitters.append((program, test_correct, report.generalizes))

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

        # Task-level pass@2: raw = any of the first 2 distinct programs correct;
        # gated = rank gate-passers first, then the rest.
        def pass2(order):
            seen = set()
            picked = 0
            for prog, test_correct, _passed in order:
                key = prog.digest
                if key in seen:
                    continue
                seen.add(key)
                picked += 1
                if test_correct:
                    return True
                if picked >= 2:
                    return False
            return False

        raw_order = fitters
        gated_order = sorted(fitters, key=lambda f: (not f[2],))  # gate-passers first
        tasks_pass2_raw += int(pass2(raw_order))
        tasks_pass2_gated += int(pass2(gated_order))

    def pct(n, d):
        return f"{100*n/d:.1f}%" if d else "n/a"

    return {
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

"""Validation-only smoke and value evaluation for local typed model proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.executor import TypedExecutor
from hyper_arc.contender.model_proposer import (
    ProposalError,
    propose_exact_hypotheses,
    propose_exact_relational_hypotheses,
)
from hyper_arc.contender.telemetry import ResourceMonitor
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)
from hyper_arc.hpm import GlobalMemoryBank
from main import solve_task


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument(
        "--contract",
        choices=("wrapper-v1", "relational-v2"),
        default="wrapper-v1",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--selection",
        choices=("prefix", "production-unsolved"),
        default="prefix",
    )
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=2_400)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _attempts(predictions, input_grid):
    distinct = []
    for prediction in predictions:
        if prediction not in distinct:
            distinct.append(prediction)
        if len(distinct) == 2:
            break
    while len(distinct) < 2:
        distinct.append([row[:] for row in input_grid])
    return distinct


def _production_attempts(
    task_id: str,
    task: dict,
    procedural_memory: ProceduralMemoryBank,
):
    """Run only the current owned production channels; never a reference solver."""
    world_model = RecursiveReasoningWorldModel(
        HyperbolicWorldMemory(),
        config=WorldModelConfig(retrieve_memory=False, learn_memory=False),
    )
    attempts, _score, _exact, _complexity = solve_task(
        task_id,
        task,
        GlobalMemoryBank(k=5),
        0.0,
        0.01,
        1,
        experience_bank=None,
        world_model=world_model,
        procedural_memory=procedural_memory,
    )
    return [[entry["attempt_1"], entry["attempt_2"]] for entry in attempts]


def main() -> int:
    arguments = _arguments()
    if arguments.count < 1:
        raise SystemExit("--count must be positive")
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    validation_ids = ids[640:]
    digest = hashlib.sha256(
        json.dumps(
            {"builder": ids[:640], "validation": validation_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(
            encoding="utf-8"
        )
    )
    exact_fit_tasks = 0
    exact_hypotheses = 0
    requested_programs = 0
    parsed_programs = 0
    model_solved = 0
    production_solved = 0
    production_unique = 0
    failures: dict[str, int] = {}
    rejections: dict[str, int] = {}
    rejection_details: list[str] = []
    started = time.perf_counter()
    selected_ids = []
    precomputed: dict[str, list] = {}
    procedural_memory = ProceduralMemoryBank.load(
        Path("hyper_arc/procedural_memory_v1.json")
    )
    with ResourceMonitor() as resource_monitor:
        for task_id in validation_ids:
            if arguments.selection == "production-unsolved":
                attempts = _production_attempts(
                    task_id, development[task_id], procedural_memory
                )
                precomputed[task_id] = attempts
                if all(
                    target in options
                    for target, options in zip(solutions[task_id], attempts)
                ):
                    continue
            selected_ids.append(task_id)
            if len(selected_ids) == arguments.count:
                break
        for ordinal, task_id in enumerate(selected_ids, start=1):
            task = development[task_id]
            tests = [pair["input"] for pair in task["test"]]
            targets = solutions[task_id]
            production_attempts = precomputed.get(task_id) or _production_attempts(
                task_id, task, procedural_memory
            )
            production_exact = all(
                target in attempts
                for target, attempts in zip(targets, production_attempts)
            )
            production_solved += production_exact
            try:
                proposer = (
                    propose_exact_relational_hypotheses
                    if arguments.contract == "relational-v2"
                    else propose_exact_hypotheses
                )
                candidate_limit = (
                    min(arguments.max_candidates, 3)
                    if arguments.contract == "relational-v2"
                    else arguments.max_candidates
                )
                batch = proposer(
                    f"validation:{task_id}",
                    task,
                    model=arguments.model,
                    max_candidates=candidate_limit,
                    timeout_seconds=arguments.timeout,
                    max_output_tokens=arguments.max_output_tokens,
                )
            except ProposalError as exc:
                key = str(exc).replace("\n", " ")[:120]
                failures[key] = failures.get(key, 0) + 1
                print(
                    json.dumps({"completed": ordinal, "status": "proposal-error"}),
                    flush=True,
                )
                continue
            exact_fit_tasks += bool(batch.exact)
            exact_hypotheses += len(batch.exact)
            requested_programs += batch.requested
            parsed_programs += batch.parsed
            for rejected in batch.rejected:
                reason = rejected.split(":", 2)[1]
                rejections[reason] = rejections.get(reason, 0) + 1
                if len(rejection_details) < 12:
                    rejection_details.append(rejected)
            executor = TypedExecutor()
            model_attempts = []
            for grid in tests:
                predictions = [
                    [
                        list(row)
                        for row in executor.execute(
                            hypothesis.program, grid, capture_trace=False
                        ).value
                    ]
                    for hypothesis in batch.exact
                ]
                model_attempts.append(_attempts(predictions, grid))
            model_exact = bool(batch.exact) and all(
                target in attempts for target, attempts in zip(targets, model_attempts)
            )
            model_solved += model_exact
            production_unique += model_exact and not production_exact
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "exact_fit": bool(batch.exact),
                        "exact_hypotheses": len(batch.exact),
                        "test_exact": model_exact,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    telemetry = resource_monitor.result(failure_count=sum(failures.values()))
    report = {
        "partition": "nested-validation-owned-production-diagnostic",
        "comparison_solver": "hyper-arc-owned-production",
        "partition_sha256": digest,
        "model": arguments.model,
        "contract": arguments.contract,
        "selection": arguments.selection,
        "tasks": len(selected_ids),
        "selected_task_ids": selected_ids,
        "max_candidates": (
            min(arguments.max_candidates, 3)
            if arguments.contract == "relational-v2"
            else arguments.max_candidates
        ),
        "max_output_tokens": arguments.max_output_tokens,
        "exact_fit_tasks": exact_fit_tasks,
        "exact_hypotheses": exact_hypotheses,
        "requested_programs": requested_programs,
        "parsed_programs": parsed_programs,
        "rejections": rejections,
        "rejection_details": rejection_details,
        "model_pass_at_2": model_solved,
        "production_pass_at_2": production_solved,
        "model_unique_vs_production": production_unique,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "telemetry": telemetry.to_dict(),
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

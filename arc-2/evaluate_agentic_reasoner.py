"""Exact held-out evaluation for Hyper-ARC's execution-guided neural agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evaluate_model_proposer import _production_attempts
from hyper_arc.contender.agentic_memory import AgenticMemoryBank, AgenticMemoryRecord
from hyper_arc.contender.agentic_reasoner import AgenticReasoner
from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.failure_memory import (
    FailureMemoryBank,
    extract_failed_families,
)
from hyper_arc.contender.model_proposer import _ollama_transport
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.contender.telemetry import ResourceMonitor
from hyper_arc.contender.offline_model import OfflineTransformersTransport
from release_policy import AGENTIC_ROUNDS, AGENTIC_CANDIDATES, OUTPUT_TOKENS, MODEL_ID, MODEL_MIN_GPU_MEMORY_GIB


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Explicit model identity; no silent model substitution")
    parser.add_argument("--backend", choices=("transformers", "ollama"), default="transformers")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=AGENTIC_ROUNDS)
    parser.add_argument("--candidates", type=int, default=AGENTIC_CANDIDATES)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument("--ordering", choices=("dataset", "smallest"), default="dataset")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--partition", choices=("builder", "validation"), default="validation")
    parser.add_argument("--memory-output", type=Path)
    parser.add_argument(
        "--failure-memory",
        type=Path,
        help="Failure-memory bank: recalled to steer away from tried-and-failed "
        "approaches, and updated with this run's non-exact families.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _canonical(grid):
    return tuple(tuple(row) for row in grid)


def _grid_area(grid) -> int:
    return len(grid) * len(grid[0]) if grid else 0


def main() -> int:
    arguments = _arguments()
    if arguments.partition != "builder" and (arguments.memory_output or arguments.failure_memory):
        raise ValueError("Validation is read-only: reusable memory output/update requires --partition builder")
    if arguments.backend == "transformers" and not arguments.model_path:
        raise ValueError("Offline evaluation requires --model-path; Ollama is research-only")
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    builder_ids = ids[:640]
    validation_ids = ids[640:]
    partition_ids = builder_ids if arguments.partition == "builder" else validation_ids
    partition_digest = hashlib.sha256(
        json.dumps(
            {"builder": ids[:640], "validation": validation_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    solutions = json.loads(
        Path("data/arc-agi-2/arc-agi_training_solutions.json").read_text(
            encoding="utf-8"
        )
    )
    memory = ProceduralMemoryBank.load("hyper_arc/procedural_memory_v1.json")
    candidate_ids = list(arguments.task_ids or partition_ids)
    unknown = sorted(set(candidate_ids) - set(partition_ids))
    if unknown:
        raise ValueError(f"Task IDs are outside the frozen validation partition: {unknown}")
    if arguments.ordering == "smallest":
        candidate_ids.sort(
            key=lambda task_id: sum(
                _grid_area(pair["input"]) + _grid_area(pair.get("output"))
                for split in ("train", "test")
                for pair in development[task_id].get(split, [])
            )
        )
    selected: list[tuple[str, list]] = []
    for task_id in candidate_ids:
        attempts = _production_attempts(task_id, development[task_id], memory)
        if all(
            target in options
            for target, options in zip(solutions[task_id], attempts)
        ):
            continue
        selected.append((task_id, attempts))
        if len(selected) == arguments.count:
            break

    transport = (OfflineTransformersTransport(arguments.model_path,
        minimum_gpu_memory_gib=MODEL_MIN_GPU_MEMORY_GIB if arguments.model == MODEL_ID else 0.0) if arguments.backend == "transformers" else
        _ollama_transport(
            arguments.model,
            "http://127.0.0.1:11434/api/chat",
            arguments.timeout,
        ))
    reasoner = AgenticReasoner(
        transport,
        model=arguments.model,
        rounds=arguments.rounds,
        candidates_per_round=arguments.candidates,
        max_output_tokens=arguments.max_output_tokens,
        thinking=arguments.thinking,
    )
    failure_bank = FailureMemoryBank()
    if arguments.failure_memory and arguments.failure_memory.is_file():
        failure_bank = FailureMemoryBank.load(arguments.failure_memory)
    learned_failures: list = []

    solved = 0
    exact_outputs = 0
    exact_fit = 0
    generated = 0
    executed = 0
    failures: dict[str, int] = {}
    details = []
    learned_records = []
    started = time.perf_counter()
    with ResourceMonitor() as monitor:
        for ordinal, (task_id, production_attempts) in enumerate(selected, start=1):
            failure_cues = failure_bank.recall(development[task_id])
            try:
                result = reasoner.solve(
                    task_id,
                    development[task_id],
                    memory_cues=memory.retrieve_cues(development[task_id]),
                    failure_cues=failure_cues,
                    deadline=time.monotonic() + arguments.timeout,
                )
            except Exception as exc:
                key = f"{type(exc).__name__}:{str(exc)[:100]}"
                failures[key] = failures.get(key, 0) + 1
                print(json.dumps({"completed": ordinal, "status": "agent-error"}), flush=True)
                continue
            exact_fit += bool(result.hypotheses)
            generated += result.candidates_generated
            executed += result.candidates_executed
            targets = [_canonical(grid) for grid in solutions[task_id]]
            exact_outputs += sum(target in options for target, options in zip(targets, result.test_predictions))
            test_exact = bool(result.hypotheses) and all(
                target in options
                for target, options in zip(targets, result.test_predictions)
            )
            solved += test_exact
            if test_exact:
                for hypothesis in result.hypotheses:
                    # Ensemble pass@2 does not prove that EACH member solved
                    # every test. Persist only individually verified programs.
                    if list(hypothesis.test_predictions) != targets:
                        continue
                    learned_records.append(
                        AgenticMemoryRecord.create(
                            source_task_id=task_id,
                            summary=hypothesis.summary,
                            code=hypothesis.code,
                            model=arguments.model,
                            task_data=development[task_id],
                            validation_scope=arguments.partition,
                            expected_test_outputs=solutions[task_id],
                        )
                    )
            else:
                # Persist only genuine executed-but-non-exact approach families
                # so future runs on similar tasks avoid them and try new ones.
                learned_failures.extend(
                    extract_failed_families(
                        task_id, development[task_id], result.failures
                    )
                )
            details.append(
                {
                    "task_id": task_id,
                    "demonstration_exact": bool(result.hypotheses),
                    "test_exact": test_exact,
                    "rounds": result.rounds_executed,
                    "generated": result.candidates_generated,
                    "executed": result.candidates_executed,
                    "hypotheses": [
                        {
                            "digest": item.digest,
                            "summary": item.summary,
                            "complexity": item.complexity,
                            "round": item.round_index,
                            "code": item.code,
                        }
                        for item in result.hypotheses
                    ],
                    "rejections": list(result.failures[:12]),
                    "owned_production_exact": all(
                        target in options
                        for target, options in zip(solutions[task_id], production_attempts)
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "demo_exact": bool(result.hypotheses),
                        "test_exact": test_exact,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    telemetry = monitor.result(failure_count=sum(failures.values()))
    report = {
        "schema_version": 1,
        "partition": f"nested-{arguments.partition}-owned-production-unsolved",
        "partition_sha256": partition_digest,
        "model": arguments.model,
        "backend": arguments.backend,
        "model_path": str(arguments.model_path) if arguments.model_path else None,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_output_tokens": arguments.max_output_tokens,
        "task_budget_seconds": arguments.timeout,
        "selected_task_ids": [task_id for task_id, _ in selected],
        "exact_test_outputs_pass_at_2": exact_outputs,
        "total_test_outputs": sum(len(development[tid]["test"]) for tid, _ in selected),
        "agent": "execution-guided-code-refinement-v1",
        "tasks": len(selected),
        "rounds": arguments.rounds,
        "candidates_per_round": arguments.candidates,
        "ordering": arguments.ordering,
        "thinking": arguments.thinking,
        "demonstration_exact_tasks": exact_fit,
        "full_task_exact_pass_at_2": solved,
        "unique_vs_owned_production": solved,
        "candidates_generated": generated,
        "candidates_executed": executed,
        "failures": failures,
        "details": details,
        "elapsed_seconds": time.perf_counter() - started,
        "telemetry": telemetry.to_dict(),
    }
    if arguments.memory_output:
        existing = (
            AgenticMemoryBank.load(arguments.memory_output)
            if arguments.memory_output.is_file()
            else AgenticMemoryBank()
        )
        merged = existing.merged(learned_records)
        merged.save(arguments.memory_output)
        report["agentic_memory"] = {
            "path": str(arguments.memory_output),
            "new_records": len(learned_records),
            "total_records": len(merged.records),
        }
    if arguments.failure_memory:
        merged_failures = failure_bank.merged(learned_failures)
        merged_failures.save(arguments.failure_memory)
        report["failure_memory"] = {
            "path": str(arguments.failure_memory),
            "recalled_and_avoided": True,
            "new_failed_families": len(learned_failures),
            "total_failed_families": len(merged_failures.records),
        }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

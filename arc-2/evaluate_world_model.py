"""Clean exact-match gate for the owned recursive world-model channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.experience_bank import ExperienceBank
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)
from hyper_arc.contender.schemas import program_to_dict
from hyper_arc.deterministic import exact_candidates, prediction_pair


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        choices=("validation", "holdout", "public-evaluation"),
        default="validation",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--bank", type=Path, default=Path("hyper_arc/experience_bank_v2.json")
    )
    parser.add_argument(
        "--world-memory",
        type=Path,
        help="Frozen HyperbolicWorldMemory file; overrides --bank.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--task-id")
    parser.add_argument(
        "--procedural-memory",
        type=Path,
        help="Builder-only procedure evidence used to rank exact-replay candidates.",
    )
    return parser.parse_args()


def _load_partition(name: str):
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    manifest = json.loads(Path("config/arc2_split_v1.json").read_text(encoding="utf-8"))
    if name == "validation":
        tasks = {task_id: development[task_id] for task_id in list(development)[640:]}
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
        digest = manifest["iteration_split"]["split_sha256"]
    elif name == "holdout":
        tasks = holdout
        solutions_path = Path("data/arc-agi-2/arc-agi_training_solutions.json")
        digest = manifest["training_split"]["split_sha256"]
    else:
        challenge_path = Path("data/arc-agi-2/arc-agi_evaluation_challenges.json")
        tasks = json.loads(challenge_path.read_text(encoding="utf-8"))
        solutions_path = Path("data/arc-agi-2/arc-agi_evaluation_solutions.json")
        digest = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    return tasks, json.loads(solutions_path.read_text(encoding="utf-8")), digest


def _grid(value):
    return tuple(tuple(row) for row in value)


def _pair(predictions, fallback):
    distinct = []
    for prediction in predictions:
        normalized = _grid(prediction)
        if normalized not in distinct:
            distinct.append(normalized)
        if len(distinct) == 2:
            break
    fallback = _grid(fallback)
    while len(distinct) < 2:
        distinct.append(fallback)
    return tuple(distinct)


def _solved(expected, attempts):
    return bool(attempts) and all(
        _grid(output) in pair for output, pair in zip(expected, attempts)
    )


def main() -> int:
    arguments = _arguments()
    tasks, solutions, digest = _load_partition(arguments.partition)
    if arguments.task_id:
        if arguments.task_id not in tasks:
            raise SystemExit(
                f"Task is not in {arguments.partition}: {arguments.task_id}"
            )
        selected = [arguments.task_id]
    else:
        selected = list(tasks)[: arguments.limit]
    memory = (
        HyperbolicWorldMemory.load(arguments.world_memory)
        if arguments.world_memory
        else HyperbolicWorldMemory.from_experience_bank(
            ExperienceBank.load(arguments.bank)
        )
    )
    model = RecursiveReasoningWorldModel(
        memory,
        config=WorldModelConfig(
            max_candidates=arguments.max_candidates,
            max_depth=arguments.max_depth,
            retrieve_memory=not arguments.no_memory,
            # Evaluation partitions use a frozen index. Their demonstrations
            # may guide task-local reasoning but never alter later tasks.
            learn_memory=False,
        ),
    )
    procedural_memory = (
        ProceduralMemoryBank.load(arguments.procedural_memory)
        if arguments.procedural_memory
        else None
    )
    solved = {name: set() for name in ("owned_baseline", "world_model", "combined")}
    exact_fit_tasks = 0
    executed_candidates = 0
    retrieved_seed_tasks = 0
    recursive_fit_tasks = 0
    distinct_world_outputs = 0
    solved_details = {}
    started = time.perf_counter()
    for ordinal, task_id in enumerate(selected, start=1):
        task = tasks[task_id]
        train = [(pair["input"], pair["output"]) for pair in task["train"]]
        tests = [pair["input"] for pair in task["test"]]
        expected = solutions[task_id]
        deterministic = exact_candidates(train, tests)
        if procedural_memory is not None:
            deterministic = procedural_memory.rank(deterministic, task)
        relational = exact_relational_candidates(task)
        result = model.solve(task_id, task)
        exact_fit_tasks += bool(result.hypotheses)
        executed_candidates += result.candidates_executed
        retrieved_seed_tasks += any(
            state.source == "hyperbolic-memory" for state in result.reasoning_states
        )
        recursive_fit_tasks += any(
            hypothesis.provenance.get("repair_depth", 0) > 0
            for hypothesis in result.hypotheses
        )
        policies = {name: [] for name in solved}
        for index, input_grid in enumerate(tests):
            deterministic_pair = prediction_pair(deterministic, input_grid)
            relational_predictions = [
                candidate.predict(input_grid) for candidate in relational
            ]
            baseline = _pair(
                [deterministic_pair[0], *relational_predictions, deterministic_pair[1]],
                input_grid,
            )
            world = _pair(result.test_predictions[index], input_grid)
            combined = _pair(
                [baseline[0], *result.test_predictions[index], baseline[1]], input_grid
            )
            policies["owned_baseline"].append(baseline)
            policies["world_model"].append(world)
            policies["combined"].append(combined)
            distinct_world_outputs += world[0] != world[1]
        for name, attempts in policies.items():
            if _solved(expected, attempts):
                solved[name].add(task_id)
        if _solved(expected, policies["world_model"]):
            winning_hypotheses = []
            for hypothesis in result.hypotheses:
                predictions = [
                    model.executor.execute(
                        hypothesis.program, grid, capture_trace=False
                    ).value
                    for grid in tests
                ]
                if all(
                    _grid(value) == _grid(target)
                    for value, target in zip(predictions, expected)
                ):
                    winning_hypotheses.append(
                        {
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "confidence": hypothesis.confidence,
                            "complexity": hypothesis.complexity,
                            "provenance": dict(hypothesis.provenance),
                            "program": program_to_dict(hypothesis.program),
                        }
                    )
            solved_details[task_id] = winning_hypotheses
        if arguments.progress_every and ordinal % arguments.progress_every == 0:
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "total": len(selected),
                        "world_model_pass_at_2": len(solved["world_model"]),
                        "combined_pass_at_2": len(solved["combined"]),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    report = {
        "schema_version": 1,
        "solver": "recursive-world-model-v1",
        "ownership": "hyper-arc-owned-only",
        "partition": arguments.partition,
        "partition_sha256": digest,
        "evaluated_tasks": len(selected),
        "exact_fit_tasks": exact_fit_tasks,
        "recursive_fit_tasks": recursive_fit_tasks,
        "retrieved_seed_tasks": retrieved_seed_tasks,
        "executed_candidates": executed_candidates,
        "world_distinct_attempt_outputs": distinct_world_outputs,
        **{f"{name}_pass_at_2": len(value) for name, value in solved.items()},
        "world_unique_vs_owned": sorted(
            solved["world_model"] - solved["owned_baseline"]
        ),
        "world_regressions_vs_owned": sorted(
            solved["owned_baseline"] - solved["world_model"]
        ),
        "combined_unique_vs_owned": sorted(
            solved["combined"] - solved["owned_baseline"]
        ),
        "combined_regressions_vs_owned": sorted(
            solved["owned_baseline"] - solved["combined"]
        ),
        "world_solved_task_ids": sorted(solved["world_model"]),
        "world_solved_details": solved_details,
        "procedural_memory": (
            {
                "bank_id": procedural_memory.bank_id,
                "records": len(procedural_memory.records),
            }
            if procedural_memory is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
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

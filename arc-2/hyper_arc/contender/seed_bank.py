"""Development-only structured schema memory and contamination-safe retrieval."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyper_arc.contender.executor import (
    ExecutionError,
    TypedExecutor,
    legacy_name_to_ast,
    replay_program,
)
from hyper_arc.contender.perception import perceive_grid
from hyper_arc.contender.schemas import (
    Grid,
    GridPair,
    ProgramAST,
    ValueType,
    program_from_dict,
    program_to_dict,
)
from hyper_arc.deterministic import ExactCandidate, exact_candidates, prediction_pair


@dataclass(frozen=True)
class TaskFingerprint:
    train_count: int
    dimension_rules: tuple[str, ...]
    palette_deltas: tuple[int, ...]
    object_deltas: tuple[int, ...]
    input_symmetries: tuple[str, ...]
    output_symmetries: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskFingerprint":
        return cls(
            train_count=int(payload["train_count"]),
            dimension_rules=tuple(payload["dimension_rules"]),
            palette_deltas=tuple(int(value) for value in payload["palette_deltas"]),
            object_deltas=tuple(int(value) for value in payload["object_deltas"]),
            input_symmetries=tuple(payload["input_symmetries"]),
            output_symmetries=tuple(payload["output_symmetries"]),
        )


@dataclass(frozen=True)
class SeedSchema:
    schema_id: str
    legacy_name: str
    program: Mapping[str, Any]
    complexity: int
    evidence_count: int
    source_task_ids: tuple[str, ...]
    source_fingerprints: tuple[TaskFingerprint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "legacy_name": self.legacy_name,
            "program": dict(self.program),
            "complexity": self.complexity,
            "evidence_count": self.evidence_count,
            "source_task_ids": list(self.source_task_ids),
            "source_fingerprints": [
                fingerprint.to_dict() for fingerprint in self.source_fingerprints
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedSchema":
        program = payload["program"]
        if not isinstance(program, Mapping):
            raise ValueError("Seed schema program must be a mapping")
        executable = program_from_dict(program)
        return cls(
            schema_id=str(payload["schema_id"]),
            legacy_name=str(payload["legacy_name"]),
            program=program_to_dict(executable),
            complexity=int(payload["complexity"]),
            evidence_count=int(payload["evidence_count"]),
            source_task_ids=tuple(payload["source_task_ids"]),
            source_fingerprints=tuple(
                TaskFingerprint.from_dict(item)
                for item in payload["source_fingerprints"]
            ),
        )


@dataclass(frozen=True)
class StructuredSeedBank:
    schema_version: int
    bank_id: str
    split_sha256: str
    training_dataset_sha256: str
    miner_version: str
    schemas: tuple[SeedSchema, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "split_sha256": self.split_sha256,
            "training_dataset_sha256": self.training_dataset_sha256,
            "miner_version": self.miner_version,
            "schemas": [schema.to_dict() for schema in self.schemas],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuredSeedBank":
        return cls(
            schema_version=int(payload["schema_version"]),
            bank_id=str(payload["bank_id"]),
            split_sha256=str(payload["split_sha256"]),
            training_dataset_sha256=str(payload["training_dataset_sha256"]),
            miner_version=str(payload["miner_version"]),
            schemas=tuple(SeedSchema.from_dict(item) for item in payload["schemas"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StructuredSeedBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def retrieve(
        self, fingerprint: TaskFingerprint, limit: int = 8
    ) -> tuple[tuple[SeedSchema, float], ...]:
        ranked = []
        for schema in self.schemas:
            distance = min(
                fingerprint_distance(fingerprint, source)
                for source in schema.source_fingerprints
            )
            ranked.append((schema, distance))
        ranked.sort(
            key=lambda item: (
                item[1],
                -item[0].evidence_count,
                item[0].complexity,
                item[0].legacy_name,
            )
        )
        return tuple(ranked[:limit])


def _dimension_rule(source: Grid, target: Grid) -> str:
    input_height, input_width = len(source), len(source[0])
    output_height, output_width = len(target), len(target[0])
    if (output_height, output_width) == (input_height, input_width):
        return "same"
    if (output_height, output_width) == (input_width, input_height):
        return "swap"
    if output_height % input_height == 0 and output_width % input_width == 0:
        return f"expand:{output_height // input_height}x{output_width // input_width}"
    if output_height <= input_height and output_width <= input_width:
        return "contract"
    return "reshape"


def task_fingerprint(task_data: Mapping[str, Any]) -> TaskFingerprint:
    dimension_rules: list[str] = []
    palette_deltas: list[int] = []
    object_deltas: list[int] = []
    input_symmetries: list[str] = []
    output_symmetries: list[str] = []
    train = task_data.get("train", [])
    for index, pair in enumerate(train):
        source = perceive_grid(pair["input"], f"fingerprint:{index}:input")
        target = perceive_grid(pair["output"], f"fingerprint:{index}:output")
        dimension_rules.append(_dimension_rule(source.grid, target.grid))
        palette_deltas.append(len(target.palette) - len(source.palette))
        object_deltas.append(len(target.objects) - len(source.objects))
        input_symmetries.extend(source.symmetries)
        output_symmetries.extend(target.symmetries)
    return TaskFingerprint(
        train_count=len(train),
        dimension_rules=tuple(sorted(dimension_rules)),
        palette_deltas=tuple(sorted(palette_deltas)),
        object_deltas=tuple(sorted(object_deltas)),
        input_symmetries=tuple(sorted(input_symmetries)),
        output_symmetries=tuple(sorted(output_symmetries)),
    )


def _counter_distance(first: Iterable[Any], second: Iterable[Any]) -> int:
    left, right = Counter(first), Counter(second)
    return sum(abs(left[key] - right[key]) for key in left.keys() | right.keys())


def fingerprint_distance(first: TaskFingerprint, second: TaskFingerprint) -> float:
    return (
        abs(first.train_count - second.train_count) * 0.25
        + _counter_distance(first.dimension_rules, second.dimension_rules) * 4.0
        + _counter_distance(first.palette_deltas, second.palette_deltas) * 1.5
        + _counter_distance(first.object_deltas, second.object_deltas)
        + _counter_distance(first.input_symmetries, second.input_symmetries) * 0.25
        + _counter_distance(first.output_symmetries, second.output_symmetries) * 0.25
    )


def _normalized_name(candidate_name: str) -> str:
    return candidate_name.removesuffix("+remap")


def mine_seed_bank(
    development_challenges: Mapping[str, Any],
    development_solutions: Mapping[str, Any],
    *,
    split_sha256: str,
    training_dataset_sha256: str,
) -> StructuredSeedBank:
    if set(development_challenges) != set(development_solutions):
        raise ValueError("Development challenges and solutions must have identical IDs")
    evidence: dict[str, dict[str, Any]] = {}
    for task_id, task in development_challenges.items():
        train_pairs = [(pair["input"], pair["output"]) for pair in task["train"]]
        test_inputs = [pair["input"] for pair in task["test"]]
        expected = development_solutions[task_id]
        fingerprint = task_fingerprint(task)
        winners = {
            _normalized_name(candidate.name)
            for candidate in exact_candidates(train_pairs, test_inputs)
            if all(
                candidate.predict(test_input) == solution
                for test_input, solution in zip(test_inputs, expected)
            )
        }
        for name in winners:
            row = evidence.setdefault(name, {"tasks": [], "fingerprints": []})
            row["tasks"].append(task_id)
            row["fingerprints"].append(fingerprint)

    schemas = []
    for name, row in evidence.items():
        program = legacy_name_to_ast(name)
        source_ids = tuple(sorted(row["tasks"]))
        unique_fingerprints = tuple(
            {
                json.dumps(item.to_dict(), sort_keys=True): item
                for item in row["fingerprints"]
            }.values()
        )
        schema_payload = json.dumps(
            {"name": name, "program": program_to_dict(program)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        schemas.append(
            SeedSchema(
                schema_id=hashlib.sha256(schema_payload).hexdigest(),
                legacy_name=name,
                program=program_to_dict(program),
                complexity=program.complexity,
                evidence_count=len(source_ids),
                source_task_ids=source_ids,
                source_fingerprints=unique_fingerprints,
            )
        )
    schemas.sort(key=lambda item: (-item.evidence_count, item.complexity, item.legacy_name))
    bank_payload = json.dumps(
        [schema.to_dict() for schema in schemas],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return StructuredSeedBank(
        schema_version=1,
        bank_id=hashlib.sha256(bank_payload).hexdigest(),
        split_sha256=split_sha256,
        training_dataset_sha256=training_dataset_sha256,
        miner_version="typed-exact-v1",
        schemas=tuple(schemas),
    )


def _infer_color_map(transformed: tuple[Grid, ...], targets: tuple[Grid, ...]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for source, target in zip(transformed, targets):
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            return None
        for source_row, target_row in zip(source, target):
            for before, after in zip(source_row, target_row):
                known = mapping.get(before)
                if known is not None and known != after:
                    return None
                mapping[before] = after
    return mapping


def bind_schema(
    schema: SeedSchema, task_data: Mapping[str, Any], executor: TypedExecutor
) -> ProgramAST | None:
    base = program_from_dict(schema.program)
    pairs = tuple(
        GridPair(
            input=tuple(tuple(row) for row in pair["input"]),
            output=tuple(tuple(row) for row in pair["output"]),
        )
        for pair in task_data["train"]
    )
    try:
        transformed = tuple(
            executor.execute(base, pair.input).value for pair in pairs
        )
    except ExecutionError:
        return None
    mapping = _infer_color_map(transformed, tuple(pair.output for pair in pairs))
    if mapping is None:
        return None
    changed = {source: target for source, target in mapping.items() if source != target}
    program = base
    if changed:
        program = ProgramAST(
            op="color_remap",
            output_type=ValueType.GRID,
            arguments={"mapping": changed},
            children=(base,),
        )
    try:
        _, _, exact = replay_program(executor, program, pairs, task_id="schema-bind")
    except ExecutionError:
        return None
    return program if exact else None


def memory_prediction_pairs(
    bank: StructuredSeedBank,
    task_data: Mapping[str, Any],
    *,
    retrieval_limit: int = 8,
    baseline_candidates: list[ExactCandidate] | None = None,
) -> list[tuple[list[list[int]], list[list[int]]]]:
    executor = TypedExecutor()
    test_inputs = [pair["input"] for pair in task_data["test"]]
    ranked = bank.retrieve(task_fingerprint(task_data), retrieval_limit)
    prediction_sets: list[list[list[list[int]]]] = []
    seen_behaviors: set[tuple[Grid, ...]] = set()
    for schema, _ in ranked:
        program = bind_schema(schema, task_data, executor)
        if program is None:
            continue
        try:
            behavior = tuple(
                executor.execute(program, grid).value for grid in test_inputs
            )
        except ExecutionError:
            continue
        if behavior in seen_behaviors:
            continue
        seen_behaviors.add(behavior)
        prediction_sets.append(
            [[list(row) for row in prediction] for prediction in behavior]
        )

    train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]
    candidates = (
        baseline_candidates
        if baseline_candidates is not None
        else exact_candidates(train_pairs, test_inputs)
    )
    for candidate in candidates:
        behavior = tuple(
            tuple(tuple(row) for row in candidate.predict(grid)) for grid in test_inputs
        )
        if behavior in seen_behaviors:
            continue
        seen_behaviors.add(behavior)
        prediction_sets.append(
            [candidate.predict(grid) for grid in test_inputs]
        )

    attempts = []
    for index, input_grid in enumerate(test_inputs):
        distinct: list[list[list[int]]] = []
        for predictions in prediction_sets:
            prediction = predictions[index]
            if prediction not in distinct:
                distinct.append(prediction)
            if len(distinct) == 2:
                break
        while len(distinct) < 2:
            distinct.append([row[:] for row in input_grid])
        attempts.append((distinct[0], distinct[1]))
    return attempts


def run_ablation(
    bank: StructuredSeedBank,
    holdout_challenges: Mapping[str, Any],
    holdout_solutions: Mapping[str, Any],
    *,
    retrieval_limit: int = 8,
) -> dict[str, Any]:
    if set(holdout_challenges) != set(holdout_solutions):
        raise ValueError("Holdout challenges and solutions must have identical IDs")
    started = time.perf_counter()
    baseline_solved: set[str] = set()
    memory_solved: set[str] = set()
    for task_id, task in holdout_challenges.items():
        train_pairs = [(pair["input"], pair["output"]) for pair in task["train"]]
        test_inputs = [pair["input"] for pair in task["test"]]
        expected = holdout_solutions[task_id]
        baseline_candidates = exact_candidates(train_pairs, test_inputs)
        baseline_attempts = [
            prediction_pair(baseline_candidates, grid) for grid in test_inputs
        ]
        if baseline_attempts and all(
            solution in attempts
            for solution, attempts in zip(expected, baseline_attempts)
        ):
            baseline_solved.add(task_id)
        memory_attempts = memory_prediction_pairs(
            bank,
            task,
            retrieval_limit=retrieval_limit,
            baseline_candidates=baseline_candidates,
        )
        if memory_attempts and all(
            solution in attempts for solution, attempts in zip(expected, memory_attempts)
        ):
            memory_solved.add(task_id)
    return {
        "schema_version": 1,
        "bank_id": bank.bank_id,
        "holdout_tasks": len(holdout_challenges),
        "retrieval_limit": retrieval_limit,
        "baseline_pass_at_2": len(baseline_solved),
        "memory_pass_at_2": len(memory_solved),
        "unique_memory_solves": sorted(memory_solved - baseline_solved),
        "memory_regressions": sorted(baseline_solved - memory_solved),
        "baseline_solved_task_ids": sorted(baseline_solved),
        "memory_solved_task_ids": sorted(memory_solved),
        "elapsed_seconds": time.perf_counter() - started,
    }

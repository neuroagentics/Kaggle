"""Versioned, executable cross-task experience memory for Hyper-ARC.

The bank stores abstraction-level evidence, not task answers.  Retrieval uses
color-invariant structural features, successful stored programs are always
replayed against the current demonstrations, and family generators may only
contribute hypotheses that exactly reproduce every demonstration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyper_arc.contender.executor import (
    ExecutionError,
    TypedExecutor,
    legacy_name_to_ast,
    replay_program,
)
from hyper_arc.contender.object_programs import ObjectProgramCandidate, exact_object_candidates
from hyper_arc.contender.perception import perceive_grid
from hyper_arc.contender.relational_plans import RelationalCandidate, exact_relational_candidates
from hyper_arc.contender.schemas import (
    Grid,
    GridPair,
    ProgramAST,
    program_from_dict,
    program_to_dict,
)
from hyper_arc.deterministic import ExactCandidate, exact_candidates


SCHEMA_VERSION = 2
MINER_VERSION = "executable-experience-v2"
FAMILIES = ("deterministic", "object", "relational")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(row) for row in value)


def _pairs(task_data: Mapping[str, Any]) -> tuple[GridPair, ...]:
    return tuple(
        GridPair(input=_grid(pair["input"]), output=_grid(pair["output"]))
        for pair in task_data.get("train", [])
    )


def _bucket(value: float, width: float = 0.1) -> int:
    return int(round(value / width))


@dataclass(frozen=True)
class ExperienceFingerprint:
    """A compact, color-invariant structural description of an ARC task."""

    features: Mapping[str, float]
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"features": dict(sorted(self.features.items())), "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceFingerprint":
        features = payload.get("features")
        tags = payload.get("tags")
        if not isinstance(features, Mapping) or not isinstance(tags, list):
            raise ValueError("Invalid experience fingerprint")
        return cls(
            features={str(key): float(value) for key, value in features.items()},
            tags=tuple(str(tag) for tag in tags),
        )


def task_fingerprint_v2(task_data: Mapping[str, Any]) -> ExperienceFingerprint:
    train = task_data.get("train", [])
    if not train:
        return ExperienceFingerprint(features={"train_count": 0.0}, tags=("empty",))
    features: defaultdict[str, float] = defaultdict(float)
    tags: list[str] = []
    features["train_count"] = float(len(train))
    for index, pair in enumerate(train):
        source = perceive_grid(pair["input"], f"experience:{index}:input")
        target = perceive_grid(pair["output"], f"experience:{index}:output")
        ih, iw = source.height, source.width
        oh, ow = target.height, target.width
        features["input_area_log"] += math.log1p(ih * iw)
        features["output_area_log"] += math.log1p(oh * ow)
        features["input_objects"] += len(source.objects)
        features["output_objects"] += len(target.objects)
        features["palette_delta"] += len(target.palette) - len(source.palette)
        features["object_delta"] += len(target.objects) - len(source.objects)
        features["separator_delta"] += (
            len(target.separator_rows)
            + len(target.separator_columns)
            - len(source.separator_rows)
            - len(source.separator_columns)
        )
        if (ih, iw) == (oh, ow):
            dimension = "same"
            changed = sum(
                pair["input"][row][col] != pair["output"][row][col]
                for row in range(ih)
                for col in range(iw)
            )
            features["changed_ratio_bucket"] += _bucket(changed / max(1, ih * iw))
        elif (ih, iw) == (ow, oh):
            dimension = "swap"
        elif oh % ih == 0 and ow % iw == 0:
            dimension = f"expand:{oh // ih}x{ow // iw}"
        elif oh <= ih and ow <= iw:
            dimension = "contract"
        else:
            dimension = "reshape"
        tags.append(f"dimension:{dimension}")
        tags.extend(f"input_symmetry:{value}" for value in source.symmetries)
        tags.extend(f"output_symmetry:{value}" for value in target.symmetries)
        if source.separator_rows or source.separator_columns:
            tags.append("input:separated")
        if target.separator_rows or target.separator_columns:
            tags.append("output:separated")
    scale = float(len(train))
    for key in tuple(features):
        if key != "train_count":
            features[key] /= scale
    return ExperienceFingerprint(features=dict(features), tags=tuple(sorted(tags)))


def fingerprint_distance(
    first: ExperienceFingerprint, second: ExperienceFingerprint
) -> float:
    keys = set(first.features) | set(second.features)
    numeric = sum(
        abs(first.features.get(key, 0.0) - second.features.get(key, 0.0))
        / (1.0 + abs(first.features.get(key, 0.0)) + abs(second.features.get(key, 0.0)))
        for key in keys
    )
    left, right = set(first.tags), set(second.tags)
    tag_distance = 0.0 if not (left or right) else 1.0 - len(left & right) / len(left | right)
    return numeric + 3.0 * tag_distance


@dataclass(frozen=True)
class ExperienceRecord:
    record_id: str
    source_digest: str
    family: str
    fingerprint: ExperienceFingerprint
    demonstration_exact: bool
    test_exact: bool
    candidate_count: int
    program: Mapping[str, Any] | None
    invariants: tuple[str, ...]
    failure_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_digest": self.source_digest,
            "family": self.family,
            "fingerprint": self.fingerprint.to_dict(),
            "demonstration_exact": self.demonstration_exact,
            "test_exact": self.test_exact,
            "candidate_count": self.candidate_count,
            "program": dict(self.program) if self.program is not None else None,
            "invariants": list(self.invariants),
            "failure_modes": list(self.failure_modes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceRecord":
        family = str(payload["family"])
        if family not in FAMILIES:
            raise ValueError(f"Unknown experience family: {family}")
        program = payload.get("program")
        if program is not None:
            if not isinstance(program, Mapping):
                raise ValueError("Experience program must be a mapping")
            program = program_to_dict(program_from_dict(program))
        record = cls(
            record_id=str(payload["record_id"]),
            source_digest=str(payload["source_digest"]),
            family=family,
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
            demonstration_exact=bool(payload["demonstration_exact"]),
            test_exact=bool(payload["test_exact"]),
            candidate_count=int(payload["candidate_count"]),
            program=program,
            invariants=tuple(str(item) for item in payload.get("invariants", [])),
            failure_modes=tuple(str(item) for item in payload.get("failure_modes", [])),
        )
        expected = _record_id(record.to_dict() | {"record_id": ""})
        if record.record_id != expected:
            raise ValueError("Experience record digest mismatch")
        return record


def _record_id(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("record_id", None)
    return _canonical_digest(canonical)


@dataclass(frozen=True)
class ExperienceBank:
    schema_version: int
    bank_id: str
    split_sha256: str
    training_dataset_sha256: str
    miner_version: str
    records: tuple[ExperienceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "split_sha256": self.split_sha256,
            "training_dataset_sha256": self.training_dataset_sha256,
            "miner_version": self.miner_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceBank":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported experience-bank schema")
        records = tuple(ExperienceRecord.from_dict(item) for item in payload["records"])
        bank = cls(
            schema_version=SCHEMA_VERSION,
            bank_id=str(payload["bank_id"]),
            split_sha256=str(payload["split_sha256"]),
            training_dataset_sha256=str(payload["training_dataset_sha256"]),
            miner_version=str(payload["miner_version"]),
            records=records,
        )
        if bank.bank_id != _bank_id(bank):
            raise ValueError("Experience bank digest mismatch")
        return bank

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def retrieve(
        self, fingerprint: ExperienceFingerprint, *, limit: int = 96
    ) -> tuple[tuple[ExperienceRecord, float], ...]:
        ranked = [
            (record, fingerprint_distance(fingerprint, record.fingerprint))
            for record in self.records
        ]
        ranked.sort(
            key=lambda item: (
                item[1],
                not item[0].test_exact,
                not item[0].demonstration_exact,
                item[0].family,
                item[0].record_id,
            )
        )
        return tuple(ranked[:limit])


def _bank_id(bank: ExperienceBank) -> str:
    return _canonical_digest(
        {
            "schema_version": bank.schema_version,
            "split_sha256": bank.split_sha256,
            "training_dataset_sha256": bank.training_dataset_sha256,
            "miner_version": bank.miner_version,
            "records": [record.to_dict() for record in bank.records],
        }
    )


@dataclass(frozen=True)
class ExperienceCandidate:
    name: str
    family: str
    program: ProgramAST
    predictions: tuple[Grid, ...]
    confidence: float
    complexity: int
    source: str

    def predict(self, index: int) -> list[list[int]]:
        return [list(row) for row in self.predictions[index]]


def _family_candidates(task_data: Mapping[str, Any], family: str) -> list[Any]:
    train = [(pair["input"], pair["output"]) for pair in task_data.get("train", [])]
    tests = [pair["input"] for pair in task_data.get("test", [])]
    if family == "deterministic":
        return exact_candidates(train, tests)
    if family == "object":
        return exact_object_candidates(task_data)
    if family == "relational":
        return exact_relational_candidates(task_data)
    raise ValueError(f"Unknown family: {family}")


def _candidate_program(candidate: Any) -> ProgramAST | None:
    if isinstance(candidate, (ObjectProgramCandidate, RelationalCandidate)):
        return candidate.program
    return None


def _stored_candidate_program(candidate: Any) -> ProgramAST | None:
    if isinstance(candidate, ExactCandidate):
        return legacy_name_to_ast(candidate.name.removesuffix("+remap"))
    return _candidate_program(candidate)


def _candidate_predictions(candidate: Any, task_data: Mapping[str, Any]) -> tuple[Grid, ...]:
    return tuple(_grid(candidate.predict(pair["input"])) for pair in task_data.get("test", []))


def mine_experience_bank(
    builder_challenges: Mapping[str, Any],
    builder_solutions: Mapping[str, Any],
    *,
    split_sha256: str,
    training_dataset_sha256: str,
) -> ExperienceBank:
    """Mine success and failure experience exclusively from the builder fold."""
    if set(builder_challenges) != set(builder_solutions):
        raise ValueError("Builder challenges and solutions must have identical IDs")
    records: list[ExperienceRecord] = []
    for task_id, task in builder_challenges.items():
        fingerprint = task_fingerprint_v2(task)
        expected = builder_solutions[task_id]
        for family in FAMILIES:
            candidates = _family_candidates(task, family)
            winner = next(
                (
                    candidate
                    for candidate in candidates
                    if all(
                        candidate.predict(pair["input"]) == output
                        for pair, output in zip(task.get("test", []), expected)
                    )
                ),
                None,
            )
            failure_modes: list[str] = []
            if not candidates:
                failure_modes.append("no_exact_demonstration_program")
            elif winner is None:
                failure_modes.append("demonstration_fit_test_failure")
            program = _stored_candidate_program(winner) if winner is not None else None
            payload = {
                "record_id": "",
                "source_digest": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
                "family": family,
                "fingerprint": fingerprint.to_dict(),
                "demonstration_exact": bool(candidates),
                "test_exact": winner is not None,
                "candidate_count": len(candidates),
                "program": program_to_dict(program) if program is not None else None,
                "invariants": ["exact-demonstration-replay", "color-label-invariant-retrieval"],
                "failure_modes": failure_modes,
            }
            payload["record_id"] = _record_id(payload)
            records.append(ExperienceRecord.from_dict(payload))
    records.sort(key=lambda record: (record.family, record.source_digest))
    prototype = ExperienceBank(
        schema_version=SCHEMA_VERSION,
        bank_id="",
        split_sha256=split_sha256,
        training_dataset_sha256=training_dataset_sha256,
        miner_version=MINER_VERSION,
        records=tuple(records),
    )
    return ExperienceBank(
        schema_version=prototype.schema_version,
        bank_id=_bank_id(prototype),
        split_sha256=prototype.split_sha256,
        training_dataset_sha256=prototype.training_dataset_sha256,
        miner_version=prototype.miner_version,
        records=prototype.records,
    )


def _family_confidence(
    retrieved: Sequence[tuple[ExperienceRecord, float]], family: str
) -> float:
    weighted_success = 1.0
    weighted_demo_failure = 1.0
    for record, distance in retrieved:
        if record.family != family:
            continue
        weight = math.exp(-distance)
        if record.test_exact:
            weighted_success += weight
        elif record.demonstration_exact:
            weighted_demo_failure += weight
        else:
            weighted_demo_failure += 0.25 * weight
    return weighted_success / (weighted_success + weighted_demo_failure)


def _replay_stored(
    record: ExperienceRecord, task_data: Mapping[str, Any], executor: TypedExecutor
) -> ProgramAST | None:
    if not record.test_exact or record.program is None:
        return None
    program = program_from_dict(record.program)
    pairs = _pairs(task_data)
    try:
        _, _, exact = replay_program(executor, program, pairs, task_id="memory-replay")
    except (ExecutionError, ValueError):
        return None
    if exact:
        return program
    if record.family != "deterministic":
        return None
    transformed: list[Grid] = []
    try:
        transformed = [
            executor.execute(program, pair.input, capture_trace=False).value
            for pair in pairs
        ]
    except ExecutionError:
        return None
    mapping: dict[int, int] = {}
    for source, pair in zip(transformed, pairs):
        if len(source) != len(pair.output) or len(source[0]) != len(pair.output[0]):
            return None
        for source_row, target_row in zip(source, pair.output):
            for before, after in zip(source_row, target_row):
                if before in mapping and mapping[before] != after:
                    return None
                mapping[before] = after
    changed = {before: after for before, after in mapping.items() if before != after}
    if not changed:
        return None
    rebound = ProgramAST(
        op="color_remap",
        output_type=program.output_type,
        arguments={"mapping": changed},
        children=(program,),
    )
    try:
        _, _, exact = replay_program(executor, rebound, pairs, task_id="memory-bind")
    except ExecutionError:
        return None
    return rebound if exact else None


def experience_candidates(
    bank: ExperienceBank,
    task_data: Mapping[str, Any],
    *,
    retrieval_limit: int = 96,
    transfer_limit: int = 32,
) -> list[ExperienceCandidate]:
    """Retrieve, bind/generate, exact-replay, and rank memory hypotheses."""
    fingerprint = task_fingerprint_v2(task_data)
    retrieved = bank.retrieve(fingerprint, limit=retrieval_limit)
    confidence = {
        family: _family_confidence(retrieved, family) for family in FAMILIES
    }
    executor = TypedExecutor()
    test_inputs = [pair["input"] for pair in task_data.get("test", [])]
    results: list[ExperienceCandidate] = []
    seen: set[tuple[Grid, ...]] = set()

    for record, distance in retrieved[:transfer_limit]:
        program = _replay_stored(record, task_data, executor)
        if program is None:
            continue
        try:
            behavior = tuple(
                executor.execute(program, grid, capture_trace=False).value
                for grid in test_inputs
            )
        except ExecutionError:
            continue
        if behavior in seen:
            continue
        seen.add(behavior)
        results.append(
            ExperienceCandidate(
                name=f"memory:{record.family}:{record.record_id[:10]}",
                family=record.family,
                program=program,
                predictions=behavior,
                confidence=min(0.999, confidence[record.family] + 0.1 * math.exp(-distance)),
                complexity=program.complexity,
                source="retrieved-program",
            )
        )

    for family in FAMILIES:
        for candidate in _family_candidates(task_data, family):
            program = _candidate_program(candidate)
            if program is None:
                continue
            behavior = _candidate_predictions(candidate, task_data)
            if behavior in seen:
                continue
            seen.add(behavior)
            results.append(
                ExperienceCandidate(
                    name=f"bound:{family}:{getattr(candidate, 'name', getattr(candidate, 'kind', 'program'))}",
                    family=family,
                    program=program,
                    predictions=behavior,
                    confidence=confidence[family],
                    complexity=getattr(candidate, "complexity", program.complexity),
                    source="memory-ranked-generator",
                )
            )
    results.sort(
        key=lambda item: (-item.confidence, item.complexity, item.family, item.name)
    )
    return results


def empty_experience_bank() -> ExperienceBank:
    prototype = ExperienceBank(
        schema_version=SCHEMA_VERSION,
        bank_id="",
        split_sha256="test-only",
        training_dataset_sha256="test-only",
        miner_version=MINER_VERSION,
        records=(),
    )
    return ExperienceBank(
        schema_version=SCHEMA_VERSION,
        bank_id=_bank_id(prototype),
        split_sha256=prototype.split_sha256,
        training_dataset_sha256=prototype.training_dataset_sha256,
        miner_version=prototype.miner_version,
        records=(),
    )

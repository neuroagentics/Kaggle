"""Lean, versioned procedural memory for exact ARC transformation chains.

The bank stores only abstraction-level evidence mined from the frozen builder
fold.  A procedure receives positive evidence only when it exactly predicts
every withheld output for a source task; exact demonstration fits that fail on
the withheld output are retained as negative boundary evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hyper_arc.contender.experience_bank import (
    ExperienceFingerprint,
    fingerprint_distance,
    task_fingerprint_v2,
)
from hyper_arc.deterministic import ExactCandidate, exact_candidates


SCHEMA_VERSION = 1
MINER_VERSION = "exact-procedure-chains-v1"


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def procedure_family(name: str) -> str:
    base = name.removesuffix("+remap")
    if base.startswith("crop-") or base.startswith("crop0+") or base.startswith("crop_mode+"):
        return "object-extraction"
    if base.startswith(("hcat-", "vcat-", "mirror-quad-")):
        return "spatial-composition"
    if base.startswith("panel") and not base.startswith("panel-"):
        return "spatial-composition"
    if base.startswith("compress-lines+"):
        return "pattern-compression"
    if base.startswith("panel-"):
        return "panel-relation"
    if (
        "upscale" in base
        or "tile" in base
        or base.startswith(("downscale", "block-"))
    ):
        return "scale-and-repetition"
    if base.startswith("select-color-"):
        return "color-selection"
    if base in {
        "rotate90",
        "rotate180",
        "rotate270",
        "reflect_horizontal",
        "reflect_vertical",
        "transpose",
        "anti_transpose",
    }:
        return "geometric"
    if name.endswith("+remap"):
        return "color-binding"
    return "identity"


def procedure_steps(name: str) -> tuple[str, ...]:
    """Return a stable, human-readable operation chain from a procedure name."""
    remap = name.endswith("+remap")
    base = name.removesuffix("+remap")
    if base.startswith(("hcat-", "vcat-", "mirror-quad-", "panel-")):
        steps = (base,)
    else:
        steps = tuple(part for part in base.split("+") if part)
    return steps + (("bind-colors",) if remap else ())


def procedure_invariants(name: str) -> tuple[str, ...]:
    invariants = [
        "must-exactly-replay-all-demonstrations",
        "must-produce-a-valid-arc-grid",
    ]
    family = procedure_family(name)
    if family == "object-extraction":
        invariants.append("requires-a-unique-selected-object-or-color-region")
    elif family == "pattern-compression":
        invariants.append("requires-redundant-adjacent-lines")
    elif family == "panel-relation":
        invariants.append("requires-two-equal-sized-separated-panels")
    elif family == "scale-and-repetition":
        invariants.append("requires-an-integer-shape-ratio")
    elif family == "color-selection":
        invariants.append("requires-a-unique-color-statistic-extremum")
    if name.endswith("+remap"):
        invariants.append("requires-a-consistent-demonstration-color-map")
    return tuple(invariants)


@dataclass(frozen=True)
class ProcedureEpisode:
    source_task_id: str
    fingerprint: ExperienceFingerprint
    test_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "fingerprint": self.fingerprint.to_dict(),
            "test_exact": self.test_exact,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcedureEpisode":
        return cls(
            source_task_id=str(payload["source_task_id"]),
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
            test_exact=bool(payload["test_exact"]),
        )


@dataclass(frozen=True)
class ProcedureRecord:
    name: str
    family: str
    steps: tuple[str, ...]
    invariants: tuple[str, ...]
    demonstration_fits: int
    exact_successes: int
    episodes: tuple[ProcedureEpisode, ...]

    @property
    def transfer_failures(self) -> int:
        return self.demonstration_fits - self.exact_successes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "steps": list(self.steps),
            "invariants": list(self.invariants),
            "demonstration_fits": self.demonstration_fits,
            "exact_successes": self.exact_successes,
            "transfer_failures": self.transfer_failures,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcedureRecord":
        record = cls(
            name=str(payload["name"]),
            family=str(payload["family"]),
            steps=tuple(str(value) for value in payload["steps"]),
            invariants=tuple(str(value) for value in payload["invariants"]),
            demonstration_fits=int(payload["demonstration_fits"]),
            exact_successes=int(payload["exact_successes"]),
            episodes=tuple(
                ProcedureEpisode.from_dict(value) for value in payload["episodes"]
            ),
        )
        if record.family != procedure_family(record.name):
            raise ValueError("Procedure family does not match its name")
        if record.steps != procedure_steps(record.name):
            raise ValueError("Procedure steps do not match its name")
        if record.invariants != procedure_invariants(record.name):
            raise ValueError("Procedure invariants do not match its name")
        if record.demonstration_fits != len(record.episodes):
            raise ValueError("Procedure demonstration count does not match episodes")
        if record.exact_successes != sum(value.test_exact for value in record.episodes):
            raise ValueError("Procedure exact-success count does not match episodes")
        if int(payload.get("transfer_failures", record.transfer_failures)) != record.transfer_failures:
            raise ValueError("Procedure transfer-failure count is inconsistent")
        return record


@dataclass(frozen=True)
class ProceduralMemoryBank:
    bank_id: str
    split_sha256: str
    training_dataset_sha256: str
    records: tuple[ProcedureRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "miner_version": MINER_VERSION,
            "bank_id": self.bank_id,
            "split_sha256": self.split_sha256,
            "training_dataset_sha256": self.training_dataset_sha256,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProceduralMemoryBank":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported procedural-memory schema")
        if payload.get("miner_version") != MINER_VERSION:
            raise ValueError("Unsupported procedural-memory miner")
        bank = cls(
            bank_id=str(payload["bank_id"]),
            split_sha256=str(payload["split_sha256"]),
            training_dataset_sha256=str(payload["training_dataset_sha256"]),
            records=tuple(ProcedureRecord.from_dict(value) for value in payload["records"]),
        )
        expected = _bank_id(bank)
        if bank.bank_id != expected:
            raise ValueError("Procedural-memory bank digest mismatch")
        return bank

    @classmethod
    def load(cls, path: str | Path) -> "ProceduralMemoryBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def rank(
        self,
        candidates: Sequence[ExactCandidate],
        task_data: Mapping[str, Any],
    ) -> list[ExactCandidate]:
        """Rank exact-replay candidates by related success/failure evidence.

        The bank never creates or validates a hypothesis. It only orders
        candidates that already passed exact replay on the current task.
        """
        fingerprint = task_fingerprint_v2(task_data)
        records = {record.name: record for record in self.records}

        def evidence(candidate: ExactCandidate) -> tuple[float, int]:
            record = records.get(candidate.name)
            if record is None:
                return 0.5, 0
            success = 1.0
            failure = 1.0
            for episode in record.episodes:
                weight = math.exp(-fingerprint_distance(fingerprint, episode.fingerprint))
                if episode.test_exact:
                    success += weight
                else:
                    failure += weight
            return success / (success + failure), record.exact_successes

        return sorted(
            candidates,
            key=lambda candidate: (
                -evidence(candidate)[0],
                -evidence(candidate)[1],
                candidate.complexity,
                len(candidate.name),
                candidate.name,
            ),
        )

    def retrieve_cues(
        self,
        task_data: Mapping[str, Any],
        *,
        limit: int = 6,
        max_distance: float = 2.5,
    ) -> tuple[str, ...]:
        """Retrieve only nearby successful procedures for neural grounding.

        Global top procedures are intentionally excluded: unrelated but common
        memories can anchor a language model on the wrong transformation family.
        """
        fingerprint = task_fingerprint_v2(task_data)
        ranked: list[tuple[float, float, str, ProcedureRecord]] = []
        for record in self.records:
            successful = [episode for episode in record.episodes if episode.test_exact]
            if not successful:
                continue
            distance = min(
                fingerprint_distance(fingerprint, episode.fingerprint)
                for episode in successful
            )
            if distance > max_distance:
                continue
            precision = (record.exact_successes + 1) / (record.demonstration_fits + 2)
            ranked.append((distance, -precision, record.name, record))
        ranked.sort(key=lambda item: item[:3])
        return tuple(
            f"{record.family}: {' then '.join(record.steps)}; similarity_distance="
            f"{distance:.3f}; withheld_exact={record.exact_successes}/"
            f"{record.demonstration_fits}"
            for distance, _precision, _name, record in ranked[:limit]
        )


def _bank_id(bank: ProceduralMemoryBank) -> str:
    return _canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "miner_version": MINER_VERSION,
            "split_sha256": bank.split_sha256,
            "training_dataset_sha256": bank.training_dataset_sha256,
            "records": [record.to_dict() for record in bank.records],
        }
    )


def mine_procedural_memory(
    builder_challenges: Mapping[str, Any],
    builder_solutions: Mapping[str, Any],
    *,
    split_sha256: str,
    training_dataset_sha256: str,
) -> ProceduralMemoryBank:
    """Mine exact successes and transfer failures from builder tasks only."""
    if set(builder_challenges) != set(builder_solutions):
        raise ValueError("Builder challenges and solutions must have identical IDs")
    episodes: dict[str, list[ProcedureEpisode]] = {}
    for task_id, task in builder_challenges.items():
        train = [(pair["input"], pair["output"]) for pair in task.get("train", [])]
        tests = [pair["input"] for pair in task.get("test", [])]
        expected = builder_solutions[task_id]
        fingerprint = task_fingerprint_v2(task)
        for candidate in exact_candidates(train, tests):
            test_exact = all(
                candidate.predict(pair["input"]) == output
                for pair, output in zip(task.get("test", []), expected)
            )
            episodes.setdefault(candidate.name, []).append(
                ProcedureEpisode(
                    source_task_id=task_id,
                    fingerprint=fingerprint,
                    test_exact=test_exact,
                )
            )
    records = tuple(
        ProcedureRecord(
            name=name,
            family=procedure_family(name),
            steps=procedure_steps(name),
            invariants=procedure_invariants(name),
            demonstration_fits=len(values),
            exact_successes=sum(value.test_exact for value in values),
            episodes=tuple(values),
        )
        for name, values in sorted(episodes.items())
    )
    prototype = ProceduralMemoryBank(
        bank_id="",
        split_sha256=split_sha256,
        training_dataset_sha256=training_dataset_sha256,
        records=records,
    )
    return ProceduralMemoryBank(
        bank_id=_bank_id(prototype),
        split_sha256=prototype.split_sha256,
        training_dataset_sha256=prototype.training_dataset_sha256,
        records=prototype.records,
    )

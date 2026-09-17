"""Versioned exact-success memory for model-authored ARC procedures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hyper_arc.contender.agentic_reasoner import execute_code, validate_code
from hyper_arc.contender.experience_bank import (
    ExperienceFingerprint,
    fingerprint_distance,
    task_fingerprint_v2,
)


SCHEMA_VERSION = 1


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AgenticMemoryRecord:
    record_id: str
    source_task_id: str
    summary: str
    code: str
    model: str
    fingerprint: ExperienceFingerprint
    validation_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_task_id": self.source_task_id,
            "summary": self.summary,
            "code": self.code,
            "model": self.model,
            "fingerprint": self.fingerprint.to_dict(),
            "validation_scope": self.validation_scope,
            "full_task_exact": True,
        }

    @classmethod
    def create(
        cls,
        *,
        source_task_id: str,
        summary: str,
        code: str,
        model: str,
        task_data: Mapping[str, Any],
        validation_scope: str,
    ) -> "AgenticMemoryRecord":
        validate_code(code)
        payload = {
            "source_task_id": source_task_id,
            "summary": summary,
            "code": code,
            "model": model,
            "fingerprint": task_fingerprint_v2(task_data).to_dict(),
            "validation_scope": validation_scope,
            "full_task_exact": True,
        }
        return cls(
            record_id=_digest(payload),
            source_task_id=source_task_id,
            summary=summary,
            code=code,
            model=model,
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
            validation_scope=validation_scope,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgenticMemoryRecord":
        if payload.get("full_task_exact") is not True:
            raise ValueError("Agentic memory accepts only full-task exact records")
        record = cls(
            record_id=str(payload["record_id"]),
            source_task_id=str(payload["source_task_id"]),
            summary=str(payload["summary"]),
            code=str(payload["code"]),
            model=str(payload["model"]),
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
            validation_scope=str(payload["validation_scope"]),
        )
        validate_code(record.code)
        expected = _digest(
            {
                "source_task_id": record.source_task_id,
                "summary": record.summary,
                "code": record.code,
                "model": record.model,
                "fingerprint": record.fingerprint.to_dict(),
                "validation_scope": record.validation_scope,
                "full_task_exact": True,
            }
        )
        if record.record_id != expected:
            raise ValueError("Agentic memory record digest mismatch")
        return record


@dataclass(frozen=True)
class AgenticMemoryCandidate:
    record: AgenticMemoryRecord
    distance: float
    test_predictions: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class AgenticMemoryBank:
    records: tuple[AgenticMemoryRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": "model-authored-full-task-exact-only",
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgenticMemoryBank":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported agentic-memory schema")
        records = tuple(AgenticMemoryRecord.from_dict(item) for item in payload["records"])
        if len({record.record_id for record in records}) != len(records):
            raise ValueError("Duplicate agentic-memory record")
        return cls(records=records)

    @classmethod
    def load(cls, path: str | Path) -> "AgenticMemoryBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def merged(self, records) -> "AgenticMemoryBank":
        values = {record.record_id: record for record in self.records}
        for record in records:
            values[record.record_id] = record
        return AgenticMemoryBank(tuple(values[key] for key in sorted(values)))

    def exact_candidates(
        self,
        task_data: Mapping[str, Any],
        *,
        retrieve_limit: int = 12,
        execution_timeout: float = 2.0,
    ) -> tuple[AgenticMemoryCandidate, ...]:
        fingerprint = task_fingerprint_v2(task_data)
        ranked = sorted(
            self.records,
            key=lambda record: (
                fingerprint_distance(fingerprint, record.fingerprint),
                record.record_id,
            ),
        )[:retrieve_limit]
        train_inputs = [pair["input"] for pair in task_data.get("train", [])]
        train_outputs = [pair["output"] for pair in task_data.get("train", [])]
        test_inputs = [pair["input"] for pair in task_data.get("test", [])]
        exact = []
        for record in ranked:
            try:
                predictions = execute_code(
                    record.code, train_inputs, timeout_seconds=execution_timeout
                )
                if predictions != train_outputs:
                    continue
                tests = execute_code(
                    record.code, test_inputs, timeout_seconds=execution_timeout
                )
            except (OSError, ValueError):
                continue
            exact.append(
                AgenticMemoryCandidate(
                    record=record,
                    distance=fingerprint_distance(fingerprint, record.fingerprint),
                    test_predictions=tuple(
                        tuple(tuple(row) for row in grid) for grid in tests
                    ),
                )
            )
        return tuple(exact)


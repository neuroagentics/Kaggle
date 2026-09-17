"""Task-conditioned Poincare-ball memory for executable ARC experience.

The memory is deliberately symbolic at its boundary: task fingerprints and
typed programs remain inspectable, while their deterministic embeddings are
used only to organize retrieval in hyperbolic space. Retrieved programs still
have to pass exact demonstration replay before they can become hypotheses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyper_arc.contender.experience_bank import (
    ExperienceBank,
    ExperienceFingerprint,
)
from hyper_arc.contender.schemas import ProgramAST, program_from_dict, program_to_dict


SCHEMA_VERSION = 1
DEFAULT_DIMENSIONS = 64
BALL_RADIUS = 0.95


def _token_slot(token: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    slot = int.from_bytes(digest[:8], "big") % dimensions
    sign = 1.0 if digest[8] & 1 else -1.0
    return slot, sign


def _ball_point(
    values: Iterable[tuple[str, float]], dimensions: int
) -> tuple[float, ...]:
    if dimensions < 2:
        raise ValueError("A Poincare ball needs at least two dimensions")
    vector = [0.0] * dimensions
    for token, weight in values:
        slot, sign = _token_slot(token, dimensions)
        vector[slot] += sign * float(weight)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    # Exponential-map-like radial compression keeps every point strictly in
    # the open unit ball while retaining the direction of the symbolic code.
    target_norm = BALL_RADIUS * math.tanh(norm / 4.0)
    scale = target_norm / norm
    return tuple(value * scale for value in vector)


def fingerprint_point(
    fingerprint: ExperienceFingerprint, *, dimensions: int = DEFAULT_DIMENSIONS
) -> tuple[float, ...]:
    """Embed a structural task fingerprint inside the open Poincare ball."""
    values: list[tuple[str, float]] = []
    for key, value in sorted(fingerprint.features.items()):
        magnitude = math.copysign(math.log1p(abs(float(value))), float(value))
        values.append((f"feature:{key}", magnitude))
        values.append((f"feature-value:{key}:{round(float(value), 3)}", 0.5))
    values.extend((f"tag:{tag}", 1.0) for tag in sorted(fingerprint.tags))
    return _ball_point(values, dimensions)


def _canonical_argument(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def program_point(
    program: ProgramAST, *, dimensions: int = DEFAULT_DIMENSIONS
) -> tuple[float, ...]:
    """Embed an AST without discarding child order, depth, or arguments."""
    values: list[tuple[str, float]] = []

    def visit(node: ProgramAST, path: str, depth: int, relation: str) -> None:
        prefix = f"{relation}:{path}:depth={depth}"
        values.append((f"{prefix}:op={node.op}", 1.0))
        values.append((f"{prefix}:type={node.output_type.value}", 0.5))
        for key, value in sorted(node.arguments.items()):
            values.append((f"{prefix}:arg:{key}={_canonical_argument(value)}", 0.75))
        for index, child in enumerate(node.children):
            visit(child, f"{path}.{index}", depth + 1, "child")
        for index, child in enumerate(node.preconditions):
            visit(child, f"{path}.pre{index}", depth + 1, "pre")
        for index, child in enumerate(node.postconditions):
            visit(child, f"{path}.post{index}", depth + 1, "post")

    visit(program, "root", 0, "root")
    return _ball_point(values, dimensions)


def poincare_distance(first: Iterable[float], second: Iterable[float]) -> float:
    """Geodesic distance in the unit-curvature Poincare ball."""
    left, right = tuple(first), tuple(second)
    if len(left) != len(right):
        raise ValueError("Poincare points must have equal dimensions")
    left_norm = sum(value * value for value in left)
    right_norm = sum(value * value for value in right)
    if left_norm >= 1.0 or right_norm >= 1.0:
        raise ValueError("Poincare points must lie inside the open unit ball")
    delta = sum((a - b) ** 2 for a, b in zip(left, right))
    denominator = max(1e-15, (1.0 - left_norm) * (1.0 - right_norm))
    return math.acosh(max(1.0, 1.0 + 2.0 * delta / denominator))


@dataclass(frozen=True)
class HyperbolicMemoryItem:
    record_id: str
    source_task_id: str
    family: str
    fingerprint: ExperienceFingerprint
    task_point: tuple[float, ...]
    program: ProgramAST | None
    program_point: tuple[float, ...] | None
    demonstration_exact: bool
    test_exact: bool
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_task_id": self.source_task_id,
            "family": self.family,
            "fingerprint": self.fingerprint.to_dict(),
            "task_point": list(self.task_point),
            "program": program_to_dict(self.program) if self.program else None,
            "program_point": list(self.program_point) if self.program_point else None,
            "demonstration_exact": self.demonstration_exact,
            "test_exact": self.test_exact,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HyperbolicMemoryItem":
        program_payload = payload.get("program")
        return cls(
            record_id=str(payload["record_id"]),
            source_task_id=str(payload["source_task_id"]),
            family=str(payload["family"]),
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
            task_point=tuple(float(value) for value in payload["task_point"]),
            program=program_from_dict(program_payload) if program_payload else None,
            program_point=(
                tuple(float(value) for value in payload["program_point"])
                if payload.get("program_point") is not None
                else None
            ),
            demonstration_exact=bool(payload["demonstration_exact"]),
            test_exact=bool(payload["test_exact"]),
            provenance=dict(payload.get("provenance", {})),
        )


@dataclass(frozen=True)
class MemoryNeighbor:
    item: HyperbolicMemoryItem
    distance: float
    weight: float


class HyperbolicWorldMemory:
    """Mutable session memory with deterministic hyperbolic retrieval."""

    def __init__(
        self,
        items: Iterable[HyperbolicMemoryItem] = (),
        *,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        self.dimensions = dimensions
        self._items = list(items)
        if any(len(item.task_point) != dimensions for item in self._items):
            raise ValueError("Memory item dimensions do not match the index")

    @property
    def items(self) -> tuple[HyperbolicMemoryItem, ...]:
        return tuple(self._items)

    @classmethod
    def from_experience_bank(
        cls, bank: ExperienceBank, *, dimensions: int = DEFAULT_DIMENSIONS
    ) -> "HyperbolicWorldMemory":
        items = []
        for record in bank.records:
            program = program_from_dict(record.program) if record.program else None
            items.append(
                HyperbolicMemoryItem(
                    record_id=record.record_id,
                    source_task_id=record.source_digest,
                    family=record.family,
                    fingerprint=record.fingerprint,
                    task_point=fingerprint_point(
                        record.fingerprint, dimensions=dimensions
                    ),
                    program=program,
                    program_point=(
                        program_point(program, dimensions=dimensions)
                        if program
                        else None
                    ),
                    demonstration_exact=record.demonstration_exact,
                    test_exact=record.test_exact,
                    provenance={
                        "source": "experience-bank-v2",
                        "bank_id": bank.bank_id,
                    },
                )
            )
        return cls(items, dimensions=dimensions)

    def retrieve(
        self,
        fingerprint: ExperienceFingerprint,
        *,
        limit: int = 32,
        require_program: bool = False,
    ) -> tuple[MemoryNeighbor, ...]:
        query = fingerprint_point(fingerprint, dimensions=self.dimensions)
        candidates = (
            item
            for item in self._items
            if not require_program or item.program is not None
        )
        ranked = [
            MemoryNeighbor(
                item=item,
                distance=poincare_distance(query, item.task_point),
                weight=0.0,
            )
            for item in candidates
        ]
        ranked = [
            MemoryNeighbor(
                item=value.item,
                distance=value.distance,
                weight=math.exp(-value.distance),
            )
            for value in ranked
        ]
        ranked.sort(
            key=lambda value: (
                value.distance,
                not value.item.test_exact,
                not value.item.demonstration_exact,
                value.item.record_id,
            )
        )
        return tuple(ranked[: max(0, limit)])

    def remember(
        self,
        *,
        source_task_id: str,
        family: str,
        fingerprint: ExperienceFingerprint,
        program: ProgramAST,
        demonstration_exact: bool,
        test_exact: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> HyperbolicMemoryItem:
        payload = f"{source_task_id}:{family}:{program.digest}:{len(self._items)}"
        record_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        item = HyperbolicMemoryItem(
            record_id=record_id,
            source_task_id=source_task_id,
            family=family,
            fingerprint=fingerprint,
            task_point=fingerprint_point(fingerprint, dimensions=self.dimensions),
            program=program,
            program_point=program_point(program, dimensions=self.dimensions),
            demonstration_exact=demonstration_exact,
            test_exact=test_exact,
            provenance=dict(provenance or {}),
        )
        self._items.append(item)
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dimensions": self.dimensions,
            "items": [item.to_dict() for item in self._items],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HyperbolicWorldMemory":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported hyperbolic-memory schema")
        return cls(
            (HyperbolicMemoryItem.from_dict(item) for item in payload.get("items", [])),
            dimensions=int(payload["dimensions"]),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "HyperbolicWorldMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

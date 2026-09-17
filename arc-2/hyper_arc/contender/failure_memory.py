"""Versioned failure memory for the model-authored ARC reasoner.

The success bank (`agentic_memory.py`) records only exact solves. On its own
that lets the agent rediscover the same dead ends every run: it re-proposes rule
families that were already shown not to reproduce the demonstrations. This bank
is the complement -- a concise, structural record of what was *tried and did not
work*, so the reasoner can recall it on similar tasks and steer toward an
untried hypothesis instead of repeating a known failure.

Design decisions (deliberately narrow to stay useful, per the "prune what
offered no insight" guidance):

* Only ``executed-but-non-exact`` approaches are stored as avoidable failures.
  A candidate that actually ran and still failed to replay the demonstrations is
  genuine evidence that the *hypothesis* is wrong. Duplicates, syntax errors, and
  sandbox rejections are pipeline noise about output *form*, not about the
  problem, so they are counted for health telemetry but never stored as
  "approaches to avoid."
* Failures are keyed by the task structural fingerprint (color-invariant), so
  recall generalizes across tasks with the same shape of transformation.
* Each stored family carries a short reason and the residual error magnitude,
  so the reasoner can reason over *why* it failed, not just that it did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyper_arc.contender.experience_bank import (
    ExperienceFingerprint,
    fingerprint_distance,
    task_fingerprint_v2,
)


SCHEMA_VERSION = 1

# Rejection categories that describe the *hypothesis* being wrong (worth
# remembering) versus categories that only describe malformed output (pipeline
# noise, not stored as avoidable approaches).
INSIGHTFUL = "non-exact"
NOISE_CATEGORIES = frozenset({"duplicate", "syntax", "forbidden", "worker", "model"})


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_rejection(rejection: str) -> str:
    """Map a raw reasoner rejection string to a coarse category."""
    if ":non-exact:" in rejection:
        return INSIGHTFUL
    if rejection.endswith(":duplicate"):
        return "duplicate"
    if "invalid Python" in rejection:
        return "syntax"
    if "not allowed" in rejection:
        return "forbidden"
    if "worker rejected" in rejection or "invalid ARC grid" in rejection:
        return "worker"
    if ":model:" in rejection:
        return "model"
    return "other"


def approach_family(summary: str) -> str:
    """Condense a hypothesis summary into a stable, comparable family key.

    Uses the leading clause of the natural-language rule, lower-cased and
    trimmed. This is intentionally coarse: the goal is to recognize "we already
    tried the recolor-by-area family and it failed" without demanding an exact
    string match.
    """
    text = summary.strip().lower()
    for stop in (".", ";", ",", " where ", " such that ", " starting "):
        index = text.find(stop)
        if index > 0:
            text = text[:index]
    return " ".join(text.split())[:120]


@dataclass(frozen=True)
class FailureRecord:
    record_id: str
    source_task_id: str
    family: str
    reason: str
    residual: int
    fingerprint: ExperienceFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_task_id": self.source_task_id,
            "family": self.family,
            "reason": self.reason,
            "residual": self.residual,
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def create(
        cls,
        *,
        source_task_id: str,
        family: str,
        reason: str,
        residual: int,
        task_data: Mapping[str, Any],
    ) -> "FailureRecord":
        fingerprint = task_fingerprint_v2(task_data)
        payload = {
            "source_task_id": source_task_id,
            "family": family,
            "fingerprint": fingerprint.to_dict(),
        }
        return cls(
            record_id=_digest(payload),
            source_task_id=source_task_id,
            family=family,
            reason=reason,
            residual=int(residual),
            fingerprint=fingerprint,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureRecord":
        return cls(
            record_id=str(payload["record_id"]),
            source_task_id=str(payload["source_task_id"]),
            family=str(payload["family"]),
            reason=str(payload["reason"]),
            residual=int(payload.get("residual", 0)),
            fingerprint=ExperienceFingerprint.from_dict(payload["fingerprint"]),
        )


def extract_failed_families(
    source_task_id: str,
    task_data: Mapping[str, Any],
    rejections: Iterable[str],
) -> tuple[FailureRecord, ...]:
    """Build avoidable-failure records from a task's rejection log.

    Only ``non-exact`` rejections (ran but wrong) become records. The rejection
    string format is ``r{round}c{col}:non-exact:{residual}:{summary}``.
    Deduplicated by (family) so one family failing many times is stored once,
    keeping the strongest (largest residual) reason.
    """
    best: dict[str, FailureRecord] = {}
    for rejection in rejections:
        if classify_rejection(rejection) != INSIGHTFUL:
            continue
        parts = rejection.split(":", 3)
        if len(parts) < 4:
            continue
        try:
            residual = int(parts[2])
        except ValueError:
            residual = 0
        family = approach_family(parts[3])
        if not family:
            continue
        record = FailureRecord.create(
            source_task_id=source_task_id,
            family=family,
            reason=f"executed but off by {residual} cells/shape on the demonstrations",
            residual=residual,
            task_data=task_data,
        )
        existing = best.get(family)
        if existing is None or residual > existing.residual:
            best[family] = record
    return tuple(best[key] for key in sorted(best))


@dataclass(frozen=True)
class FailureMemoryBank:
    records: tuple[FailureRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": "executed-but-non-exact-approaches-only",
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureMemoryBank":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported failure-memory schema")
        records = tuple(FailureRecord.from_dict(item) for item in payload["records"])
        return cls(records=records)

    @classmethod
    def load(cls, path: str | Path) -> "FailureMemoryBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def merged(self, records: Iterable[FailureRecord]) -> "FailureMemoryBank":
        # Deduplicate by record_id (task-fingerprint + family), keeping the
        # entry with the largest residual as the representative reason.
        values: dict[str, FailureRecord] = {r.record_id: r for r in self.records}
        for record in records:
            existing = values.get(record.record_id)
            if existing is None or record.residual > existing.residual:
                values[record.record_id] = record
        return FailureMemoryBank(tuple(values[key] for key in sorted(values)))

    def recall(
        self,
        task_data: Mapping[str, Any],
        *,
        limit: int = 12,
        max_distance: float = 1.5,
    ) -> tuple[str, ...]:
        """Return concise 'already failed on similar tasks' cues for planning.

        Ranked by structural similarity; only reasonably-close matches are
        returned so the reasoner is not warned off families that failed on
        structurally unrelated tasks.
        """
        fingerprint = task_fingerprint_v2(task_data)
        scored = sorted(
            (
                (fingerprint_distance(fingerprint, record.fingerprint), record)
                for record in self.records
            ),
            key=lambda item: (item[0], item[1].record_id),
        )
        seen: set[str] = set()
        cues: list[str] = []
        for distance, record in scored:
            if distance > max_distance or record.family in seen:
                continue
            seen.add(record.family)
            cues.append(f"{record.family} ({record.reason})")
            if len(cues) >= limit:
                break
        return tuple(cues)

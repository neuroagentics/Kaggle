"""Deterministic ARC-AGI-2 data partitioning and contamination controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class DataProtocolError(ValueError):
    """Raised when local data does not match the frozen protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_digest(development: Iterable[str], holdout: Iterable[str]) -> str:
    payload = json.dumps(
        {"development": list(development), "holdout": list(holdout)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    dataset_sha256: str
    task_count: int
    development_count: int
    holdout_count: int
    algorithm: str
    salt: str
    split_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        split = payload["training_split"]
        return cls(
            schema_version=int(payload["schema_version"]),
            dataset_sha256=str(payload["datasets"]["training_challenges"]["sha256"]),
            task_count=int(split["task_count"]),
            development_count=int(split["development_count"]),
            holdout_count=int(split["holdout_count"]),
            algorithm=str(split["algorithm"]),
            salt=str(split["salt"]),
            split_sha256=str(split["split_sha256"]),
        )


def build_split(
    task_ids: Iterable[str], manifest: SplitManifest
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact frozen development/holdout membership.

    Membership is determined only by task ID and the public manifest salt. No
    solution content or manually assigned task family influences the split.
    """
    ids = tuple(task_ids)
    if len(ids) != len(set(ids)):
        raise DataProtocolError("Task IDs must be unique")
    if len(ids) != manifest.task_count:
        raise DataProtocolError(
            f"Expected {manifest.task_count} tasks, found {len(ids)}"
        )
    if manifest.algorithm != "sha256-salted-rank-v1":
        raise DataProtocolError(f"Unsupported split algorithm: {manifest.algorithm}")

    ranked = tuple(
        sorted(
            ids,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{manifest.salt}:{task_id}".encode("utf-8")
                ).hexdigest(),
                task_id,
            ),
        )
    )
    development = ranked[: manifest.development_count]
    holdout = ranked[manifest.development_count :]
    if len(holdout) != manifest.holdout_count:
        raise DataProtocolError(
            f"Expected {manifest.holdout_count} holdout tasks, found {len(holdout)}"
        )
    actual_digest = _split_digest(development, holdout)
    if actual_digest != manifest.split_sha256:
        raise DataProtocolError(
            "Split membership does not match the frozen manifest: "
            f"expected {manifest.split_sha256}, got {actual_digest}"
        )
    return development, holdout


def load_verified_split(
    challenges_path: str | Path, manifest_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen training dataset and return development and holdout maps."""
    source = Path(challenges_path)
    manifest = SplitManifest.load(manifest_path)
    actual_hash = sha256_file(source)
    if actual_hash != manifest.dataset_sha256:
        raise DataProtocolError(
            "Training dataset hash mismatch: "
            f"expected {manifest.dataset_sha256}, got {actual_hash}"
        )
    payload: Mapping[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    development, holdout = build_split(payload, manifest)
    return (
        {task_id: payload[task_id] for task_id in development},
        {task_id: payload[task_id] for task_id in holdout},
    )

"""Training corpus manifest — R09 implementation.

Provides the source/license/split ledger required before any model training.
No training occurs in this module. It establishes:
  - SourceEntry: provenance, license, and clearance status for each data source
  - SplitManifest: whole-game/mechanic-family splits with cryptographic binding
  - ManifestBuilder: constructs and validates a corpus manifest
  - load_manifest / save_manifest: serialisation helpers
  - check_clearance: fail-closed gate; blocks training until data is cleared

Per blueprint §5 and delivery plan G2/G9:
  - Splits are by whole game/mechanic family; no cross-split trajectories
  - Hashes must be recorded before any fitting or tuning
  - Synthetic success is held out by mechanic combinations (not combined with real ARC evidence)
  - Full compliance review remains an external human requirement (R02 gate)

Linear issue: R09
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence


MANIFEST_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Clearance statuses
# ---------------------------------------------------------------------------

CLEARANCE_PENDING = "pending"
CLEARANCE_CLEARED = "cleared"
CLEARANCE_BLOCKED = "blocked"

_VALID_CLEARANCE = frozenset({CLEARANCE_PENDING, CLEARANCE_CLEARED, CLEARANCE_BLOCKED})
_VALID_SPLITS = frozenset({"train", "dev", "holdout"})


# ---------------------------------------------------------------------------
# Source entry
# ---------------------------------------------------------------------------

@dataclass
class SourceEntry:
    """Provenance and license record for one data source.

    Fields
    ------
    source_id : str
        Unique identifier for this source (e.g. 'arc3_public_train_2026').
    name : str
        Human-readable name.
    url : str
        Canonical URL or local path description.
    license : str
        SPDX identifier or 'proprietary' or 'unknown'. Must not be blank.
    data_type : str
        One of: 'arc_official', 'synthetic_curriculum', 'permitted_external',
                'holdout_only'.
    clearance_status : str
        One of: 'pending', 'cleared', 'blocked'.
    clearance_notes : str
        Human-authored rationale. Required before status can be 'cleared'.
    file_hash : str | None
        SHA-256 of the data file if locally available; None until computed.
    split_assignment : str
        One of: 'train', 'dev', 'holdout'.
    mechanic_families : tuple[str, ...]
        Mechanic family tags for this source (used for holdout splits).
    schema_version : str
    """
    source_id: str
    name: str
    url: str
    license: str
    data_type: str
    clearance_status: str
    clearance_notes: str
    file_hash: str | None
    split_assignment: str
    mechanic_families: tuple[str, ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    _VALID_TYPES = frozenset({
        "arc_official", "synthetic_curriculum", "permitted_external", "holdout_only"
    })

    def __post_init__(self):
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.license.strip():
            raise ValueError("license must not be empty (use 'unknown' if unresolved)")
        if self.data_type not in self._VALID_TYPES:
            raise ValueError(
                f"data_type {self.data_type!r} not in {sorted(self._VALID_TYPES)}"
            )
        if self.clearance_status not in _VALID_CLEARANCE:
            raise ValueError(
                f"clearance_status {self.clearance_status!r} not in {sorted(_VALID_CLEARANCE)}"
            )
        if self.split_assignment not in _VALID_SPLITS:
            raise ValueError(
                f"split_assignment {self.split_assignment!r} not in {sorted(_VALID_SPLITS)}"
            )
        if self.clearance_status == CLEARANCE_CLEARED and not self.clearance_notes.strip():
            raise ValueError(
                "clearance_notes must document the clearance decision before status='cleared'"
            )
        if self.file_hash is not None:
            if not isinstance(self.file_hash, str) or len(self.file_hash) != 64:
                raise ValueError("file_hash must be a 64-hex-char SHA-256 digest")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {MANIFEST_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )

    def is_cleared(self) -> bool:
        return self.clearance_status == CLEARANCE_CLEARED

    def with_hash(self, path: Path) -> "SourceEntry":
        """Return a copy with file_hash populated from the given path."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        import dataclasses
        return dataclasses.replace(self, file_hash=digest.hexdigest())


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------

@dataclass
class SplitManifest:
    """Immutable record of game/mechanic-family split assignments.

    The split is frozen before any model fitting or hyperparameter tuning.
    Hashes bind the split to the exact data files.

    Fields
    ------
    manifest_id : str
        Unique manifest identifier (set once, never changed after freeze).
    frozen : bool
        True once the split is finalised. No additions after freeze.
    train_game_ids : tuple[str, ...]
    dev_game_ids : tuple[str, ...]
    holdout_game_ids : tuple[str, ...]
    train_mechanic_families : tuple[str, ...]
    holdout_mechanic_families : tuple[str, ...]
        Must be disjoint from train_mechanic_families.
    source_hashes : dict[str, str]
        Maps source_id -> SHA-256 of the data file. Populated before freeze.
    schema_version : str
    """
    manifest_id: str
    frozen: bool
    train_game_ids: tuple[str, ...]
    dev_game_ids: tuple[str, ...]
    holdout_game_ids: tuple[str, ...]
    train_mechanic_families: tuple[str, ...]
    holdout_mechanic_families: tuple[str, ...]
    source_hashes: dict[str, str]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self):
        if not self.manifest_id.strip():
            raise ValueError("manifest_id must not be empty")
        # No game may appear in multiple splits
        all_games = (
            list(self.train_game_ids)
            + list(self.dev_game_ids)
            + list(self.holdout_game_ids)
        )
        if len(all_games) != len(set(all_games)):
            raise ValueError("A game_id appears in more than one split")
        # No mechanic family may be in both train and holdout
        train_set = set(self.train_mechanic_families)
        holdout_set = set(self.holdout_mechanic_families)
        overlap = train_set & holdout_set
        if overlap:
            raise ValueError(
                f"Mechanic families appear in both train and holdout: {sorted(overlap)}"
            )
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {MANIFEST_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )

    def is_held_out(self, game_id: str) -> bool:
        return game_id in self.holdout_game_ids

    def mechanic_is_held_out(self, family: str) -> bool:
        return family in self.holdout_mechanic_families

    def verify_hashes(self, source_paths: dict[str, Path]) -> list[str]:
        """Return a list of error strings; empty list means all hashes match."""
        errors = []
        for source_id, expected_hash in self.source_hashes.items():
            path = source_paths.get(source_id)
            if path is None:
                errors.append(f"No path provided for source_id {source_id!r}")
                continue
            if not path.is_file():
                errors.append(f"Missing file for source_id {source_id!r}: {path}")
                continue
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected_hash:
                errors.append(
                    f"Hash mismatch for source_id {source_id!r}: "
                    f"expected {expected_hash!r}, got {actual!r}"
                )
        return errors


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

class ManifestBuilder:
    """Constructs a corpus manifest with clearance tracking.

    Usage
    -----
    builder = ManifestBuilder("my-manifest-v1")
    builder.add_source(SourceEntry(...))
    builder.set_split(SplitManifest(...))
    manifest = builder.build()   # raises if any source is not cleared
    """

    def __init__(self, manifest_id: str):
        if not manifest_id.strip():
            raise ValueError("manifest_id must not be empty")
        self.manifest_id = manifest_id
        self._sources: dict[str, SourceEntry] = {}
        self._split: SplitManifest | None = None

    def add_source(self, entry: SourceEntry) -> None:
        if entry.source_id in self._sources:
            raise ValueError(f"Duplicate source_id: {entry.source_id!r}")
        self._sources[entry.source_id] = entry

    def set_split(self, split: SplitManifest) -> None:
        if split.frozen:
            raise ValueError(
                "Cannot set an already-frozen split; use a new SplitManifest"
            )
        self._split = split

    def sources(self) -> list[SourceEntry]:
        return list(self._sources.values())

    def check_clearance(self) -> list[str]:
        """Return clearance blockers. Empty list = all sources cleared."""
        blockers = []
        for entry in self._sources.values():
            if entry.clearance_status == CLEARANCE_BLOCKED:
                blockers.append(f"BLOCKED: {entry.source_id} — {entry.clearance_notes}")
            elif entry.clearance_status == CLEARANCE_PENDING:
                blockers.append(f"PENDING: {entry.source_id} — license/permission unresolved")
        return blockers

    def build(self) -> "CorpusManifest":
        """Build and return a CorpusManifest. Raises if any source is not cleared."""
        blockers = self.check_clearance()
        if blockers:
            raise RuntimeError(
                "Corpus manifest has unresolved clearance issues:\n"
                + "\n".join(f"  {b}" for b in blockers)
            )
        if self._split is None:
            raise RuntimeError(
                "No SplitManifest provided. Set one with set_split() before building."
            )
        return CorpusManifest(
            manifest_id=self.manifest_id,
            sources=tuple(self._sources.values()),
            split=self._split,
            schema_version=MANIFEST_SCHEMA_VERSION,
        )


# ---------------------------------------------------------------------------
# Corpus manifest (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusManifest:
    """Immutable, hash-bound record of all training data provenance.

    Produced by ManifestBuilder.build() once all sources are cleared.
    Serialised to JSON for the release gate.
    """
    manifest_id: str
    sources: tuple[SourceEntry, ...]
    split: SplitManifest
    schema_version: str

    def to_dict(self) -> dict:
        import dataclasses
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "sources": [asdict(s) for s in self.sources],
            "split": asdict(self.split),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_manifest(manifest: CorpusManifest, path: Path) -> None:
    """Write a manifest to a JSON file."""
    path.write_text(manifest.to_json(), encoding="utf-8")


def load_manifest(path: Path) -> CorpusManifest:
    """Load and validate a manifest from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version: {raw.get('schema_version')!r}"
        )
    sources = tuple(
        SourceEntry(
            source_id=s["source_id"],
            name=s["name"],
            url=s["url"],
            license=s["license"],
            data_type=s["data_type"],
            clearance_status=s["clearance_status"],
            clearance_notes=s["clearance_notes"],
            file_hash=s.get("file_hash"),
            split_assignment=s["split_assignment"],
            mechanic_families=tuple(s.get("mechanic_families", [])),
            schema_version=s.get("schema_version", MANIFEST_SCHEMA_VERSION),
        )
        for s in raw.get("sources", [])
    )
    sd = raw["split"]
    split = SplitManifest(
        manifest_id=sd["manifest_id"],
        frozen=sd["frozen"],
        train_game_ids=tuple(sd.get("train_game_ids", [])),
        dev_game_ids=tuple(sd.get("dev_game_ids", [])),
        holdout_game_ids=tuple(sd.get("holdout_game_ids", [])),
        train_mechanic_families=tuple(sd.get("train_mechanic_families", [])),
        holdout_mechanic_families=tuple(sd.get("holdout_mechanic_families", [])),
        source_hashes=sd.get("source_hashes", {}),
        schema_version=sd.get("schema_version", MANIFEST_SCHEMA_VERSION),
    )
    return CorpusManifest(
        manifest_id=raw["manifest_id"],
        sources=sources,
        split=split,
        schema_version=raw["schema_version"],
    )


# ---------------------------------------------------------------------------
# Fail-closed clearance gate
# ---------------------------------------------------------------------------

def check_clearance(manifest_path: Path) -> list[str]:
    """Load a manifest and return clearance blockers.

    Returns an empty list only when every source is 'cleared'.
    Used by the release gate (G0 data clearance) and pre-training checks.
    """
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        return [f"Cannot load manifest: {exc}"]
    blockers = []
    for s in manifest.sources:
        if s.clearance_status != CLEARANCE_CLEARED:
            blockers.append(
                f"{s.clearance_status.upper()}: {s.source_id} ({s.name})"
            )
    return blockers

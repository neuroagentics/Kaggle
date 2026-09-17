"""Tests for training/manifest.py — R09: corpus manifest and clearance gate.

Synthetic fixtures only. No real training data is created or consumed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from training.manifest import (
    CLEARANCE_BLOCKED,
    CLEARANCE_CLEARED,
    CLEARANCE_PENDING,
    MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    ManifestBuilder,
    SourceEntry,
    SplitManifest,
    check_clearance,
    load_manifest,
    save_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_source(
    source_id="src-01",
    status=CLEARANCE_CLEARED,
    notes="reviewed by human",
    split="train",
    data_type="arc_official",
    families=("movement",),
) -> SourceEntry:
    return SourceEntry(
        source_id=source_id,
        name="ARC-3 public train",
        url="https://example.com/data",
        license="CC-BY-4.0",
        data_type=data_type,
        clearance_status=status,
        clearance_notes=notes,
        file_hash=None,
        split_assignment=split,
        mechanic_families=families,
    )


def make_split(
    train=("g1", "g2"),
    dev=("g3",),
    holdout=("g4",),
    train_families=("movement",),
    holdout_families=("transformation",),
) -> SplitManifest:
    return SplitManifest(
        manifest_id="split-v1",
        frozen=False,
        train_game_ids=tuple(train),
        dev_game_ids=tuple(dev),
        holdout_game_ids=tuple(holdout),
        train_mechanic_families=tuple(train_families),
        holdout_mechanic_families=tuple(holdout_families),
        source_hashes={},
    )


def make_builder_cleared() -> ManifestBuilder:
    b = ManifestBuilder("manifest-v1")
    b.add_source(make_source("src-01", status=CLEARANCE_CLEARED))
    b.set_split(make_split())
    return b


# ---------------------------------------------------------------------------
# SourceEntry validation
# ---------------------------------------------------------------------------

def test_source_constructs_valid():
    s = make_source()
    assert s.clearance_status == CLEARANCE_CLEARED


def test_source_empty_id_rejected():
    with pytest.raises(ValueError, match="source_id"):
        make_source(source_id="")


def test_source_empty_license_rejected():
    with pytest.raises(ValueError):
        SourceEntry(
            source_id="x", name="n", url="u", license="",
            data_type="arc_official", clearance_status=CLEARANCE_PENDING,
            clearance_notes="", file_hash=None,
            split_assignment="train", mechanic_families=(),
        )


def test_source_invalid_data_type():
    with pytest.raises(ValueError, match="data_type"):
        make_source(data_type="secret_scrape")


def test_source_invalid_clearance_status():
    with pytest.raises(ValueError, match="clearance_status"):
        SourceEntry(
            source_id="x", name="n", url="u", license="MIT",
            data_type="arc_official", clearance_status="maybe",
            clearance_notes="", file_hash=None,
            split_assignment="train", mechanic_families=(),
        )


def test_source_cleared_requires_notes():
    with pytest.raises(ValueError, match="clearance_notes"):
        SourceEntry(
            source_id="x", name="n", url="u", license="MIT",
            data_type="arc_official", clearance_status=CLEARANCE_CLEARED,
            clearance_notes="   ",  # blank
            file_hash=None,
            split_assignment="train", mechanic_families=(),
        )


def test_source_invalid_split_assignment():
    with pytest.raises(ValueError, match="split_assignment"):
        make_source(split="validation")


def test_source_invalid_hash_length():
    with pytest.raises(ValueError, match="file_hash"):
        SourceEntry(
            source_id="x", name="n", url="u", license="MIT",
            data_type="arc_official", clearance_status=CLEARANCE_PENDING,
            clearance_notes="", file_hash="tooshort",
            split_assignment="train", mechanic_families=(),
        )


def test_source_is_cleared_flag():
    assert make_source(status=CLEARANCE_CLEARED).is_cleared()
    assert not make_source(status=CLEARANCE_PENDING, notes="").is_cleared()


# ---------------------------------------------------------------------------
# SplitManifest validation
# ---------------------------------------------------------------------------

def test_split_constructs_valid():
    s = make_split()
    assert not s.frozen


def test_split_duplicate_game_rejected():
    with pytest.raises(ValueError, match="more than one split"):
        SplitManifest(
            manifest_id="s", frozen=False,
            train_game_ids=("g1",),
            dev_game_ids=("g1",),  # duplicate
            holdout_game_ids=(),
            train_mechanic_families=(), holdout_mechanic_families=(),
            source_hashes={},
        )


def test_split_mechanic_overlap_rejected():
    with pytest.raises(ValueError, match="both train and holdout"):
        SplitManifest(
            manifest_id="s", frozen=False,
            train_game_ids=(), dev_game_ids=(), holdout_game_ids=(),
            train_mechanic_families=("movement",),
            holdout_mechanic_families=("movement",),  # overlap
            source_hashes={},
        )


def test_split_is_held_out():
    s = make_split(holdout=("g99",))
    assert s.is_held_out("g99")
    assert not s.is_held_out("g1")


def test_split_mechanic_is_held_out():
    s = make_split(holdout_families=("transformation",))
    assert s.mechanic_is_held_out("transformation")
    assert not s.mechanic_is_held_out("movement")


# ---------------------------------------------------------------------------
# ManifestBuilder
# ---------------------------------------------------------------------------

def test_builder_duplicate_source_rejected():
    b = ManifestBuilder("x")
    b.add_source(make_source("src-01"))
    with pytest.raises(ValueError, match="Duplicate source_id"):
        b.add_source(make_source("src-01"))


def test_builder_build_succeeds_when_all_cleared():
    b = make_builder_cleared()
    manifest = b.build()
    assert manifest.manifest_id == "manifest-v1"
    assert len(manifest.sources) == 1


def test_builder_build_blocked_when_pending():
    b = ManifestBuilder("x")
    b.add_source(make_source("src-01", status=CLEARANCE_PENDING, notes=""))
    b.set_split(make_split())
    with pytest.raises(RuntimeError, match="PENDING"):
        b.build()


def test_builder_build_blocked_when_source_blocked():
    b = ManifestBuilder("x")
    b.add_source(make_source("src-01", status=CLEARANCE_BLOCKED, notes="denied"))
    b.set_split(make_split())
    with pytest.raises(RuntimeError, match="BLOCKED"):
        b.build()


def test_builder_build_fails_without_split():
    b = ManifestBuilder("x")
    b.add_source(make_source())
    with pytest.raises(RuntimeError, match="SplitManifest"):
        b.build()


def test_builder_check_clearance_returns_blockers():
    b = ManifestBuilder("x")
    b.add_source(make_source("src-01", status=CLEARANCE_PENDING, notes=""))
    b.add_source(make_source("src-02", status=CLEARANCE_CLEARED))
    blockers = b.check_clearance()
    assert len(blockers) == 1
    assert "PENDING" in blockers[0]


def test_builder_set_frozen_split_rejected():
    b = ManifestBuilder("x")
    frozen_split = SplitManifest(
        manifest_id="s", frozen=True,
        train_game_ids=(), dev_game_ids=(), holdout_game_ids=(),
        train_mechanic_families=(), holdout_mechanic_families=(),
        source_hashes={},
    )
    with pytest.raises(ValueError, match="frozen"):
        b.set_split(frozen_split)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_manifest_serialises_and_loads():
    b = make_builder_cleared()
    manifest = b.build()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
    assert loaded.manifest_id == manifest.manifest_id
    assert len(loaded.sources) == len(manifest.sources)
    assert loaded.sources[0].source_id == manifest.sources[0].source_id
    assert loaded.split.manifest_id == manifest.split.manifest_id


def test_manifest_json_has_schema_version():
    b = make_builder_cleared()
    manifest = b.build()
    raw = json.loads(manifest.to_json())
    assert raw["schema_version"] == MANIFEST_SCHEMA_VERSION


def test_load_manifest_rejects_wrong_schema_version():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"schema_version": "9.9"}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_manifest(path)


# ---------------------------------------------------------------------------
# check_clearance gate
# ---------------------------------------------------------------------------

def test_check_clearance_empty_on_all_cleared():
    b = make_builder_cleared()
    manifest = b.build()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        save_manifest(manifest, path)
        blockers = check_clearance(path)
    assert blockers == []


def test_check_clearance_reports_pending_source():
    # Manually craft a manifest JSON with a pending source to test the gate
    # without requiring ManifestBuilder.build() to pass clearance
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test",
        "sources": [
            {
                "source_id": "pending-src",
                "name": "Unreviewed",
                "url": "http://example.com",
                "license": "unknown",
                "data_type": "arc_official",
                "clearance_status": CLEARANCE_PENDING,
                "clearance_notes": "",
                "file_hash": None,
                "split_assignment": "train",
                "mechanic_families": [],
                "schema_version": MANIFEST_SCHEMA_VERSION,
            }
        ],
        "split": {
            "manifest_id": "s",
            "frozen": False,
            "train_game_ids": [],
            "dev_game_ids": [],
            "holdout_game_ids": [],
            "train_mechanic_families": [],
            "holdout_mechanic_families": [],
            "source_hashes": {},
            "schema_version": MANIFEST_SCHEMA_VERSION,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        blockers = check_clearance(path)
    assert any("PENDING" in b for b in blockers)


def test_check_clearance_missing_file():
    blockers = check_clearance(Path("/nonexistent/path/manifest.json"))
    assert len(blockers) == 1
    assert "Cannot load" in blockers[0]

"""Frozen data split and contamination-boundary tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hyper_arc.contender.data_protocol import (
    DataProtocolError,
    SplitManifest,
    build_split,
    load_verified_split,
)


def test_frozen_arc2_split_matches_manifest():
    manifest = SplitManifest.load("config/arc2_split_v1.json")
    development, holdout = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )

    assert len(development) == 800
    assert len(holdout) == 200
    assert set(development).isdisjoint(holdout)
    assert len(development | holdout) == manifest.task_count


def test_split_rejects_changed_membership_digest():
    manifest = SplitManifest(
        schema_version=1,
        dataset_sha256="unused",
        task_count=2,
        development_count=1,
        holdout_count=1,
        algorithm="sha256-salted-rank-v1",
        salt="test",
        split_sha256="wrong",
    )
    with pytest.raises(DataProtocolError, match="frozen manifest"):
        build_split(["a", "b"], manifest)


def test_public_evaluation_solutions_are_scoring_only():
    registry = json.loads(open("config/arc2_split_v1.json", encoding="utf-8").read())
    assert (
        registry["datasets"]["public_evaluation_solutions"]["access"]
        == "scoring-only-never-memory-or-training"
    )


def test_channel_provenance_excludes_training_known_import_from_clean_claims():
    provenance = json.loads(
        open("config/channel_provenance_v1.json", encoding="utf-8").read()
    )
    imported = provenance["channels"]["verified-symbolic"]

    assert imported["predates_split"] is True
    assert imported["explicit_task_references"]["training"] == 111
    assert imported["clean_validation_claims"] == "excluded"

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in Path("reference/verified-symbolic").rglob("*.py")
    )
    referenced = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])", source_text))
    training = set(
        json.loads(
            Path("data/arc-agi-2/arc-agi_training_challenges.json").read_text(
                encoding="utf-8"
            )
        )
    )
    evaluation = set(
        json.loads(
            Path("data/arc-agi-2/arc-agi_evaluation_challenges.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert (
        len(referenced & training) == imported["explicit_task_references"]["training"]
    )
    assert (
        len(referenced & evaluation)
        == imported["explicit_task_references"]["public_evaluation"]
    )
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    validation = set(list(development)[640:])
    assert (
        len(referenced & validation)
        == imported["explicit_task_references"]["nested_validation"]
    )

"""Procedural-memory integrity and ranking tests."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.procedural_memory import (
    ProceduralMemoryBank,
    mine_procedural_memory,
    procedure_family,
    procedure_steps,
)
from hyper_arc.deterministic import ExactCandidate


def _task() -> dict:
    return {
        "train": [{"input": [[1]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[2]]}],
    }


def test_mining_keeps_successes_failures_semantics_and_episodes():
    bank = mine_procedural_memory(
        {"episode-1": _task()},
        {"episode-1": [[[2, 2], [2, 2]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )
    record = next(value for value in bank.records if value.name == "identity+upscale2")

    assert record.family == "scale-and-repetition"
    assert record.steps == ("identity", "upscale2")
    assert record.demonstration_fits == 1
    assert record.exact_successes == 1
    assert record.episodes[0].source_task_id == "episode-1"
    assert ProceduralMemoryBank.from_dict(bank.to_dict()) == bank


def test_memory_ranks_related_exact_success_before_unseen_candidate():
    bank = mine_procedural_memory(
        {"episode-1": _task()},
        {"episode-1": [[[2, 2], [2, 2]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )
    unseen = ExactCandidate("invented", 0, lambda grid: grid)
    supported = ExactCandidate("identity+upscale2", 1, lambda grid: grid)

    assert bank.rank([unseen, supported], _task())[0].name == supported.name


def test_neural_cues_are_similarity_gated():
    bank = mine_procedural_memory(
        {"episode-1": _task()},
        {"episode-1": [[[2, 2], [2, 2]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )

    cues = bank.retrieve_cues(_task(), max_distance=0.0)
    unrelated = {
        "train": [{"input": [[1, 2, 3]], "output": [[3], [2], [1]]}],
        "test": [{"input": [[4, 5, 6]]}],
    }

    assert cues and "identity then upscale2" in cues[0]
    assert bank.retrieve_cues(unrelated, max_distance=0.0) == ()


def test_bank_rejects_tampering_and_names_have_stable_descriptions():
    bank = mine_procedural_memory(
        {"episode-1": _task()},
        {"episode-1": [[[2, 2], [2, 2]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )
    payload = json.loads(json.dumps(bank.to_dict()))
    payload["records"][0]["exact_successes"] += 1

    with pytest.raises(ValueError):
        ProceduralMemoryBank.from_dict(payload)
    assert procedure_family("crop-rarest-color-raw+rotate90") == "object-extraction"
    assert procedure_steps("crop-rarest-color-raw+rotate90+remap") == (
        "crop-rarest-color-raw",
        "rotate90",
        "bind-colors",
    )

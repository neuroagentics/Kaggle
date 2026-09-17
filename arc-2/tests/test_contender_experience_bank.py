"""Executable experience-memory tests."""

from __future__ import annotations

import json

import pytest

from hyper_arc.contender.experience_bank import (
    ExperienceBank,
    experience_candidates,
    fingerprint_distance,
    mine_experience_bank,
    task_fingerprint_v2,
)


def move_task(color: int) -> dict:
    return {
        "train": [
            {
                "input": [[0, color, 0], [0, color, 0], [0, 0, 0]],
                "output": [[0, 0, color], [0, 0, color], [0, 0, 0]],
            }
        ],
        "test": [
            {"input": [[color, 0, 0], [color, 0, 0], [0, 0, 0]]}
        ],
    }


def test_fingerprint_is_color_invariant():
    first = move_task(1)
    second = move_task(7)

    assert task_fingerprint_v2(first) == task_fingerprint_v2(second)
    assert fingerprint_distance(task_fingerprint_v2(first), task_fingerprint_v2(second)) == 0


def test_bank_round_trip_and_memory_replays_a_stored_program():
    source = move_task(1)
    bank = mine_experience_bank(
        {"source": source},
        {"source": [[[0, 1, 0], [0, 1, 0], [0, 0, 0]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )

    restored = ExperienceBank.from_dict(bank.to_dict())
    candidates = experience_candidates(restored, move_task(7))

    assert restored.bank_id == bank.bank_id
    assert len(restored.records) == 3
    assert any(record.family == "object" and record.test_exact for record in restored.records)
    assert any(
        candidate.predict(0) == [[0, 7, 0], [0, 7, 0], [0, 0, 0]]
        for candidate in candidates
    )
    assert all(candidate.source in {"retrieved-program", "memory-ranked-generator"} for candidate in candidates)


def test_bank_rejects_tampered_record():
    task = move_task(1)
    bank = mine_experience_bank(
        {"source": task},
        {"source": [[[0, 1, 0], [0, 1, 0], [0, 0, 0]]]},
        split_sha256="split",
        training_dataset_sha256="dataset",
    )
    payload = json.loads(json.dumps(bank.to_dict()))
    payload["records"][0]["candidate_count"] += 1

    with pytest.raises(ValueError, match="record digest"):
        ExperienceBank.from_dict(payload)

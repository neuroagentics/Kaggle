"""Tests for the reasoner's failure memory: store, recall, prune, avoid-repeat."""

from __future__ import annotations

import pytest

from hyper_arc.contender.failure_memory import (
    FailureMemoryBank,
    FailureRecord,
    approach_family,
    classify_rejection,
    extract_failed_families,
)


def _task(color: int) -> dict:
    # A trivial same-shape recolor task; the exact content only needs to be a
    # valid ARC task so task_fingerprint_v2 can run.
    return {
        "train": [
            {"input": [[color, 0], [0, color]], "output": [[0, color], [color, 0]]},
            {"input": [[color, color], [0, 0]], "output": [[0, 0], [color, color]]},
        ],
        "test": [{"input": [[color, 0], [0, 0]]}],
    }


def test_classify_rejection_separates_insight_from_noise():
    assert classify_rejection("r0c1:non-exact:12:reflect the grid") == "non-exact"
    assert classify_rejection("r0c1:duplicate") == "duplicate"
    assert classify_rejection("r0c0:ValueError:invalid Python: bad") == "syntax"
    assert classify_rejection("r0c0:ValueError:method is not allowed: foo") == "forbidden"
    assert classify_rejection("r1:model:AgenticReasoningError:x") == "model"


def test_approach_family_condenses_to_leading_clause():
    fam = approach_family(
        "Reflect the grid horizontally, where each row is reversed."
    )
    assert fam == "reflect the grid horizontally"


def test_extract_failed_families_keeps_only_non_exact_and_dedupes():
    rejections = [
        "r0c0:non-exact:12:Reflect the grid horizontally",
        "r0c1:non-exact:40:Reflect the grid horizontally",  # same family, larger residual
        "r0c2:duplicate",  # noise, ignored
        "r0c3:ValueError:invalid Python: bad",  # noise, ignored
        "r1c0:non-exact:8:Rotate the grid 180 degrees",
    ]
    records = extract_failed_families("t1", _task(3), rejections)
    families = {r.family for r in records}
    assert families == {"reflect the grid horizontally", "rotate the grid 180 degrees"}
    reflect = next(r for r in records if r.family == "reflect the grid horizontally")
    assert reflect.residual == 40  # kept the strongest evidence


def test_recall_returns_families_for_similar_tasks_and_skips_far_ones():
    records = extract_failed_families(
        "t1", _task(3), ["r0c0:non-exact:12:Reflect the grid horizontally"]
    )
    bank = FailureMemoryBank().merged(records)
    # A structurally similar task recalls the failed family.
    cues = bank.recall(_task(5))
    assert any("reflect the grid horizontally" in cue for cue in cues)


def test_merged_dedupes_by_family_keeping_largest_residual():
    # Same task (same fingerprint) + same family must collapse to one record,
    # keeping the larger residual as the representative evidence.
    task = _task(3)
    a = extract_failed_families("t1", task, ["r0c0:non-exact:5:Recolor by area"])
    b = extract_failed_families("t1", task, ["r0c0:non-exact:50:Recolor by area"])
    bank = FailureMemoryBank().merged(a).merged(b)
    assert len(bank.records) == 1
    assert bank.records[0].residual == 50


def test_same_family_on_different_task_structure_stays_separate():
    # A family failing on structurally different tasks is two distinct lessons.
    a = extract_failed_families("t1", _task(3), ["r0c0:non-exact:5:Recolor by area"])
    b = extract_failed_families(
        "t2",
        {
            "train": [{"input": [[1]], "output": [[1, 1], [1, 1]]}],
            "test": [{"input": [[2]]}],
        },
        ["r0c0:non-exact:5:Recolor by area"],
    )
    bank = FailureMemoryBank().merged(a).merged(b)
    assert len(bank.records) == 2


def test_bank_roundtrips_through_dict(tmp_path):
    records = extract_failed_families(
        "t1", _task(3), ["r0c0:non-exact:12:Reflect the grid horizontally"]
    )
    bank = FailureMemoryBank().merged(records)
    path = tmp_path / "failures.json"
    bank.save(path)
    restored = FailureMemoryBank.load(path)
    assert len(restored.records) == len(bank.records)
    assert restored.records[0].family == bank.records[0].family


def test_empty_bank_recall_is_empty():
    assert FailureMemoryBank().recall(_task(1)) == ()

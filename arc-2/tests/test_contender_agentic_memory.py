"""Exact-only learning and replay tests for neural procedural memory."""

import json

import pytest

from hyper_arc.contender.agentic_memory import AgenticMemoryBank, AgenticMemoryRecord


TASK = {
    "train": [{"input": [[1, 1], [2, 2]], "output": [[1, 2], [2, 1]]}],
    "test": [{"input": [[3, 3], [4, 4]]}],
}
CODE = (
    "def transform(grid):\n"
    "    a = grid[0][0]\n"
    "    b = grid[1][0]\n"
    "    return [[a, b], [b, a]]"
)


def test_exact_agentic_memory_replays_only_after_current_demo_verification(tmp_path):
    record = AgenticMemoryRecord.create(
        source_task_id="source",
        summary="alternate row colors",
        code=CODE,
        model="model",
        task_data=TASK,
        validation_scope="builder",
        expected_test_outputs=[[[3, 4], [4, 3]]],
    )
    bank = AgenticMemoryBank((record,))
    path = tmp_path / "memory.json"
    bank.save(path)

    restored = AgenticMemoryBank.load(path)
    candidates = restored.exact_candidates(TASK)

    assert len(candidates) == 1
    assert candidates[0].test_predictions == (((3, 4), (4, 3)),)
    unrelated = {
        "train": [{"input": [[1]], "output": [[9]]}],
        "test": [{"input": [[2]]}],
    }
    assert restored.exact_candidates(unrelated) == ()


def test_agentic_memory_rejects_tampering():
    record = AgenticMemoryRecord.create(
        source_task_id="source",
        summary="alternate",
        code=CODE,
        model="model",
        task_data=TASK,
        validation_scope="builder",
        expected_test_outputs=[[[3, 4], [4, 3]]],
    )
    payload = AgenticMemoryBank((record,)).to_dict()
    payload["records"][0]["code"] += "\n"
    with pytest.raises(ValueError, match="digest"):
        AgenticMemoryBank.from_dict(json.loads(json.dumps(payload)))

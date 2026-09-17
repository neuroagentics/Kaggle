"""Canonical contender record tests."""

from __future__ import annotations

from hyper_arc.contender.schemas import MemoryRecord, ProgramAST, ValueType


def test_program_digest_is_stable_across_argument_order():
    first = ProgramAST(
        op="translate",
        output_type=ValueType.GRID,
        arguments={"dy": -1, "dx": 2},
    )
    second = ProgramAST(
        op="translate",
        output_type=ValueType.GRID,
        arguments={"dx": 2, "dy": -1},
    )
    different = ProgramAST(
        op="translate",
        output_type=ValueType.GRID,
        arguments={"dx": 1, "dy": -1},
    )

    assert first.digest == second.digest
    assert first.digest != different.digest


def test_memory_record_digest_excludes_no_implicit_geometry():
    program = ProgramAST(op="identity", output_type=ValueType.GRID)
    record = MemoryRecord(
        record_id="record-1",
        source_task_ids=("task-1",),
        task_signature="one-object",
        program=program,
        bindings={},
        invariants=("palette-invariant",),
        failure_modes=(),
        exact_replay=True,
        validation_scope="development",
    )

    assert len(record.digest) == 64
    assert not hasattr(record, "embedding")
    assert not hasattr(record, "coordinates")

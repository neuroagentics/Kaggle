import math

from hyper_arc.contender.experience_bank import ExperienceFingerprint
from hyper_arc.contender.hyperbolic_memory import (
    HyperbolicWorldMemory,
    fingerprint_point,
    poincare_distance,
    program_point,
)
from hyper_arc.contender.schemas import ProgramAST, ValueType


def _program(degrees: int) -> ProgramAST:
    return ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": degrees},
        children=(ProgramAST(op="input", output_type=ValueType.GRID),),
    )


def test_embeddings_are_inside_the_ball_and_self_distance_is_zero():
    fingerprint = ExperienceFingerprint(
        features={"train_count": 3.0, "object_delta": -1.0},
        tags=("dimension:contract",),
    )
    point = fingerprint_point(fingerprint)
    assert math.sqrt(sum(value * value for value in point)) < 1.0
    assert poincare_distance(point, point) == 0.0


def test_program_embedding_preserves_arguments_and_tree_order():
    rotate_90 = _program(90)
    rotate_180 = _program(180)
    nested = ProgramAST(
        op="reflect",
        output_type=ValueType.GRID,
        arguments={"axis": "vertical"},
        children=(rotate_90,),
    )
    reversed_order = ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": 90},
        children=(
            ProgramAST(
                op="reflect",
                output_type=ValueType.GRID,
                arguments={"axis": "vertical"},
                children=(ProgramAST(op="input", output_type=ValueType.GRID),),
            ),
        ),
    )
    assert program_point(rotate_90) != program_point(rotate_180)
    assert program_point(nested) != program_point(reversed_order)


def test_retrieval_is_task_conditioned_and_round_trips(tmp_path):
    near = ExperienceFingerprint(
        features={"train_count": 2.0, "object_delta": 0.0},
        tags=("dimension:same",),
    )
    far = ExperienceFingerprint(
        features={"train_count": 4.0, "object_delta": 8.0},
        tags=("dimension:expand:3x3", "output:separated"),
    )
    memory = HyperbolicWorldMemory()
    expected = memory.remember(
        source_task_id="near",
        family="test",
        fingerprint=near,
        program=_program(90),
        demonstration_exact=True,
        test_exact=True,
    )
    memory.remember(
        source_task_id="far",
        family="test",
        fingerprint=far,
        program=_program(180),
        demonstration_exact=True,
        test_exact=True,
    )
    assert (
        memory.retrieve(near, require_program=True)[0].item.record_id
        == expected.record_id
    )

    path = tmp_path / "memory.json"
    memory.save(path)
    restored = HyperbolicWorldMemory.load(path)
    assert restored.to_dict() == memory.to_dict()
    assert restored.retrieve(near)[0].item.program == _program(90)

from hyper_arc.contender.experience_bank import task_fingerprint_v2
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.schemas import ProgramAST, ValueType
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
    _attempt_selection_order,
)
from hyper_arc.contender.schemas import Hypothesis


def _hypothesis(hypothesis_id, source):
    program = ProgramAST(op="input", output_type=ValueType.GRID)
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        program=program,
        traces=(),
        residuals=(),
        exact_replay=True,
        complexity=1,
        confidence=0.8,
        channel="test",
        provenance={"source": source},
    )


def test_experimental_family_cannot_displace_promoted_attempts():
    ranked = [
        _hypothesis("new-a", "structural-world"),
        _hypothesis("new-b", "structural-world"),
        _hypothesis("old-a", "object-world"),
        _hypothesis("old-b", "relational-world"),
    ]
    ordered = _attempt_selection_order(ranked)
    assert [item.hypothesis_id for item in ordered] == [
        "old-a",
        "old-b",
        "new-a",
        "new-b",
    ]


def test_recursive_reasoning_repairs_geometry_with_color_binding():
    task = {
        "train": [
            {
                "input": [[1, 0], [2, 0]],
                "output": [[3, 4], [0, 0]],
            },
            {
                "input": [[2, 1], [0, 0]],
                "output": [[0, 3], [0, 4]],
            },
        ],
        "test": [{"input": [[1, 2], [0, 0]]}],
    }
    result = RecursiveReasoningWorldModel(
        config=WorldModelConfig(max_depth=2, max_object_seeds=0)
    ).solve("rotate-remap", task)
    assert result.hypotheses
    assert any(item.provenance["repair_depth"] > 0 for item in result.hypotheses)
    assert result.test_predictions[0][0] == ((0, 4), (0, 3))
    assert all(item.exact_replay for item in result.hypotheses)


def test_hyperbolic_memory_program_is_executed_as_a_seed():
    task = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }
    rotate = ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": 90},
        children=(ProgramAST(op="input", output_type=ValueType.GRID),),
    )
    memory = HyperbolicWorldMemory()
    memory.remember(
        source_task_id="prior",
        family="geometry",
        fingerprint=task_fingerprint_v2(task),
        program=rotate,
        demonstration_exact=True,
        test_exact=True,
    )
    result = RecursiveReasoningWorldModel(
        memory,
        config=WorldModelConfig(max_depth=0, max_object_seeds=0),
    ).solve("memory-seed", task)
    assert result.reasoning_states[0].source == "hyperbolic-memory"
    assert result.hypotheses[0].provenance["source"] == "hyperbolic-memory"
    assert result.test_predictions[0][0] == ((7, 5), (8, 6))


def test_pass_two_keeps_distinct_exact_demo_hypotheses():
    task = {
        "train": [{"input": [[1, 1], [1, 1]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[1, 2], [3, 4]]}],
    }
    result = RecursiveReasoningWorldModel(
        config=WorldModelConfig(max_depth=0, max_object_seeds=0, attempts=2)
    ).solve("ambiguous", task)
    assert len(result.test_predictions[0]) == 2
    assert result.test_predictions[0][0] != result.test_predictions[0][1]
    assert all(item.exact_replay for item in result.hypotheses)


def test_non_exact_programs_never_become_hypotheses():
    task = {
        "train": [{"input": [[1, 0]], "output": [[1, 1, 1]]}],
        "test": [{"input": [[2, 0]]}],
    }
    result = RecursiveReasoningWorldModel(
        config=WorldModelConfig(max_depth=0, max_object_seeds=0)
    ).solve("no-fit", task)
    assert result.hypotheses == ()
    assert result.test_predictions == ((),)


def test_memory_can_be_disabled_for_ablation():
    task = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }
    memory = HyperbolicWorldMemory()
    memory.remember(
        source_task_id="prior",
        family="geometry",
        fingerprint=task_fingerprint_v2(task),
        program=ProgramAST(
            op="reflect",
            output_type=ValueType.GRID,
            arguments={"axis": "vertical"},
            children=(ProgramAST(op="input", output_type=ValueType.GRID),),
        ),
        demonstration_exact=True,
    )
    before = len(memory.items)
    result = RecursiveReasoningWorldModel(
        memory,
        config=WorldModelConfig(
            max_depth=0,
            max_object_seeds=0,
            retrieve_memory=False,
            learn_memory=False,
        ),
    ).solve("ablation", task)
    assert result.memory_neighbors == ()
    assert all(state.source != "hyperbolic-memory" for state in result.reasoning_states)
    assert len(memory.items) == before

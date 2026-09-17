"""Tests for task-state-derived structural rules."""

from hyper_arc.contender.executor import TypedExecutor
from hyper_arc.contender.perception import perceive_task
from hyper_arc.contender.structural_rules import generate_structural_programs
from hyper_arc.contender.world_model import RecursiveReasoningWorldModel, WorldModelConfig


def test_component_area_rule_recolors_only_the_largest_object():
    task = {
        "train": [
            {
                "input": [[1, 0, 0, 2, 2, 0], [0, 0, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[1, 0, 0, 3, 3, 0], [0, 0, 0, 3, 3, 0], [0, 0, 0, 0, 0, 0]],
            },
            {
                "input": [[4, 4, 0, 0, 5, 0], [4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[3, 3, 0, 0, 5, 0], [3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            },
        ],
        "test": [{"input": [[6, 0, 0, 7, 7, 0], [0, 0, 0, 7, 7, 0], [0, 0, 0, 0, 0, 0]]}],
    }
    state = perceive_task("area-recolor", task)
    candidates = generate_structural_programs(state)
    predictions = {
        TypedExecutor().execute(candidate.program, task["test"][0]["input"]).value
        for candidate in candidates
    }
    assert ((6, 0, 0, 3, 3, 0), (0, 0, 0, 3, 3, 0), (0, 0, 0, 0, 0, 0)) in predictions


def test_structural_programs_reject_mixed_color_output_inside_one_object():
    task = {
        "train": [
            {
                "input": [[1, 1], [0, 0]],
                "output": [[2, 3], [0, 0]],
            }
        ],
        "test": [{"input": [[4, 4], [0, 0]]}],
    }
    assert generate_structural_programs(perceive_task("mixed", task)) == ()


def test_world_model_records_structural_lineage_and_exact_replay():
    task = {
        "train": [
            {
                "input": [[1, 0, 0, 2, 2, 0], [0, 0, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[1, 0, 0, 3, 3, 0], [0, 0, 0, 3, 3, 0], [0, 0, 0, 0, 0, 0]],
            },
            {
                "input": [[4, 4, 0, 0, 5, 0], [4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                "output": [[3, 3, 0, 0, 5, 0], [3, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            },
        ],
        "test": [{"input": [[6, 0, 0, 7, 7, 0], [0, 0, 0, 7, 7, 0], [0, 0, 0, 0, 0, 0]]}],
    }
    result = RecursiveReasoningWorldModel(
        config=WorldModelConfig(
            max_depth=0,
            max_memory_seeds=0,
            max_object_seeds=0,
            max_relational_seeds=0,
            retrieve_memory=False,
            learn_memory=False,
        )
    ).solve("structural-lineage", task)
    structural = [
        hypothesis
        for hypothesis in result.hypotheses
        if hypothesis.provenance["source"] == "structural-world"
    ]
    assert structural
    assert all(hypothesis.exact_replay for hypothesis in structural)
    assert any(
        prediction == ((6, 0, 0, 3, 3, 0), (0, 0, 0, 3, 3, 0), (0, 0, 0, 0, 0, 0))
        for prediction in result.test_predictions[0]
    )

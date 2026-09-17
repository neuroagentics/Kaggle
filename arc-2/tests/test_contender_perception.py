"""Canonical object/relation perception tests."""

from __future__ import annotations

from hyper_arc.contender.perception import perceive_grid, perceive_task


def test_perception_extracts_objects_holes_and_relations():
    state = perceive_grid(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 0],
            [0, 1, 1, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0, 0],
            [0, 3, 0, 3, 0, 0],
            [0, 3, 3, 3, 0, 0],
        ],
        "fixture",
    )

    assert len(state.objects) == 3
    assert sorted(item.area for item in state.objects) == [2, 4, 8]
    assert max(item.holes for item in state.objects) == 1
    assert any(relation.kind == "left_of" for relation in state.relations)
    assert all(item.grid_ref == "fixture" for item in state.objects)


def test_relation_graph_is_sparse_for_repeated_objects():
    grid = [[0 for _ in range(20)] for _ in range(20)]
    for row in range(0, 20, 2):
        for col in range(0, 20, 2):
            grid[row][col] = 1

    state = perceive_grid(grid, "sparse")

    assert len(state.objects) == 100
    assert len(state.relations) <= len(state.objects) * 3


def test_signature_is_invariant_to_color_permutation_and_common_translation():
    original = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    recolored = [[{0: 9, 1: 7, 2: 3}[cell] for cell in row] for row in original]
    translated = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 2, 0],
    ]

    signature = perceive_grid(original, "original").canonical_signature
    assert perceive_grid(recolored, "recolored").canonical_signature == signature
    assert perceive_grid(translated, "translated").canonical_signature == signature


def test_ambiguous_background_tie_is_color_invariant():
    original = [[1, 1, 2, 2], [1, 1, 2, 2]]
    recolored = [[8 if cell == 1 else 3 for cell in row] for row in original]

    assert (
        perceive_grid(original, "original").canonical_signature
        == perceive_grid(recolored, "recolored").canonical_signature
    )


def test_connectivity_is_explicit():
    grid = [[1, 0], [0, 1]]
    assert len(perceive_grid(grid, "four", connectivity=4).objects) == 2
    assert len(perceive_grid(grid, "eight", connectivity=8).objects) == 1


def test_grid_features_include_periodicity_and_separators():
    state = perceive_grid(
        [[1, 2, 1, 2], [0, 0, 0, 0], [1, 2, 1, 2], [0, 0, 0, 0]],
        "pattern",
    )

    assert ("rows", 2) in state.periodicity
    assert ("columns", 2) in state.periodicity
    assert state.separator_rows == (1, 3)


def test_task_perception_records_exact_shape_correspondence_and_delta():
    task = {
        "train": [
            {
                "input": [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 0, 0], [2, 2, 0]],
            }
        ],
        "test": [{"input": [[0, 3, 3], [0, 0, 0], [0, 0, 0]]}],
    }

    state = perceive_task("delta", task, provenance={"split": "development"})

    assert len(state.grids) == 3
    assert len(state.deltas) == 1
    delta = state.deltas[0]
    assert len(delta.correspondences) == 1
    assert delta.correspondences[0].translation == (2, -1)
    assert delta.correspondences[0].color_changed is True
    assert not delta.added_object_ids
    assert not delta.removed_object_ids

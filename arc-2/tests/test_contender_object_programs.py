"""Exact object-program generation tests."""

from __future__ import annotations

from hyper_arc.contender.object_programs import exact_object_candidates


def test_object_programs_crop_the_largest_component():
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 0, 2, 2],
                    [0, 0, 0, 2, 2],
                    [3, 3, 3, 0, 0],
                    [3, 3, 3, 0, 0],
                ],
                "output": [[3, 3, 3], [3, 3, 3]],
            }
        ],
        "test": [
            {
                "input": [
                    [4, 0, 5, 5, 0],
                    [0, 0, 5, 5, 0],
                    [6, 6, 6, 0, 0],
                    [6, 6, 6, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            }
        ],
    }

    candidates = exact_object_candidates(task)

    assert candidates
    assert any(
        candidate.predict(task["test"][0]["input"])
        == [[6, 6, 6], [6, 6, 6]]
        for candidate in candidates
    )


def test_object_programs_move_and_recolor_selected_component():
    task = {
        "train": [
            {
                "input": [[0, 1, 1, 0], [0, 0, 0, 0], [0, 2, 0, 0]],
                "output": [[0, 0, 0, 0], [0, 3, 3, 0], [0, 2, 0, 0]],
            }
        ],
        "test": [
            {"input": [[0, 4, 4, 0], [0, 0, 0, 0], [0, 2, 0, 0]]}
        ],
    }

    candidates = exact_object_candidates(task)

    assert any(candidate.name.startswith("move-recolor") for candidate in candidates)
    assert any(
        candidate.predict(task["test"][0]["input"])
        == [[0, 0, 0, 0], [0, 3, 3, 0], [0, 2, 0, 0]]
        for candidate in candidates
    )

"""Closed relational-plan generation and invariance tests."""

from __future__ import annotations

import json

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.contender.relational_primitives import (
    edge_marker_motif_propagation,
    keyed_object_frames,
)


def test_edge_marker_propagation_is_color_permutation_invariant():
    source = (
        (0, 1, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 2, 2, 2, 0, 0, 0),
        (0, 2, 0, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 0, 0),
    )
    expected = (
        (0, 2, 2, 2, 0, 0, 0),
        (0, 2, 0, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 0, 0),
        (0, 2, 0, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 0, 0),
    )
    assert edge_marker_motif_propagation(source) == expected
    recolored = tuple(
        tuple({1: 6, 2: 8}.get(value, value) for value in row) for row in source
    )
    recolored_expected = tuple(
        tuple({2: 8}.get(value, value) for value in row) for row in expected
    )
    assert edge_marker_motif_propagation(recolored) == recolored_expected


def test_keyed_frames_follow_key_colors_not_fixed_palette():
    source = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (3, 4, 0, 0, 3, 3, 0, 0),
        (3, 4, 0, 0, 3, 3, 0, 0),
        (0, 0, 0, 0, 3, 3, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )
    result = keyed_object_frames(source)
    assert result[0][3:7] == (4, 4, 4, 4)
    assert result[4][3:7] == (4, 4, 4, 4)


def test_relational_v2_expresses_designated_development_cases():
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    solutions = json.loads(
        open("data/arc-agi-2/arc-agi_training_solutions.json", encoding="utf-8").read()
    )
    expected_kinds = {
        "c62e2108": "edge-marker-motif-propagation",
        "5adee1b2": "keyed-object-frames",
        "db118e2a": "frame-signal-repeat",
    }
    for task_id, expected_kind in expected_kinds.items():
        candidates = exact_relational_candidates(development[task_id])
        assert any(candidate.kind == expected_kind for candidate in candidates)
        assert any(
            all(
                candidate.predict(pair["input"]) == target
                for pair, target in zip(
                    development[task_id]["test"], solutions[task_id]
                )
            )
            for candidate in candidates
            if candidate.kind == expected_kind
        )

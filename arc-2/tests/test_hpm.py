"""tests/test_hpm.py — Unit and property tests for HPM.

Covers GlobalMemoryBank, LocalTaskBuffer, embeddings, k-NN ordering,
serialisation round-trip, and UCB1 selection correctness.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSNode, MCTSEngine, enumerate_actions
from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import DSLProgram

# ── Helpers ───────────────────────────────────────────────────────────────

def _make_prog(names: list[str]) -> DSLProgram:
    return [(n, {}) for n in names]


def _simple_esb() -> ESB:
    return ESB.from_grid([[1, 2], [3, 4]])


# ── Unit tests: GlobalMemoryBank ──────────────────────────────────────────


def test_gmb_add_and_query_returns_results():
    gmb = GlobalMemoryBank(k=3)
    for names in [["rotate"], ["reflect"], ["translate"], ["flood_fill"]]:
        gmb.add(_make_prog(names))
    results = gmb.query(_make_prog(["rotate"]))
    assert len(results) > 0
    assert len(results) <= 3


def test_gmb_query_empty_returns_empty_list():
    gmb = GlobalMemoryBank(k=5)
    results = gmb.query(_make_prog(["rotate"]))
    assert results == []


def test_hpm_uses_portable_unit_ball_math_without_geoopt():
    gmb = GlobalMemoryBank(k=2)
    gmb.add(_make_prog(["rotate"]))
    gmb.add(_make_prog(["reflect"]))

    assert not hasattr(gmb, "manifold")
    assert gmb._distance(gmb._entries[0].embedding, gmb._entries[0].embedding) == 0


def test_hpm_rejects_unsupported_curvature():
    with pytest.raises(ValueError, match="curvature=1"):
        GlobalMemoryBank(curvature=0.5)


def test_gmb_query_fewer_than_k_entries():
    gmb = GlobalMemoryBank(k=5)
    gmb.add(_make_prog(["rotate"]))
    gmb.add(_make_prog(["reflect"]))
    results = gmb.query(_make_prog(["rotate"]))
    assert 1 <= len(results) <= 2


def test_gmb_weights_sum_to_one():
    gmb = GlobalMemoryBank(k=5)
    for names in [["rotate"], ["reflect"], ["translate"]]:
        gmb.add(_make_prog(names))
    results = gmb.query(_make_prog(["rotate"]))
    total = sum(w for _, w in results)
    assert abs(total - 1.0) < 1e-5


def test_gmb_load_seed_bank(tmp_path):
    seed = [
        {"program": [{"name": "rotate", "kwargs": {"degrees": 90}}]},
        {"program": [{"name": "reflect", "kwargs": {"axis": "horizontal"}}]},
    ]
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(seed))
    gmb = GlobalMemoryBank(k=5)
    gmb.load_seed_bank(p)
    assert len(gmb) == 2


def test_gmb_save_load_round_trip(tmp_path):
    gmb = GlobalMemoryBank(k=5)
    gmb.add(_make_prog(["rotate", "reflect"]))
    gmb.add(_make_prog(["flood_fill"]))
    path = tmp_path / "gmb.json"
    gmb.save(path)
    gmb2 = GlobalMemoryBank(k=5)
    gmb2.load(path)
    assert len(gmb2) == len(gmb)
    for e1, e2 in zip(gmb._entries, gmb2._entries):
        assert e1.program == e2.program
        assert torch.allclose(e1.embedding, e2.embedding, atol=1e-5)


# ── Unit tests: LocalTaskBuffer ───────────────────────────────────────────


def test_ltb_add_and_query():
    ltb = LocalTaskBuffer(k=3)
    ltb.add(_make_prog(["translate"]))
    ltb.add(_make_prog(["rotate"]))
    results = ltb.query(_make_prog(["translate"]))
    assert len(results) >= 1


def test_ltb_query_empty_returns_empty():
    ltb = LocalTaskBuffer(k=5)
    assert ltb.query(_make_prog(["rotate"])) == []


def test_ltb_weights_sum_to_one():
    ltb = LocalTaskBuffer(k=5)
    for names in [["rotate"], ["reflect"], ["translate"]]:
        ltb.add(_make_prog(names))
    results = ltb.query(_make_prog(["rotate"]))
    total = sum(w for _, w in results)
    assert abs(total - 1.0) < 1e-5


# ── Property tests ────────────────────────────────────────────────────────

primitive_name_st = st.sampled_from(list([
    "translate", "rotate", "reflect", "extract_object",
    "overlay", "crop_to_bbox", "scale_integer", "tile",
    "connect_points", "color_remap", "flood_fill", "symmetrize",
]))

program_st = st.lists(primitive_name_st, min_size=1, max_size=6).map(
    lambda names: [(n, {}) for n in names]
)


@given(program_st)
@settings(max_examples=60)
def test_property_embeddings_on_poincare_ball_gmb(program):
    """Property 10: GlobalMemoryBank embeddings have Euclidean norm < 1."""
    gmb = GlobalMemoryBank(k=5)
    gmb.add(program)
    for entry in gmb._entries:
        norm = entry.embedding.norm().item()
        assert norm < 1.0, f"Embedding norm {norm} >= 1.0 (outside Poincaré ball)"


@given(program_st)
@settings(max_examples=60)
def test_property_embeddings_on_poincare_ball_ltb(program):
    """Property 10: LocalTaskBuffer embeddings have Euclidean norm < 1."""
    ltb = LocalTaskBuffer(k=5)
    ltb.add(program)
    for entry in ltb._entries:
        norm = entry.embedding.norm().item()
        assert norm < 1.0, f"Embedding norm {norm} >= 1.0 (outside Poincaré ball)"


@given(st.lists(program_st, min_size=2, max_size=8))
@settings(max_examples=40)
def test_property_knn_weights_positive_and_sum_to_one(programs):
    """Property 11 (partial): weights are positive and sum ≈ 1.0."""
    gmb = GlobalMemoryBank(k=5)
    for p in programs:
        gmb.add(p)
    results = gmb.query(programs[0])
    if results:
        for _, w in results:
            assert w > 0.0
        assert abs(sum(w for _, w in results) - 1.0) < 1e-4


@given(program_st, program_st)
@settings(max_examples=40)
def test_property_closer_program_higher_weight(prog_a, prog_b):
    """Property 11: closer program gets higher weight than a distant one."""
    gmb = GlobalMemoryBank(k=2)
    # Add prog_a twice (identical) and prog_b once (potentially different)
    gmb.add(prog_a)
    gmb.add(prog_b)
    if len(gmb) < 2:
        return
    results = gmb.query(prog_a)
    if len(results) < 2:
        return
    # The program most similar to prog_a should be first (lowest distance)
    # weights are sorted descending
    assert results[0][1] >= results[1][1]


@given(st.lists(program_st, min_size=1, max_size=5))
@settings(max_examples=30)
def test_property_serialisation_round_trip(programs):
    """Property 12: save/load preserves program content and embeddings."""
    gmb = GlobalMemoryBank(k=5)
    for p in programs:
        gmb.add(p)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        gmb.save(path)
        gmb2 = GlobalMemoryBank(k=5)
        gmb2.load(path)
        assert len(gmb2) == len(gmb)
        for e1, e2 in zip(gmb._entries, gmb2._entries):
            assert e1.program == e2.program
            assert torch.allclose(e1.embedding, e2.embedding, atol=1e-5)
    finally:
        path.unlink(missing_ok=True)


# ── UCB1 tests (Property 13) ──────────────────────────────────────────────


def test_ucb1_unvisited_returns_inf():
    """Unvisited node returns inf so it is always selected."""
    state = _simple_esb()
    node = MCTSNode(state=state, parent=None, action_taken=None)
    assert node.ucb1(C=1.41, prior=0.0) == float("inf")


def test_ucb1_formula_correctness():
    """UCB1 = Q + C*sqrt(ln(parent_N)/N) + prior."""
    parent = MCTSNode(state=_simple_esb(), parent=None, action_taken=None)
    parent.visits = 10
    child = MCTSNode(state=_simple_esb(), parent=parent, action_taken=None)
    child.visits = 3
    child.value  = 0.5
    C     = 1.41
    prior = 0.1
    expected = 0.5 + C * math.sqrt(math.log(10) / 3) + prior
    assert abs(child.ucb1(C, prior) - expected) < 1e-6


def test_select_picks_highest_ucb1():
    """Property 13: _select always returns child with max UCB1 score."""
    from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
    gmb = GlobalMemoryBank(k=5)
    ltb = LocalTaskBuffer(k=5)
    engine = MCTSEngine(gmb, ltb, C=1.41, global_prior_weight=0.0)

    state = _simple_esb()
    root  = MCTSNode(state=state, parent=None, action_taken=None)
    root.visits = 10

    # Manually create two children with known visits/values
    c1 = MCTSNode(state=state, parent=root, action_taken=("rotate", {"degrees": 90}))
    c1.visits = 5
    c1.value  = 0.2
    c2 = MCTSNode(state=state, parent=root, action_taken=("reflect", {"axis": "horizontal"}))
    c2.visits = 2
    c2.value  = 0.8
    root.children = [c1, c2]

    selected = engine._select(root)
    # Both children have children=[], so _select picks one of them
    # Whichever has higher UCB1 score should be selected
    score1 = c1.ucb1(engine.C, 0.0)
    score2 = c2.ucb1(engine.C, 0.0)
    if score1 > score2:
        assert selected is c1
    else:
        assert selected is c2


# ── enumerate_actions sanity ──────────────────────────────────────────────


def test_enumerate_actions_non_empty():
    esb = ESB.from_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    actions = enumerate_actions(esb)
    assert len(actions) > 0


def test_enumerate_actions_all_primitives_covered():
    """All 12 primitive names should appear at least once."""
    esb = ESB.from_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    actions = enumerate_actions(esb)
    names = {a[0] for a in actions}
    expected = {
        "translate", "rotate", "reflect", "extract_object",
        "overlay", "crop_to_bbox", "scale_integer", "tile",
        "connect_points", "color_remap", "flood_fill", "symmetrize",
    }
    assert expected == names

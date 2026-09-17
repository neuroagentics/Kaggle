"""tests/test_cost.py — Unit and property tests for cost functions."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from hyper_arc.esb import ESB
from hyper_arc.cost import exact_match, partial_reward

# ── Helpers ───────────────────────────────────────────────────────────────

grid_strategy = st.lists(
    st.lists(st.integers(0, 9), min_size=1, max_size=15),
    min_size=1,
    max_size=15,
).filter(lambda g: all(len(r) == len(g[0]) for r in g))


def make(grid):
    return ESB.from_grid(grid)


# ── Unit tests: exact_match ───────────────────────────────────────────────


def test_exact_match_identical():
    g = make([[1, 2], [3, 4]])
    assert exact_match(g, g) is True


def test_exact_match_equal_content():
    a = make([[1, 2], [3, 4]])
    b = make([[1, 2], [3, 4]])
    assert exact_match(a, b) is True


def test_exact_match_different_values():
    a = make([[1, 2], [3, 4]])
    b = make([[1, 2], [3, 5]])
    assert exact_match(a, b) is False


def test_exact_match_different_shapes():
    a = make([[1, 2], [3, 4]])
    b = make([[1, 2, 3]])
    assert exact_match(a, b) is False


def test_exact_match_one_differing_cell():
    a = make([[0, 0, 0], [0, 0, 0]])
    b = make([[0, 0, 0], [0, 0, 1]])
    assert exact_match(a, b) is False


# ── Unit tests: partial_reward ────────────────────────────────────────────


def test_partial_reward_full_match():
    g = make([[1, 2], [3, 4]])
    assert partial_reward(g, g) == 1.0


def test_partial_reward_no_match():
    a = make([[1, 1], [1, 1]])
    b = make([[2, 2], [2, 2]])
    assert partial_reward(a, b) == 0.0


def test_partial_reward_half_match():
    a = make([[1, 2]])
    b = make([[1, 9]])
    # 1 of 2 cells match
    assert partial_reward(a, b) == 0.5


def test_partial_reward_shape_mismatch():
    a = make([[1, 2]])
    b = make([[1, 2, 3]])
    assert partial_reward(a, b) == 0.0


def test_partial_reward_range():
    a = make([[1, 0], [0, 2]])
    b = make([[1, 1], [1, 1]])
    r = partial_reward(a, b)
    assert 0.0 <= r <= 1.0


# ── Property tests ────────────────────────────────────────────────────────


@given(grid_strategy)
@settings(max_examples=100)
def test_property_exact_match_identity(grid):
    """Property 16: exact_match(g, g) is always True."""
    g = make(grid)
    assert exact_match(g, g) is True


@given(grid_strategy, grid_strategy)
@settings(max_examples=100)
def test_property_exact_match_different_shapes(g1_grid, g2_grid):
    """Property 16: different shapes always return False."""
    g1 = make(g1_grid)
    g2 = make(g2_grid)
    if g1.data.shape != g2.data.shape:
        assert exact_match(g1, g2) is False


@given(grid_strategy, grid_strategy)
@settings(max_examples=100)
def test_property_partial_reward_range(g1_grid, g2_grid):
    """Property 17: partial_reward always in [0.0, 1.0]."""
    g1 = make(g1_grid)
    g2 = make(g2_grid)
    r = partial_reward(g1, g2)
    assert 0.0 <= r <= 1.0


@given(grid_strategy)
@settings(max_examples=100)
def test_property_partial_reward_self_is_one(grid):
    """Property 17: partial_reward(g, g) == 1.0."""
    g = make(grid)
    assert partial_reward(g, g) == 1.0

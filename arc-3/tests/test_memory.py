"""Tests for agent/memory.py — R06: bounded memory, evidence lineage, eviction.

All fixtures are synthetic. No ARC game data, trained models, or hidden engine
state is used. Fixture tests do not claim ARC-3 performance.
"""
from __future__ import annotations

import pytest
from agent.memory import (
    EPISODE_CAP,
    MECHANIC_CAP,
    GOAL_CAP,
    PROCEDURE_CAP,
    MEMORY_CEILING_BYTES,
    WorkingMemory,
    MechanicBelief,
    GoalHypothesis,
    ProcedureRecord,
    ImaginedBranch,
    _grid_to_bytes,
    _bytes_to_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID_2X2: tuple[tuple[int, ...], ...] = ((1, 2), (3, 4))
GRID_2X2B: tuple[tuple[int, ...], ...] = ((1, 2), (3, 5))  # one cell different


def make_memory(**kwargs) -> WorkingMemory:
    tick = [0.0]

    def clock():
        tick[0] += 0.001
        return tick[0]

    return WorkingMemory("test-game", clock=clock, **kwargs)


def add_episode(mem: WorkingMemory, *, level=0, success=False,
                predicted=GRID_2X2, observed=GRID_2X2, obs_id="obs0001"):
    return mem.add_episode(
        level_id=level,
        observation_id=obs_id,
        model_version="v1",
        action_id=3,
        action_x=None,
        action_y=None,
        before_grid=GRID_2X2,
        predicted_grid=predicted,
        observed_grid=observed,
        actual_success=success,
    )


# ---------------------------------------------------------------------------
# Grid codec
# ---------------------------------------------------------------------------

def test_grid_round_trip():
    grid = ((0, 15, 3), (7, 8, 1), (0, 0, 0))
    assert _bytes_to_grid(_grid_to_bytes(grid)) == grid


def test_grid_round_trip_single_cell():
    assert _bytes_to_grid(_grid_to_bytes(((5,),))) == ((5,),)


def test_grid_round_trip_max_size():
    grid = tuple(tuple(c % 16 for c in range(64)) for _ in range(64))
    assert _bytes_to_grid(_grid_to_bytes(grid)) == grid


# ---------------------------------------------------------------------------
# WorkingMemory construction
# ---------------------------------------------------------------------------

def test_construction_requires_game_id():
    with pytest.raises(ValueError, match="game_id"):
        WorkingMemory("")


def test_construction_rejects_zero_caps():
    with pytest.raises(ValueError):
        WorkingMemory("g", episode_cap=0)
    with pytest.raises(ValueError):
        WorkingMemory("g", goal_cap=0)


def test_construction_rejects_tiny_ceiling():
    with pytest.raises(ValueError, match="KiB"):
        WorkingMemory("g", ceiling_bytes=512)


# ---------------------------------------------------------------------------
# Episode records
# ---------------------------------------------------------------------------

def test_add_episode_returns_record_and_computes_error():
    mem = make_memory()
    ep = add_episode(mem, predicted=GRID_2X2, observed=GRID_2X2B)
    assert ep.changed_cell_error == 1
    assert ep.discrepancy_score == pytest.approx(0.25)
    assert ep.game_id == "test-game"
    assert ep.actual_success is False


def test_add_episode_rejects_non_bool_success():
    mem = make_memory()
    with pytest.raises(ValueError, match="bool"):
        mem.add_episode(
            level_id=0, observation_id="x", model_version="v1",
            action_id=3, action_x=None, action_y=None,
            before_grid=GRID_2X2, predicted_grid=GRID_2X2, observed_grid=GRID_2X2,
            actual_success="True",
        )


def test_episode_grid_round_trips_correctly():
    mem = make_memory()
    ep = add_episode(mem, predicted=GRID_2X2B, observed=GRID_2X2)
    assert ep.before_grid() == GRID_2X2
    assert ep.predicted_grid() == GRID_2X2B
    assert ep.observed_grid() == GRID_2X2


def test_episode_count_capped_at_limit():
    mem = make_memory(episode_cap=3)
    ids = [add_episode(mem, obs_id=f"obs{i:04d}").record_id for i in range(5)]
    assert len(mem._episodes) == 3
    # Oldest two evicted
    assert ids[0] not in mem._episodes
    assert ids[1] not in mem._episodes
    assert ids[2] in mem._episodes


def test_episode_query_filters_by_level():
    mem = make_memory()
    e0 = add_episode(mem, level=0, obs_id="obs0000")
    e1 = add_episode(mem, level=1, obs_id="obs0001")
    results = mem.query_episodes(level_id=0)
    assert e0 in results
    assert e1 not in results


def test_episode_query_filters_by_success():
    mem = make_memory()
    ep_f = add_episode(mem, success=False, obs_id="obs0001")
    ep_t = add_episode(mem, success=True, obs_id="obs0002")
    successes = mem.query_episodes(min_success=True)
    assert ep_t in successes
    assert ep_f not in successes


def test_episode_query_newest_first():
    mem = make_memory()
    ids = [add_episode(mem, obs_id=f"obs{i:04d}").record_id for i in range(4)]
    result_ids = [ep.record_id for ep in mem.query_episodes()]
    assert result_ids[0] == ids[-1]  # newest first


# ---------------------------------------------------------------------------
# Byte ceiling and eviction
# ---------------------------------------------------------------------------

def test_byte_accounting_increases_after_add():
    mem = make_memory()
    before = mem.bytes_used()
    add_episode(mem)
    assert mem.bytes_used() > before


def test_eviction_triggered_by_byte_ceiling():
    # Use a tiny ceiling that forces eviction
    mem = make_memory(ceiling_bytes=2048, episode_cap=100)
    # Add episodes until we exceed 2 KiB; eviction should keep us under
    for i in range(30):
        add_episode(mem, obs_id=f"obs{i:04d}")
    assert mem.bytes_used() <= mem.ceiling_bytes


def test_eviction_invalidates_unsupported_belief():
    mem = make_memory(episode_cap=2, ceiling_bytes=2048)
    ep1 = add_episode(mem, obs_id="obs0001")
    belief = mem.add_belief(
        hypothesis="obj moves right",
        supporting_ids=[ep1.record_id],
    )
    # Add two more episodes to force ep1 off the count cap
    add_episode(mem, obs_id="obs0002")
    add_episode(mem, obs_id="obs0003")
    # ep1 should be gone; belief should now be 'unsupported'
    assert ep1.record_id not in mem._episodes
    updated = mem._beliefs[belief.belief_id]
    assert updated.status == "unsupported"
    assert ep1.record_id not in updated.supporting_ids


def test_eviction_invalidates_procedure_when_all_support_evicted():
    mem = make_memory(episode_cap=2)
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    proc = mem.add_procedure(
        name="move right twice",
        preconditions=["avatar_exists"],
        steps=["action3", "action3"],
        supporting_ids=[ep1.record_id, ep2.record_id],
    )
    assert proc.status == "active"
    # Force eviction by adding 2 more (cap=2 pushes ep1 and ep2 out)
    add_episode(mem, obs_id="obs0003")
    add_episode(mem, obs_id="obs0004")
    updated = mem._procedures[proc.procedure_id]
    assert updated.status == "invalidated"


# ---------------------------------------------------------------------------
# Mechanic beliefs
# ---------------------------------------------------------------------------

def test_add_belief_requires_valid_evidence_ids():
    mem = make_memory()
    with pytest.raises(ValueError, match="not found"):
        mem.add_belief(hypothesis="x", supporting_ids=["a" * 32])


def test_add_belief_provisional_with_one_support():
    mem = make_memory()
    ep = add_episode(mem)
    b = mem.add_belief(hypothesis="move right", supporting_ids=[ep.record_id])
    assert b.status == "provisional"
    assert b.revision == 1


def test_contradict_belief_increments_revision():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    b = mem.add_belief(hypothesis="move right", supporting_ids=[ep1.record_id])
    b2 = mem.contradict_belief(b.belief_id, contradicting_ids=[ep2.record_id])
    assert b2.status == "contradicted"
    assert b2.revision == 2
    assert ep1.record_id in b2.supporting_ids
    assert ep2.record_id in b2.contradicting_ids


def test_contradict_belief_preserves_old_hypothesis():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    b = mem.add_belief(hypothesis="original", supporting_ids=[ep1.record_id])
    b2 = mem.contradict_belief(b.belief_id, contradicting_ids=[ep2.record_id])
    assert b2.hypothesis == "original"


def test_contradict_belief_replaces_hypothesis_when_given():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    b = mem.add_belief(hypothesis="original", supporting_ids=[ep1.record_id])
    b2 = mem.contradict_belief(b.belief_id, contradicting_ids=[ep2.record_id],
                               new_hypothesis="revised")
    assert b2.hypothesis == "revised"


def test_belief_count_cap():
    mem = make_memory(mechanic_cap=2)
    ep = add_episode(mem)
    for i in range(3):
        mem.add_belief(hypothesis=f"h{i}", supporting_ids=[ep.record_id])
    assert len(mem._beliefs) == 2


def test_belief_query_by_mechanic_tags():
    mem = make_memory()
    ep = add_episode(mem)
    b1 = mem.add_belief(hypothesis="push", mechanic_tags=["push"])
    b2 = mem.add_belief(hypothesis="pull", mechanic_tags=["pull"],
                        supporting_ids=[ep.record_id])
    results = mem.query_beliefs(mechanic_tags=["push"])
    assert b1 in results
    assert b2 not in results or results.index(b1) < results.index(b2)


def test_belief_revision_chain_preserved():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    b1 = mem.add_belief(hypothesis="v1", supporting_ids=[ep1.record_id])
    b2 = mem.add_belief(hypothesis="v2", supporting_ids=[ep1.record_id],
                        revision_of=b1.belief_id)
    assert b2.revision_of == b1.belief_id


def test_belief_revision_of_nonexistent_raises():
    mem = make_memory()
    with pytest.raises(ValueError, match="not found"):
        mem.add_belief(hypothesis="x", revision_of="a" * 32)


# ---------------------------------------------------------------------------
# Goal hypotheses
# ---------------------------------------------------------------------------

def test_goal_provisional_on_first_support():
    mem = make_memory()
    ep = add_episode(mem)
    g = mem.add_goal(
        description="reach green",
        observable_target="green_cell_at_exit",
        disconfirming_test="green_cell_not_at_exit",
        supporting_ids=[ep.record_id],
    )
    assert g.status == "provisional"


def test_goal_active_on_second_independent_support():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    g = mem.add_goal(
        description="reach green",
        observable_target="green_cell_at_exit",
        disconfirming_test="green_cell_not_at_exit",
        supporting_ids=[ep1.record_id, ep2.record_id],
    )
    assert g.status == "active"


def test_promote_goal_to_active():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    g = mem.add_goal(
        description="reach green",
        observable_target="green_cell_at_exit",
        disconfirming_test="green_cell_not_at_exit",
        supporting_ids=[ep1.record_id],
    )
    assert g.status == "provisional"
    g2 = mem.promote_goal(g.goal_id, new_supporting_id=ep2.record_id)
    assert g2.status == "active"


def test_promote_goal_duplicate_evidence_rejected():
    mem = make_memory()
    ep = add_episode(mem)
    g = mem.add_goal(
        description="g",
        observable_target="t",
        disconfirming_test="d",
        supporting_ids=[ep.record_id],
    )
    with pytest.raises(ValueError, match="already registered"):
        mem.promote_goal(g.goal_id, new_supporting_id=ep.record_id)


def test_falsify_goal():
    mem = make_memory()
    g = mem.add_goal(description="g", observable_target="t", disconfirming_test="d")
    g2 = mem.falsify_goal(g.goal_id)
    assert g2.status == "falsified"


def test_goal_count_cap():
    mem = make_memory(goal_cap=2)
    for i in range(3):
        mem.add_goal(description=f"g{i}", observable_target="t", disconfirming_test="d")
    assert len(mem._goals) == 2


def test_goal_requires_nonempty_fields():
    mem = make_memory()
    with pytest.raises(ValueError, match="observable_target"):
        GoalHypothesis(
            goal_id="a" * 32, game_id="g", level_context=None,
            description="x", observable_target="   ",
            disconfirming_test="d", supporting_ids=(),
            status="provisional", schema_version="1.0", timestamp=0.0,
        )


# ---------------------------------------------------------------------------
# Procedure records
# ---------------------------------------------------------------------------

def test_procedure_provisional_on_one_support():
    mem = make_memory()
    ep = add_episode(mem)
    p = mem.add_procedure(
        name="move_right",
        preconditions=["avatar_free"],
        steps=["action3"],
        supporting_ids=[ep.record_id],
    )
    assert p.status == "provisional"


def test_procedure_active_on_two_supports():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    p = mem.add_procedure(
        name="move_right",
        preconditions=["avatar_free"],
        steps=["action3"],
        supporting_ids=[ep1.record_id, ep2.record_id],
    )
    assert p.status == "active"


def test_promote_procedure():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    p = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                          supporting_ids=[ep1.record_id])
    assert p.status == "provisional"
    p2 = mem.promote_procedure(p.procedure_id, new_supporting_id=ep2.record_id)
    assert p2.status == "active"


def test_invalidate_procedure():
    mem = make_memory()
    p = mem.add_procedure(name="p", preconditions=[], steps=["a"])
    assert p.status == "provisional"
    p2 = mem.invalidate_procedure(p.procedure_id)
    assert p2.status == "invalidated"


def test_procedure_requires_steps():
    with pytest.raises(ValueError, match="step"):
        ProcedureRecord(
            procedure_id="a" * 32, game_id="g", level_context=None,
            name="empty", preconditions=(), parameters=(),
            steps=(),          # ← empty
            milestones=(), postconditions=(), exceptions=(),
            supporting_ids=(), status="provisional",
            schema_version="1.0", timestamp=0.0,
        )


def test_procedure_chaining():
    mem = make_memory()
    ep1 = add_episode(mem, obs_id="obs0001")
    ep2 = add_episode(mem, obs_id="obs0002")
    p1 = mem.add_procedure(
        name="step1",
        preconditions=["start"],
        steps=["action3"],
        postconditions=["at_midpoint"],
        supporting_ids=[ep1.record_id, ep2.record_id],
    )
    ep3 = add_episode(mem, obs_id="obs0003")
    ep4 = add_episode(mem, obs_id="obs0004")
    p2 = mem.add_procedure(
        name="step2",
        preconditions=["at_midpoint"],
        steps=["action4"],
        supporting_ids=[ep3.record_id, ep4.record_id],
    )
    chainable = mem.find_chainable(p1)
    assert p2 in chainable
    assert p1 not in chainable


def test_procedure_chaining_requires_compatible_postconditions():
    p1 = ProcedureRecord(
        procedure_id="a" * 32, game_id="g", level_context=None,
        name="p1", preconditions=(), parameters=(), steps=("a",),
        milestones=(), postconditions=("state_x",), exceptions=(),
        supporting_ids=(), status="active", schema_version="1.0", timestamp=0.0,
    )
    p2 = ProcedureRecord(
        procedure_id="b" * 32, game_id="g", level_context=None,
        name="p2", preconditions=("state_y",),  # incompatible
        parameters=(), steps=("b",),
        milestones=(), postconditions=(), exceptions=(),
        supporting_ids=(), status="active", schema_version="1.0", timestamp=0.0,
    )
    assert not p1.chains_with(p2)


def test_procedure_count_cap():
    mem = make_memory(procedure_cap=2)
    for i in range(3):
        mem.add_procedure(name=f"p{i}", preconditions=[], steps=[f"a{i}"])
    assert len(mem._procedures) == 2


# ---------------------------------------------------------------------------
# Imagined branches (current plan only)
# ---------------------------------------------------------------------------

def test_set_plan_branches_replaces_old():
    mem = make_memory()
    b1 = ImaginedBranch(
        branch_id="a" * 32, game_id="test-game", parent_branch_id=None,
        action_sequence=(3,), predicted_grids_bytes=(_grid_to_bytes(GRID_2X2),),
        utility_sequence=(1.0,), risk_sequence=(0.0,), uncertainty_sequence=(0.1,),
        terminal=False, schema_version="1.0",
    )
    b2 = ImaginedBranch(
        branch_id="b" * 32, game_id="test-game", parent_branch_id="a" * 32,
        action_sequence=(4,), predicted_grids_bytes=(_grid_to_bytes(GRID_2X2B),),
        utility_sequence=(2.0,), risk_sequence=(0.0,), uncertainty_sequence=(0.0,),
        terminal=True, schema_version="1.0",
    )
    mem.set_plan_branches([b1])
    assert len(mem.get_branches()) == 1
    mem.set_plan_branches([b2])
    assert len(mem.get_branches()) == 1
    assert mem.get_branches()[0].branch_id == "b" * 32


def test_branches_cleared_on_new_plan():
    mem = make_memory()
    b = ImaginedBranch(
        branch_id="c" * 32, game_id="test-game", parent_branch_id=None,
        action_sequence=(3,), predicted_grids_bytes=(_grid_to_bytes(GRID_2X2),),
        utility_sequence=(0.5,), risk_sequence=(0.1,), uncertainty_sequence=(0.2,),
        terminal=False, schema_version="1.0",
    )
    mem.set_plan_branches([b])
    mem.clear_branches()
    assert mem.get_branches() == []


def test_branch_rejects_wrong_game():
    mem = make_memory()
    b = ImaginedBranch(
        branch_id="d" * 32, game_id="other-game",  # ← wrong
        parent_branch_id=None,
        action_sequence=(3,), predicted_grids_bytes=(_grid_to_bytes(GRID_2X2),),
        utility_sequence=(1.0,), risk_sequence=(0.0,), uncertainty_sequence=(0.0,),
        terminal=False, schema_version="1.0",
    )
    with pytest.raises(ValueError, match="game_id"):
        mem.set_plan_branches([b])


def test_branch_non_finite_score_rejected():
    with pytest.raises(ValueError, match="finite"):
        ImaginedBranch(
            branch_id="e" * 32, game_id="g", parent_branch_id=None,
            action_sequence=(3,),
            predicted_grids_bytes=(_grid_to_bytes(GRID_2X2),),
            utility_sequence=(float("nan"),),
            risk_sequence=(0.0,), uncertainty_sequence=(0.0,),
            terminal=False, schema_version="1.0",
        )


def test_branch_length_mismatch_rejected():
    with pytest.raises(ValueError, match="grids count"):
        ImaginedBranch(
            branch_id="f" * 32, game_id="g", parent_branch_id=None,
            action_sequence=(3, 4),                            # 2 actions
            predicted_grids_bytes=(_grid_to_bytes(GRID_2X2),),  # 1 grid
            utility_sequence=(1.0, 1.0), risk_sequence=(0.0, 0.0),
            uncertainty_sequence=(0.0, 0.0),
            terminal=False, schema_version="1.0",
        )


# ---------------------------------------------------------------------------
# Game isolation
# ---------------------------------------------------------------------------

def test_game_isolation_separate_instances():
    mem_a = make_memory()
    mem_b = WorkingMemory("other-game")
    add_episode(mem_a)
    assert len(mem_b._episodes) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_report():
    mem = make_memory()
    add_episode(mem)
    stats = mem.stats()
    assert stats["episodes"] == 1
    assert stats["game_id"] == "test-game"
    assert stats["bytes_used"] > 0
    assert stats["ceiling_bytes"] == MEMORY_CEILING_BYTES

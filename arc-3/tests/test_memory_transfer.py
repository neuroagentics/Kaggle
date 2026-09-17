"""Tests for Step 4 memory-as-transfer using the REAL WorkingMemory.

Guards that a successful action sequence is stored with full context and can be
retrieved and reused in a different applicable situation (matching preconditions).
"""
from agent.memory import WorkingMemory


def _mk_memory_with_procedure():
    mem = WorkingMemory(game_id="cw_movement")
    g = ((0, 0), (0, 3))
    ep = mem.add_episode(
        level_id=0, observation_id="o0", model_version="cw",
        action_id=1, action_x=None, action_y=None,
        before_grid=g, predicted_grid=g, observed_grid=g,
        actual_success=True, prediction_available=True,
    )
    proc = mem.add_procedure(
        name="navigate_avatar_to_target",
        preconditions=("avatar_present", "target_reachable"),
        parameters=("avatar", "target"),
        steps=("move:dr>0", "move:dc>0"),
        milestones=("distance_to_target_decreases",),
        postconditions=("avatar_on_target",),
        exceptions=("avatar_blocked",),
        supporting_ids=(ep.record_id,),
    )
    return mem, proc


def test_procedure_stored_with_full_context():
    mem, proc = _mk_memory_with_procedure()
    assert proc.preconditions == ("avatar_present", "target_reachable")
    assert proc.milestones == ("distance_to_target_decreases",)   # expected intermediate
    assert proc.exceptions == ("avatar_blocked",)                 # failure condition
    assert len(proc.supporting_ids) == 1                          # supporting observation
    assert proc.steps  # non-empty action sequence


def test_procedure_retrievable_for_reuse_when_precondition_matches():
    mem, proc = _mk_memory_with_procedure()
    # A different applicable situation queries stored procedures; preconditions
    # are checked by the caller. Here we confirm the stored procedure is found.
    found = mem.query_procedures(status=None)
    assert any(p.procedure_id == proc.procedure_id for p in found)
    assert "avatar_present" in found[0].preconditions


def test_diverse_promotion_requires_second_distinct_success():
    mem, proc = _mk_memory_with_procedure()
    assert proc.status == "provisional"  # one success
    # second supporting observation from a DIFFERENT context promotes to active
    g2 = ((3, 0), (0, 0))
    ep2 = mem.add_episode(
        level_id=1, observation_id="o1", model_version="cw",
        action_id=2, action_x=None, action_y=None,
        before_grid=g2, predicted_grid=g2, observed_grid=g2,
        actual_success=True, prediction_available=True,
    )
    updated = mem.promote_procedure(proc.procedure_id, new_supporting_id=ep2.record_id)
    assert updated.status == "active"

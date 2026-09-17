"""Tests for R12: adapter safeguards + diverse-context procedure promotion.

Adapter-manager tests that touch the model require torch (present in dev venv).
Bookkeeping-only tests (caps, history) run regardless.
"""
from __future__ import annotations

import pytest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from agent.memory import WorkingMemory
from training.train_simulator import TrainingConfig


# ---------------------------------------------------------------------------
# Memory: diverse-context promotion + contradiction revocation
# ---------------------------------------------------------------------------

def _mem() -> WorkingMemory:
    tick = [0.0]

    def clock():
        tick[0] += 0.001
        return tick[0]

    return WorkingMemory("g", clock=clock)


def _episode(mem, *, level, success, obs):
    return mem.add_episode(
        level_id=level, observation_id=obs, model_version="v1",
        action_id=3, action_x=None, action_y=None,
        before_grid=((1, 2), (3, 4)), predicted_grid=((1, 2), (3, 4)),
        observed_grid=((1, 2), (3, 5)), actual_success=success,
    )


def test_same_context_two_successes_stays_provisional():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id])
    assert proc.status == "provisional"
    e2 = _episode(mem, level=0, success=True, obs="o2")  # SAME level
    updated = mem.promote_procedure_diverse(proc.procedure_id, new_supporting_id=e2.record_id)
    assert updated.status == "provisional"  # not promoted: same context


def test_diverse_context_two_successes_promotes():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id])
    e2 = _episode(mem, level=1, success=True, obs="o2")  # DIFFERENT level
    updated = mem.promote_procedure_diverse(proc.procedure_id, new_supporting_id=e2.record_id)
    assert updated.status == "active"


def test_promote_diverse_rejects_failure_evidence():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id])
    e2 = _episode(mem, level=1, success=False, obs="o2")
    with pytest.raises(ValueError, match="successful"):
        mem.promote_procedure_diverse(proc.procedure_id, new_supporting_id=e2.record_id)


def test_promote_diverse_rejects_duplicate_evidence():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id])
    with pytest.raises(ValueError, match="already registered"):
        mem.promote_procedure_diverse(proc.procedure_id, new_supporting_id=e1.record_id)


def test_contradiction_revokes_active_procedure():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    e2 = _episode(mem, level=1, success=True, obs="o2")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id, e2.record_id])
    assert proc.status == "active"
    fail = _episode(mem, level=0, success=False, obs="o3")
    revoked = mem.revoke_procedure_on_contradiction(
        proc.procedure_id, failing_episode_id=fail.record_id
    )
    assert revoked.status == "invalidated"
    assert fail.record_id in revoked.supporting_ids  # evidence preserved


def test_contradiction_requires_failure_evidence():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    proc = mem.add_procedure(name="p", preconditions=[], steps=["a"],
                             supporting_ids=[e1.record_id])
    good = _episode(mem, level=0, success=True, obs="o2")
    with pytest.raises(ValueError, match="UNsuccessful"):
        mem.revoke_procedure_on_contradiction(proc.procedure_id, failing_episode_id=good.record_id)


def test_revoked_procedure_not_chainable():
    mem = _mem()
    e1 = _episode(mem, level=0, success=True, obs="o1")
    e2 = _episode(mem, level=1, success=True, obs="o2")
    p1 = mem.add_procedure(name="p1", preconditions=[], steps=["a"],
                           postconditions=["mid"],
                           supporting_ids=[e1.record_id, e2.record_id])
    e3 = _episode(mem, level=0, success=True, obs="o3")
    e4 = _episode(mem, level=1, success=True, obs="o4")
    p2 = mem.add_procedure(name="p2", preconditions=["mid"], steps=["b"],
                           supporting_ids=[e3.record_id, e4.record_id])
    assert p2 in mem.find_chainable(p1)
    fail = _episode(mem, level=0, success=False, obs="o5")
    mem.revoke_procedure_on_contradiction(p2.procedure_id, failing_episode_id=fail.record_id)
    assert p2 not in mem.find_chainable(p1)  # invalidated -> not reusable


# ---------------------------------------------------------------------------
# AdapterManager bookkeeping (no torch needed for cap/history logic)
# ---------------------------------------------------------------------------

def test_adapter_manager_begin_decision_resets_counter():
    from training.adapter import AdapterManager
    cfg = TrainingConfig(adapter_max_updates_per_decision=1)
    mgr = AdapterManager(cfg)
    mgr._updates_this_decision = 5
    mgr.begin_decision()
    assert mgr._updates_this_decision == 0


def test_adapter_manager_rejects_regression_tolerance_below_one():
    from training.adapter import AdapterManager
    with pytest.raises(ValueError, match="regression_tolerance"):
        AdapterManager(TrainingConfig(), regression_tolerance=0.9)


def test_adapter_manager_initial_version_zero():
    from training.adapter import AdapterManager
    mgr = AdapterManager(TrainingConfig())
    assert mgr.adapter_version == 0
    assert mgr.stats()["accepted"] == 0


# ---------------------------------------------------------------------------
# AdapterManager with a real tiny model (torch)
# ---------------------------------------------------------------------------

pytestmark_torch = pytest.mark.skipif(not _TORCH, reason="PyTorch not available")


def _make_batch(B, T=4, H=8, W=8):
    import torch
    from training.train_simulator import TrajectoryBatch
    g = torch.randint(0, 16, (B, T, H, W), dtype=torch.long)
    return TrajectoryBatch(
        grid_tokens=g,
        validity_mask=torch.ones(B, T, H, W, dtype=torch.bool),
        action_ids=torch.randint(0, 8, (B, T), dtype=torch.long),
        action_x=torch.full((B, T), -1.0),
        action_y=torch.full((B, T), -1.0),
        coord_valid=torch.zeros(B, T),
        progress_labels=torch.zeros(B, T),
        failure_labels=torch.zeros(B, T),
        avail_labels=torch.ones(B, T, 8),
        game_ids=[f"g{i}" for i in range(B)],
        level_ids=torch.zeros(B, T, dtype=torch.long),
    )


@pytestmark_torch
def test_temporal_split_is_disjoint_and_recent():
    from training.adapter import temporal_split
    batch = _make_batch(10)
    train, val = temporal_split(batch, 0.3)
    assert train.grid_tokens.shape[0] == 7
    assert val.grid_tokens.shape[0] == 3
    # val is the most-recent slice (indices 7,8,9 of the original)
    import torch
    assert torch.equal(val.grid_tokens[0], batch.grid_tokens[7])


@pytestmark_torch
def test_temporal_split_too_small_returns_none():
    from training.adapter import temporal_split
    train, val = temporal_split(_make_batch(1), 0.3)
    assert train is None and val is None


@pytestmark_torch
def test_adapter_per_decision_cap_enforced():
    import torch
    from training.adapter import AdapterManager
    from agent.simulator import ArcSimulator
    cfg = TrainingConfig(feature_dim=32, latent_dim=8, action_dim=16,
                         adapter_max_updates_per_decision=1,
                         adapter_replay_min_samples=4,
                         adapter_validation_fraction=0.3)
    model = ArcSimulator(feature_dim=32, latent_dim=8, action_dim=16)
    mgr = AdapterManager(cfg)
    mgr.begin_decision()
    batch = _make_batch(8)
    device = torch.device("cpu")
    r1 = mgr.maybe_update(model, batch, device)
    r2 = mgr.maybe_update(model, batch, device)  # second within same decision
    assert "cap" in r2.reason  # capped regardless of r1 outcome


@pytestmark_torch
def test_adapter_insufficient_samples_rejected():
    import torch
    from training.adapter import AdapterManager
    from agent.simulator import ArcSimulator
    cfg = TrainingConfig(feature_dim=32, latent_dim=8, action_dim=16,
                         adapter_replay_min_samples=16)
    model = ArcSimulator(feature_dim=32, latent_dim=8, action_dim=16)
    mgr = AdapterManager(cfg)
    mgr.begin_decision()
    r = mgr.maybe_update(model, _make_batch(4), torch.device("cpu"))
    assert not r.accepted
    assert "insufficient" in r.reason


@pytestmark_torch
def test_adapter_rollback_leaves_weights_unchanged():
    """A rejected update must restore the exact prior weights."""
    import copy
    import torch
    from training.adapter import AdapterManager
    from agent.simulator import ArcSimulator
    cfg = TrainingConfig(feature_dim=32, latent_dim=8, action_dim=16,
                         adapter_replay_min_samples=4,
                         adapter_validation_fraction=0.3)
    model = ArcSimulator(feature_dim=32, latent_dim=8, action_dim=16)
    # Force rollback by making any regression intolerable.
    mgr = AdapterManager(cfg, regression_tolerance=1.0)
    mgr.begin_decision()
    before = copy.deepcopy(model.state_dict())
    r = mgr.maybe_update(model, _make_batch(8), torch.device("cpu"))
    if not r.accepted:
        after = model.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), f"weights changed at {k} despite rollback"
        assert mgr.adapter_version == 0

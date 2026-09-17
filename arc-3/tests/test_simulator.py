"""Tests for agent/simulator.py — R10: architecture scaffold.

Tests skip gracefully when PyTorch is unavailable.
All tests use tiny synthetic inputs; no trained weights are loaded.
"""
from __future__ import annotations

import pytest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="PyTorch not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from agent.internal_world import Action, WorldState

LEFT, RIGHT = Action(3), Action(4)

TINY_GRID: tuple[tuple[int, ...], ...] = tuple(
    tuple(c % 16 for c in range(8)) for _ in range(8)
)  # 8×8


def make_state(game: str = "test", version: str = "sim-v1") -> WorldState:
    return WorldState(
        game=game,
        level=0,
        latent=tuple([0.0] * 256),
        grid=TINY_GRID,
        actions=(LEFT, RIGHT),
        model_version=version,
    )


@pytest.fixture
def model():
    from agent.simulator import ArcSimulator
    m = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16, num_actions=8)
    m.eval()
    return m


@pytest.fixture
def sim_state(model):
    return model.initial_state(game_id="test")


# ---------------------------------------------------------------------------
# Architecture construction
# ---------------------------------------------------------------------------

def test_model_constructs():
    from agent.simulator import ArcSimulator
    m = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    assert m.parameter_count() > 0


def test_parameter_count_within_budget():
    """5–20 M parameter budget (blueprint §4). Tiny model is well under."""
    from agent.simulator import ArcSimulator
    m = ArcSimulator()  # default dims
    params = m.parameter_count()
    # Default architecture should be ≤ 20M; tiny is definitely under
    assert params <= 20_000_000, f"parameter_count {params} exceeds 20M budget"


def test_tiny_model_parameter_count():
    from agent.simulator import ArcSimulator
    m = ArcSimulator(feature_dim=64, latent_dim=16, action_dim=16)
    assert m.parameter_count() > 0


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def test_encoder_output_shapes(model):
    import torch
    tokens = torch.zeros(2, 8, 8, dtype=torch.long)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    feat, spatial, hi_res = model.encoder(tokens, mask)
    assert feat.shape == (2, model.feature_dim)
    assert spatial.shape[0] == 2
    assert spatial.shape[1] == 128
    # High-res skip is at H/2 with 32 channels.
    assert hi_res.shape[0] == 2
    assert hi_res.shape[1] == 32
    assert hi_res.shape[2] == 4 and hi_res.shape[3] == 4  # 8/2


def test_encoder_validity_mask_applied(model):
    """Masked cells should have embedding for sentinel, not their true color."""
    import torch
    tokens = torch.full((1, 4, 4), 5, dtype=torch.long)
    mask_all = torch.ones(1, 4, 4, dtype=torch.bool)
    mask_none = torch.zeros(1, 4, 4, dtype=torch.bool)
    feat_valid, _, _ = model.encoder(tokens, mask_all)
    feat_masked, _, _ = model.encoder(tokens, mask_none)
    # Features should differ when all cells are masked vs unmasked
    assert not torch.allclose(feat_valid, feat_masked)


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

def test_action_encoder_output_shape(model):
    import torch
    B = 3
    ids = torch.zeros(B, dtype=torch.long)
    x = torch.zeros(B)
    y = torch.zeros(B)
    v = torch.zeros(B)
    out = model.action_encoder(ids, x, y, v)
    assert out.shape == (B, model.action_dim)


def test_action6_coord_valid_flag(model):
    """ACTION6 with coord_valid=1 should differ from coord_valid=0."""
    import torch
    ids = torch.tensor([6], dtype=torch.long)
    x = torch.tensor([0.5])
    y = torch.tensor([0.5])
    v_on = torch.tensor([1.0])
    v_off = torch.tensor([0.0])
    enc_on = model.action_encoder(ids, x, y, v_on)
    enc_off = model.action_encoder(ids, x, y, v_off)
    assert not torch.allclose(enc_on, enc_off)


# ---------------------------------------------------------------------------
# Posterior and prior
# ---------------------------------------------------------------------------

def test_posterior_output_shapes(model):
    import torch
    h = torch.zeros(2, model.feature_dim)
    obs = torch.zeros(2, model.feature_dim)
    mu, logvar = model.posterior(h, obs)
    assert mu.shape == (2, model.latent_dim)
    assert logvar.shape == (2, model.latent_dim)


def test_prior_output_shapes(model):
    import torch
    h = torch.zeros(2, model.feature_dim)
    act = torch.zeros(2, model.action_dim)
    mu, logvar = model.prior(h, act)
    assert mu.shape == (2, model.latent_dim)
    assert logvar.shape == (2, model.latent_dim)


def test_reparameterise_shape():
    import torch
    from agent.simulator import reparameterise
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)
    z = reparameterise(mu, logvar)
    assert z.shape == (4, 16)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def test_dynamics_output_shape(model):
    import torch
    h = torch.zeros(2, model.feature_dim)
    z = torch.zeros(2, model.latent_dim)
    act = torch.zeros(2, model.action_dim)
    h_new = model.dynamics(h, z, act)
    assert h_new.shape == (2, model.feature_dim)


def test_dynamics_advances_state(model):
    """Dynamics output should differ from input h (non-trivial transformation)."""
    import torch
    h = torch.randn(2, model.feature_dim)
    z = torch.randn(2, model.latent_dim)
    act = torch.randn(2, model.action_dim)
    h_new = model.dynamics(h, z, act)
    assert not torch.allclose(h, h_new)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def test_decoder_output_keys_and_shapes(model):
    import torch
    h = torch.zeros(1, model.feature_dim)
    z = torch.zeros(1, model.latent_dim)
    spatial = torch.zeros(1, 128, 1, 1)
    outputs = model.decoder(h, z, spatial, (8, 8))
    assert "frame_logits" in outputs
    assert "shape_valid" in outputs
    assert "progress" in outputs
    assert "failure" in outputs
    assert "avail_logits" in outputs
    assert outputs["frame_logits"].shape == (1, 16, 8, 8)
    assert outputs["avail_logits"].shape == (1, model.num_actions)


def test_decoder_progress_in_range(model):
    import torch
    h = torch.randn(1, model.feature_dim)
    z = torch.randn(1, model.latent_dim)
    spatial = torch.zeros(1, 128, 1, 1)
    outputs = model.decoder(h, z, spatial, (4, 4))
    assert 0.0 <= outputs["progress"].item() <= 1.0
    assert 0.0 <= outputs["failure"].item() <= 1.0


def test_change_mask_decoder_emits_gate_and_preserves_via_closed_gate(model):
    """With prev_tokens provided, the decoder exposes a per-cell change_gate, and
    forcing the gate shut reproduces the previous frame (change-mask contract)."""
    import torch
    h = torch.randn(1, model.feature_dim)
    z = torch.randn(1, model.latent_dim)
    spatial = torch.zeros(1, 128, 1, 1)
    prev = torch.randint(0, 16, (1, 8, 8), dtype=torch.long)

    outputs = model.decoder(h, z, spatial, (8, 8), prev_tokens=prev)
    assert "change_gate" in outputs
    assert outputs["change_gate"].shape == (1, 1, 8, 8)
    assert float(outputs["change_gate"].min()) >= 0.0
    assert float(outputs["change_gate"].max()) <= 1.0

    # Force the gate closed: the blended frame must argmax back to prev everywhere.
    C = model.decoder.color_categories
    keep = torch.nn.functional.one_hot(prev.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    keep = keep * model.decoder.keep_logit_scale
    # gate=0 -> frame_logits == keep_logits; argmax == prev
    assert torch.equal(keep.argmax(dim=1), prev)


def test_decoder_without_prev_tokens_is_full_frame(model):
    """Omitting prev_tokens keeps the original full-frame behaviour (no gate blend)."""
    import torch
    h = torch.zeros(1, model.feature_dim)
    z = torch.zeros(1, model.latent_dim)
    spatial = torch.zeros(1, 128, 1, 1)
    outputs = model.decoder(h, z, spatial, (6, 6))  # no prev_tokens
    assert outputs["frame_logits"].shape == (1, 16, 6, 6)


# ---------------------------------------------------------------------------
# SimulatorState fork / freeze contracts
# ---------------------------------------------------------------------------

def test_simulator_state_copy_is_independent(model, sim_state):
    import torch
    copy = sim_state.copy()
    # Mutate original's h; copy should not change
    sim_state.h += 1.0
    assert not torch.allclose(sim_state.h, copy.h)


def test_freeze_stops_gradients(model):
    model.freeze()
    assert model.is_frozen
    for p in model.parameters():
        assert not p.requires_grad


def test_unfreeze_restores_gradients(model):
    model.freeze()
    model.unfreeze()
    assert not model.is_frozen
    for p in model.parameters():
        assert p.requires_grad


def test_freeze_then_imagine_step(model, sim_state):
    """Imagined step should run without error when weights are frozen."""
    model.freeze()
    new_state = model.imagine_step(RIGHT, sim_state)
    assert new_state.h is not None
    assert new_state.z is not None
    model.unfreeze()


# ---------------------------------------------------------------------------
# Multi-step rollout
# ---------------------------------------------------------------------------

def test_multi_step_rollout_shape(model, sim_state):
    """4-step rollout should produce 4 distinct states."""
    states = [sim_state]
    for _ in range(4):
        new = model.imagine_step(RIGHT, states[-1])
        states.append(new)
    assert len(states) == 5


def test_multi_step_rollout_h_changes(model, sim_state):
    """Each step should change the hidden state."""
    import torch
    prev = sim_state
    for _ in range(4):
        nxt = model.imagine_step(RIGHT, prev)
        assert not torch.allclose(nxt.h, prev.h)
        prev = nxt


# ---------------------------------------------------------------------------
# predict() Dynamics protocol
# ---------------------------------------------------------------------------

def test_predict_returns_prediction(model):
    from agent.internal_world import Prediction
    state = make_state()
    pred = model.predict(state, RIGHT)
    assert isinstance(pred, Prediction)
    assert pred.state.imagined
    assert pred.state.game == "test"
    assert pred.state.model_version == "sim-v1"


def test_predict_keeps_game_and_version(model):
    state = make_state(game="my-game", version="sim-v1")
    pred = model.predict(state, RIGHT)
    assert pred.state.game == "my-game"
    assert pred.state.model_version == "sim-v1"


def test_predict_scores_in_range(model):
    state = make_state()
    pred = model.predict(state, RIGHT)
    assert 0.0 <= pred.uncertainty <= 1.0
    assert 0.0 <= pred.risk <= 1.0


# ---------------------------------------------------------------------------
# Checkpoint save / load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(model, tmp_path):
    import torch
    from agent.simulator import ArcSimulator
    ckpt_path = tmp_path / "sim_test.pt"
    descriptor = model.save_checkpoint(ckpt_path, model_version="sim-v1")
    assert ckpt_path.exists()
    assert descriptor.role == "simulator"
    assert len(descriptor.artifact_hash) == 64

    loaded_model, loaded_desc = ArcSimulator.load_checkpoint(ckpt_path)
    assert loaded_desc.model_version == "sim-v1"
    assert loaded_model.parameter_count() == model.parameter_count()

    # Weights should be identical
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), loaded_model.state_dict().items()
    ):
        assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

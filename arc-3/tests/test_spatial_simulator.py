"""Tests for the full-resolution spatial-dynamics simulator.

Locks in the interface and the key property: the recurrent state is a
FULL-RESOLUTION feature map (not a pooled vector), which is what lets the model
learn where an object moves TO. (The learning result itself is measured by
scripts/step_spatial_controlled_world.py; these tests guard the wiring.)
"""
import pytest

torch = pytest.importorskip("torch")

from agent.spatial_simulator import SpatialArcSimulator, SpatialState
from agent.internal_world import Action


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SpatialArcSimulator()


def test_state_is_full_resolution(model):
    st = model.initial_state(game_id="g", device=torch.device("cpu"), hw=(12, 12))
    # Hidden and latent maps are at the FULL grid resolution.
    assert st.h.shape == (1, model.h_ch, 12, 12)
    assert st.z.shape == (1, model.z_ch, 12, 12)


def test_encode_and_posterior_refresh_scene(model):
    tok = torch.randint(0, 16, (1, 10, 10), dtype=torch.long)
    mask = torch.ones(1, 10, 10, dtype=torch.bool)
    obs = model.encoder(tok, mask)
    assert obs.shape == (1, model.h_ch, 10, 10)  # full-res, no downsample
    st = model.initial_state(game_id="g", hw=(10, 10))
    st2 = model.update_posterior(obs, st)
    # posterior refreshes h to the observed scene map
    assert torch.equal(st2.h, obs)


def test_imagine_step_preserves_resolution_and_changes_state(model):
    st = model.initial_state(game_id="g", hw=(8, 8))
    # seed a non-zero hidden map so dynamics have something to transform
    st = SpatialState(h=torch.randn_like(st.h), z=torch.randn_like(st.z),
                      game_id="g", model_version=st.model_version)
    st2 = model.imagine_step(Action(id=1), st)
    assert st2.h.shape == st.h.shape
    assert not torch.equal(st2.h, st.h)


def test_decode_change_mask_shapes(model):
    st = model.initial_state(game_id="g", hw=(8, 8))
    st = SpatialState(h=torch.randn_like(st.h), z=torch.randn_like(st.z),
                      game_id="g", model_version=st.model_version)
    prev = torch.randint(0, 16, (1, 8, 8), dtype=torch.long)
    out = model._heads(st, (8, 8), prev_tokens=prev)
    assert out["frame_logits"].shape == (1, 16, 8, 8)
    assert out["change_gate"].shape == (1, 1, 8, 8)
    assert 0.0 <= float(out["change_gate"].min()) and float(out["change_gate"].max()) <= 1.0


def test_opposing_move_actions_differ(model):
    """Different move actions must drive the dynamics to different states —
    the property a pooled-vector core could not express for position."""
    st = model.initial_state(game_id="g", hw=(8, 8))
    st = SpatialState(h=torch.randn_like(st.h), z=torch.randn_like(st.z),
                      game_id="g", model_version=st.model_version)
    left = model.imagine_step(Action(id=3), st)
    right = model.imagine_step(Action(id=4), st)
    assert not torch.allclose(left.h, right.h)


def test_param_count_within_small_budget(model):
    # Full-res ConvGRU is parameter-efficient; keep it well under the vector model.
    assert model.parameter_count() < 1_000_000

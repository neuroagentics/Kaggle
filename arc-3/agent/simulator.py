"""Simulator architecture scaffold — R10 implementation.

Defines the full neural architecture for the ARC-3 learned world model:
  - FrameEncoder: convolutional encoder (32/64/128 ch, stride-2) → 256-dim feature
  - PosteriorNet: encodes (h, obs_feature) → (mu_z, logvar_z) diagonal Gaussian
  - PriorNet:     encodes (h, action_encoding) → (mu_z, logvar_z)
  - DynamicsNet:  GRU recurrent core (h=256) conditioned on (z, action_encoding)
  - SimulatorDecoder: decodes (h, z) → frame_logits + outcome_head + avail_head
  - ArcSimulator: assembles all heads; implements Dynamics + HypothesisConditionedDynamics

No trained weights are included. This module defines architecture and forward-pass
contracts only. Call ArcSimulator.load_checkpoint() to attach real weights.

Blueprint §4 targets:
  - Grid: up to 64×64, values 0-15, color embeddings dim=16
  - Encoder output: 256-dim feature (spatial features retained for decoder)
  - h (GRU hidden): width 256
  - z (stochastic latent): width 64, diagonal Gaussian
  - 5–20 M parameter initial budget
  - Device: single GPU (16–24 GB); CPU for tests only

Imports PyTorch. Tests skip gracefully if torch is unavailable (CPU-only CI).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from agent.internal_world import Action, Dynamics, InternalGame, Prediction, WorldState
from agent.model import (
    CheckpointDescriptor,
    SimulatorCondition,
    HypothesisConditionedDynamics,
    SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Architecture hyper-parameters (initial defaults, all tunable)
# ---------------------------------------------------------------------------

COLOR_CATEGORIES = 16      # ARC color values 0..15
COLOR_EMBED_DIM = 16       # §4: 16-dimensional color embeddings
FEATURE_DIM = 256          # encoder output and GRU hidden width
LATENT_DIM = 64            # stochastic latent z width
ACTION_DIM = 32            # action encoding width
NUM_ACTIONS = 8            # ARC-3 action IDs 0..7


# ---------------------------------------------------------------------------
# Require torch for architecture classes
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the simulator architecture. "
            "Install it with the project's training dependencies."
        )


# ---------------------------------------------------------------------------
# Spatial feature shape helper
# ---------------------------------------------------------------------------

def _conv_output_size(input_size: int, n_stride2_layers: int) -> int:
    """Size after n stride-2 conv layers (floor division)."""
    s = input_size
    for _ in range(n_stride2_layers):
        s = (s + 1) // 2
    return s


# ---------------------------------------------------------------------------
# Frame encoder
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module if _TORCH_AVAILABLE else object):
    """Convolutional encoder: ARC grid → 256-dim spatial feature map + pooled vector.

    Architecture (blueprint §4):
      - Color embedding: vocab=16+1 (16 colors + validity mask sentinel), dim=16
      - Conv stage 1: 16 → 32 channels, 3×3, stride 2, ReLU
      - Conv stage 2: 32 → 64 channels, 3×3, stride 2, ReLU
      - Conv stage 3: 64 → 128 channels, 3×3, stride 2, ReLU
      - Global average pool → 128-dim
      - Linear 128 → 256

    Spatial features from stage 3 are retained for the decoder skip connection.
    A high-resolution stage-1 map (H/2) is ALSO retained: the pooled global vector
    destroys object position, and the H/8 skip is too coarse to localize single-cell
    changes (measured: coordinate-free spatial mechanics learned poorly). The H/2
    skip lets the decoder's change-mask heads localize precisely.
    Inputs: (B, H, W) int64 grid tokens, validity_mask (B, H, W) bool.
    Outputs: feature (B, 256), spatial (B, 128, H/8, W/8), hi_res (B, 32, H/2, W/2).
    """

    def __init__(
        self,
        color_embed_dim: int = COLOR_EMBED_DIM,
        feature_dim: int = FEATURE_DIM,
    ):
        _require_torch()
        super().__init__()
        self.color_embed_dim = color_embed_dim
        self.feature_dim = feature_dim

        # +1 for the padding/validity-mask sentinel value
        self.color_embed = nn.Embedding(COLOR_CATEGORIES + 1, color_embed_dim)

        self.conv1 = nn.Conv2d(color_embed_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(128, feature_dim)

    def forward(
        self,
        grid_tokens: "torch.Tensor",       # (B, H, W) int64
        validity_mask: "torch.Tensor",     # (B, H, W) bool; True = valid
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return (feature: B×256, spatial: B×128×H/8×W/8, hi_res: B×32×H/2×W/2)."""
        B, H, W = grid_tokens.shape

        # Zero out invalid (padded) positions before embedding
        masked = grid_tokens.clone()
        masked[~validity_mask] = COLOR_CATEGORIES  # sentinel index

        emb = self.color_embed(masked)           # (B, H, W, C)
        emb = emb.permute(0, 3, 1, 2).float()   # (B, C, H, W)

        hi_res = F.relu(self.conv1(emb))         # (B, 32, H/2, W/2) — high-res skip
        x = F.relu(self.conv2(hi_res))           # (B, 64, H/4, W/4)
        spatial = F.relu(self.conv3(x))          # (B, 128, H/8, W/8)

        pooled = self.pool(spatial).flatten(1)   # (B, 128)
        feature = self.project(pooled)           # (B, 256)
        return feature, spatial, hi_res


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module if _TORCH_AVAILABLE else object):
    """Encodes an Action (id, x, y, coord_valid) into a fixed-dim vector.

    blueprint §4: ID plus normalized coordinates and a coordinate-valid flag.
    Distinguishes RESET (action_id=7) and unknown/unavailable effects.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS, action_dim: int = ACTION_DIM):
        _require_torch()
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        # ID embedding + (x_norm, y_norm, coord_valid) → action_dim
        self.id_embed = nn.Embedding(num_actions, action_dim - 3)
        self.out = nn.Linear(action_dim, action_dim)

    def forward(
        self,
        action_ids: "torch.Tensor",    # (B,) int64
        action_x: "torch.Tensor",      # (B,) float32, normalised 0..1; -1 if n/a
        action_y: "torch.Tensor",      # (B,) float32, normalised 0..1; -1 if n/a
        coord_valid: "torch.Tensor",   # (B,) float32  0 or 1
    ) -> "torch.Tensor":               # (B, action_dim)
        id_emb = self.id_embed(action_ids)                            # (B, D-3)
        extra = torch.stack([action_x, action_y, coord_valid], dim=1) # (B, 3)
        combined = torch.cat([id_emb, extra], dim=1)                  # (B, D)
        return F.relu(self.out(combined))


# ---------------------------------------------------------------------------
# Posterior and prior networks
# ---------------------------------------------------------------------------

class PosteriorNet(nn.Module if _TORCH_AVAILABLE else object):
    """q(z_t | h_{t-1}, obs_feature_t) → (mu, logvar) of z_t.

    Conditions on the recurrent state and the new observation feature.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        _require_torch()
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + feature_dim, feature_dim)
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

    def forward(
        self,
        h: "torch.Tensor",            # (B, feature_dim)
        obs_feature: "torch.Tensor",  # (B, feature_dim)
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        x = F.relu(self.fc1(torch.cat([h, obs_feature], dim=1)))
        return self.mu(x), self.logvar(x)


class PriorNet(nn.Module if _TORCH_AVAILABLE else object):
    """p(z_t | h_{t-1}, a_{t-1}) → (mu, logvar) of z_t.

    Used during imagined rollouts when no real observation is available.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
    ):
        _require_torch()
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, feature_dim)
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.logvar = nn.Linear(feature_dim, latent_dim)

    def forward(
        self,
        h: "torch.Tensor",           # (B, feature_dim)
        action_enc: "torch.Tensor",  # (B, action_dim)
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        x = F.relu(self.fc1(torch.cat([h, action_enc], dim=1)))
        return self.mu(x), self.logvar(x)


def reparameterise(
    mu: "torch.Tensor", logvar: "torch.Tensor"
) -> "torch.Tensor":
    """Sample z = mu + eps * std (reparameterisation trick)."""
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# Dynamics (GRU recurrent core)
# ---------------------------------------------------------------------------

class DynamicsNet(nn.Module if _TORCH_AVAILABLE else object):
    """GRU recurrent dynamics: (h_{t-1}, z_t, a_{t-1}) → h_t.

    blueprint §4: recurrent hidden state h of width 256.
    Input to GRU: concatenate z and action_encoding, then project to feature_dim.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
    ):
        _require_torch()
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + action_dim, feature_dim)
        self.gru_cell = nn.GRUCell(feature_dim, feature_dim)

    def forward(
        self,
        h: "torch.Tensor",           # (B, feature_dim)
        z: "torch.Tensor",           # (B, latent_dim)
        action_enc: "torch.Tensor",  # (B, action_dim)
    ) -> "torch.Tensor":             # (B, feature_dim) — new h
        inp = F.relu(self.input_proj(torch.cat([z, action_enc], dim=1)))
        return self.gru_cell(inp, h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SimulatorDecoder(nn.Module if _TORCH_AVAILABLE else object):
    """Decodes (h, z, spatial_skip) → frame logits + outcome head + avail head.

    blueprint §4 output heads:
      - frame_logits: (B, 16, H, W) categorical next-frame predictions
      - shape_valid:  (B, 1)         validity/shape prediction
      - progress:     (B, 1)         level progress prediction in [0,1]
      - failure:      (B, 1)         failure/game-over prediction in [0,1]
      - avail_logits: (B, NUM_ACTIONS) candidate legal-action probabilities

    Decoder upsamples via transposed convolutions mirroring the encoder stages.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
        color_categories: int = COLOR_CATEGORIES,
        num_actions: int = NUM_ACTIONS,
    ):
        _require_torch()
        super().__init__()
        self.feature_dim = feature_dim

        # Fuse h and z → spatial seed
        self.fuse = nn.Linear(feature_dim + latent_dim, 128 * 4 * 4)

        self.color_categories = color_categories

        # Upsample back through encoder stages (mirror)
        self.deconv3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Trunk now emits a shared feature map; two heads read from it:
        #   - update_head: the new-color logits for cells that DO change
        #   - gate_head:   a per-cell change probability (sigmoid)
        # This lets the decoder express "keep this cell, change that one" instead of
        # regenerating the whole frame from a pooled latent. See ARC3_STEP1 report:
        # the full-redraw head could not learn action-conditioned localized change.
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        # Fuse the encoder's HIGH-RESOLUTION (H/2) skip so the change-mask heads can
        # localize single-cell changes. The pooled global state destroys position;
        # the H/8 skip is too coarse. hi_res (32ch) is resized to the trunk feature
        # resolution and concatenated. When hi_res is absent, a zero map is used so
        # behaviour is unchanged for callers that don't supply it.
        self.hi_res_fuse = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
        self.update_head = nn.Conv2d(32, color_categories, kernel_size=1)
        self.gate_head = nn.Conv2d(32, 1, kernel_size=1)
        # Scale applied to the previous-frame one-hot when the gate says "no change",
        # so that a closed gate yields logits that argmax back to the previous color.
        self.keep_logit_scale = 12.0

        # Outcome heads (operate on pooled h)
        self.shape_valid_head = nn.Linear(feature_dim, 1)
        self.progress_head = nn.Linear(feature_dim, 1)
        self.failure_head = nn.Linear(feature_dim, 1)
        self.avail_head = nn.Linear(feature_dim, num_actions)

    def forward(
        self,
        h: "torch.Tensor",                # (B, feature_dim)
        z: "torch.Tensor",                # (B, latent_dim)
        spatial_skip: "torch.Tensor",     # (B, 128, H/8, W/8) from encoder
        target_hw: tuple[int, int],       # (H, W) to crop/pad to
        prev_tokens: "torch.Tensor | None" = None,  # (B, H, W) int64 previous frame
        hi_res_skip: "torch.Tensor | None" = None,  # (B, 32, H/2, W/2) high-res skip
    ) -> dict[str, "torch.Tensor"]:
        """Decode next-frame logits + outcome heads.

        Change-mask decoding (when prev_tokens is given):
          frame_logits = gate * update_logits + (1 - gate) * keep_logits
        where `gate` is a learned per-cell change probability and `keep_logits`
        are a scaled one-hot of the PREVIOUS frame. A closed gate (no change)
        reproduces the previous cell exactly; an open gate lets update_logits
        choose the new color. "No change" is the default, so the model must
        actively decide WHERE and TO WHAT to change — the pathway the full-redraw
        decoder lacked. Full-frame change stays possible (gate can open anywhere);
        nothing assumes a stationary background.

        When prev_tokens is None, returns pure update_logits (full-frame redraw),
        preserving the original behaviour for callers/tests that don't pass it.
        """
        B = h.shape[0]
        H, W = target_hw

        # Build spatial seed from fused h+z
        seed = F.relu(self.fuse(torch.cat([h, z], dim=1)))  # (B, 128*4*4)
        seed = seed.view(B, 128, 4, 4)

        # Concatenate skip connection from encoder at same resolution
        Hs, Ws = spatial_skip.shape[2], spatial_skip.shape[3]
        seed_up = F.interpolate(seed, size=(Hs, Ws), mode="nearest")
        x = torch.cat([seed_up, spatial_skip], dim=1)       # (B, 256, Hs, Ws)

        x = F.relu(self.deconv3(x))                          # (B, 64, 2Hs, 2Ws)
        x = F.relu(self.deconv2(x))                          # (B, 32, 4Hs, 4Ws)
        feat = F.relu(self.deconv1(x))                       # (B, 32, 8Hs, 8Ws)

        # Fuse the high-resolution skip so the heads can localize single-cell
        # changes. Resize the trunk feature to the hi-res resolution and concat.
        if hi_res_skip is not None:
            Hh, Wh = hi_res_skip.shape[2], hi_res_skip.shape[3]
            feat_up = F.interpolate(feat, size=(Hh, Wh), mode="nearest")
            feat = F.relu(self.hi_res_fuse(torch.cat([feat_up, hi_res_skip], dim=1)))
        else:
            # No hi-res skip: keep behaviour stable by fusing with a zero map.
            zeros = torch.zeros_like(feat)
            feat = F.relu(self.hi_res_fuse(torch.cat([feat, zeros], dim=1)))

        update_logits = self.update_head(feat)               # (B, 16, *, *)
        gate_logit = self.gate_head(feat)                    # (B, 1, *, *)

        # Crop or pad to match exact target H×W
        def _resize_logits(t):
            if t.shape[2] != H or t.shape[3] != W:
                return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            return t

        update_logits = _resize_logits(update_logits)
        gate_logit = _resize_logits(gate_logit)
        gate = torch.sigmoid(gate_logit)                     # (B, 1, H, W) in [0,1]

        if prev_tokens is not None:
            C = self.color_categories
            prev_clamped = prev_tokens.clamp(0, C - 1).long()          # (B, H, W)
            keep_onehot = F.one_hot(prev_clamped, C).permute(0, 3, 1, 2).float()
            keep_logits = keep_onehot * self.keep_logit_scale          # (B, C, H, W)
            frame_logits = gate * update_logits + (1.0 - gate) * keep_logits
        else:
            frame_logits = update_logits

        return {
            "frame_logits": frame_logits,                                      # (B, 16, H, W)
            "change_gate": gate,                                               # (B, 1, H, W)
            "shape_valid": torch.sigmoid(self.shape_valid_head(h)),            # (B, 1)
            "progress": torch.sigmoid(self.progress_head(h)),                  # (B, 1)
            "failure": torch.sigmoid(self.failure_head(h)),                    # (B, 1)
            "avail_logits": self.avail_head(h),                                # (B, NUM_ACTIONS)
        }


# ---------------------------------------------------------------------------
# Recurrent state bundle
# ---------------------------------------------------------------------------

@dataclass
class SimulatorState:
    """Mutable recurrent state carried across steps.

    Forked by copy() for imagined branches; weights are never copied.
    """
    h: "torch.Tensor"   # (1, feature_dim) — batch size 1 for step-by-step
    z: "torch.Tensor"   # (1, latent_dim)
    game_id: str
    model_version: str

    def copy(self) -> "SimulatorState":
        return SimulatorState(
            h=self.h.detach().clone(),
            z=self.z.detach().clone(),
            game_id=self.game_id,
            model_version=self.model_version,
        )


# ---------------------------------------------------------------------------
# ArcSimulator: the full model
# ---------------------------------------------------------------------------

class ArcSimulator(nn.Module if _TORCH_AVAILABLE else object):
    """Full simulator implementing Dynamics and HypothesisConditionedDynamics.

    Assembles encoder, posterior/prior, dynamics, and decoder. Exposes:
      - encode_observation(): produce a feature from a real frame
      - update_posterior():   update z from a new observation
      - imagine_step():       advance h,z from prior given an action
      - decode():             produce frame logits and outcome heads
      - predict():            Dynamics protocol — unconditioned rollout
      - predict_conditioned(): HypothesisConditionedDynamics protocol

    Weight freezing contract (blueprint §5):
      - freeze() is called at plan start; no gradient flows during imagined steps.
      - unfreeze() restores gradient tracking for adapter updates between plans.
    """

    VERSION = "sim-v1"

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        color_embed_dim: int = COLOR_EMBED_DIM,
    ):
        _require_torch()
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self._frozen = False

        self.encoder = FrameEncoder(color_embed_dim, feature_dim)
        self.action_encoder = ActionEncoder(num_actions, action_dim)
        self.posterior = PosteriorNet(feature_dim, latent_dim)
        self.prior = PriorNet(feature_dim, latent_dim, action_dim)
        self.dynamics = DynamicsNet(feature_dim, latent_dim, action_dim)
        self.decoder = SimulatorDecoder(feature_dim, latent_dim, COLOR_CATEGORIES, num_actions)

    # ------------------------------------------------------------------
    # Utility: parameter count
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Weight freeze / unfreeze (blueprint §5: no gradient during planning)
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze weights for imagined rollouts. Must be called at plan start."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def unfreeze(self) -> None:
        """Restore gradient tracking after planning completes."""
        for p in self.parameters():
            p.requires_grad_(True)
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_to_tensors(
        grid: tuple[tuple[int, ...], ...], device: "torch.device"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Convert a WorldState grid to (tokens, validity_mask) on device."""
        H = len(grid)
        W = len(grid[0]) if H else 0
        tokens = torch.zeros(1, H, W, dtype=torch.long, device=device)
        mask = torch.ones(1, H, W, dtype=torch.bool, device=device)
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                tokens[0, r, c] = val
        return tokens, mask

    @staticmethod
    def _action_to_tensors(
        action: Action, device: "torch.device"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Convert an Action to (id, x_norm, y_norm, coord_valid) tensors."""
        coord_valid = 1.0 if action.x is not None else 0.0
        x_norm = action.x / 63.0 if action.x is not None else 0.0
        y_norm = action.y / 63.0 if action.y is not None else 0.0
        return (
            torch.tensor([action.id], dtype=torch.long, device=device),
            torch.tensor([x_norm], dtype=torch.float32, device=device),
            torch.tensor([y_norm], dtype=torch.float32, device=device),
            torch.tensor([coord_valid], dtype=torch.float32, device=device),
        )

    # ------------------------------------------------------------------
    # Step interfaces
    # ------------------------------------------------------------------

    def encode_observation(
        self,
        grid: tuple[tuple[int, ...], ...],
        sim_state: SimulatorState,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Encode a real grid observation. Returns (feature, spatial)."""
        device = sim_state.h.device
        tokens, mask = self._grid_to_tensors(grid, device)
        return self.encoder(tokens, mask)

    def update_posterior(
        self,
        obs_feature: "torch.Tensor",
        sim_state: SimulatorState,
    ) -> SimulatorState:
        """Update z from new observation (posterior update). Returns updated state."""
        mu, logvar = self.posterior(sim_state.h, obs_feature)
        z = reparameterise(mu, logvar) if self.training else mu
        return SimulatorState(
            h=sim_state.h.detach(), z=z.detach(),
            game_id=sim_state.game_id,
            model_version=sim_state.model_version,
        )

    def imagine_step(
        self,
        action: Action,
        sim_state: SimulatorState,
    ) -> SimulatorState:
        """Advance h and z from prior given action (imagined step). Returns new state."""
        device = sim_state.h.device
        aid, ax, ay, av = self._action_to_tensors(action, device)
        action_enc = self.action_encoder(aid, ax, ay, av)

        # Consume the current posterior/prior BEFORE predicting the next latent.
        # Otherwise a real observation's z is discarded before it can affect h.
        with torch.no_grad():
            new_h = self.dynamics(sim_state.h, sim_state.z, action_enc)
            mu_p, logvar_p = self.prior(new_h, action_enc)
            z = reparameterise(mu_p, logvar_p) if self.training else mu_p
        return SimulatorState(
            h=new_h, z=z,
            game_id=sim_state.game_id,
            model_version=sim_state.model_version,
        )

    def decode(
        self,
        sim_state: SimulatorState,
        spatial_skip: "torch.Tensor",
        target_hw: tuple[int, int],
        prev_tokens: "torch.Tensor | None" = None,
        hi_res_skip: "torch.Tensor | None" = None,
    ) -> dict[str, "torch.Tensor"]:
        """Decode current state to frame logits and outcome heads.

        prev_tokens (the frame being predicted FROM) enables change-mask decoding:
        the decoder preserves unchanged cells and only edits where it predicts a
        change. hi_res_skip (encoder H/2 map) lets the heads localize single-cell
        changes. Both are optional for back-compat.
        """
        return self.decoder(
            sim_state.h, sim_state.z, spatial_skip, target_hw,
            prev_tokens, hi_res_skip,
        )

    # ------------------------------------------------------------------
    # Dynamics protocol implementation
    # ------------------------------------------------------------------

    def _make_world_state(
        self,
        outputs: dict[str, "torch.Tensor"],
        base_state: WorldState,
        imagined: bool = True,
    ) -> WorldState:
        """Convert decoder outputs to a WorldState grid."""
        frame_logits = outputs["frame_logits"]       # (1, 16, H, W)
        pred_grid_idx = frame_logits[0].argmax(dim=0) # (H, W)
        # One device transfer. Per-cell .item() synchronizes CUDA thousands of
        # times per prediction and can consume the entire game budget.
        grid = tuple(tuple(row) for row in pred_grid_idx.detach().cpu().tolist())
        progress = float(outputs["progress"][0, 0].item())
        failure = float(outputs["failure"][0, 0].item())
        terminal = failure > 0.5 or progress >= 0.99

        return replace(
            base_state,
            grid=grid,
            imagined=imagined,
            terminal=terminal,
        )

    def _make_prediction(
        self,
        outputs: dict[str, "torch.Tensor"],
        next_state: WorldState,
    ) -> Prediction:
        """Build a Prediction from decoder outputs.

        This is predictive entropy, NOT calibrated prediction reliability or
        epistemic uncertainty. It is measured as the mean per-cell
        entropy of the decoded next-frame distribution, normalised to [0,1] by
        log(num_colors). Low entropy = the model is confident about the next
        frame (peaked per-cell distributions); confidence may still be wrong.
        This replaces the earlier bug where uncertainty came from a softmax over
        action-AVAILABILITY logits (which has nothing to do with how reliable the
        frame prediction is, and pinned uncertainty near 0.875 regardless).
        """
        import math as _math
        frame_logits = outputs["frame_logits"][0]              # (C, H, W)
        probs = torch.softmax(frame_logits, dim=0)             # per-cell dist over colors
        # Per-cell entropy, averaged over cells.
        cell_entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=0)  # (H, W)
        mean_entropy = float(cell_entropy.mean().item())
        max_entropy = _math.log(frame_logits.shape[0])         # log(num_colors)
        uncertainty = mean_entropy / max_entropy if max_entropy > 0 else 0.0

        risk = float(outputs["failure"][0, 0].item())
        utility = float(outputs["progress"][0, 0].item())
        return Prediction(
            state=next_state,
            utility=utility,
            uncertainty=max(0.0, min(1.0, uncertainty)),
            risk=max(0.0, min(1.0, risk)),
        )

    # ------------------------------------------------------------------
    # Recurrent-state <-> WorldState.latent packing
    # ------------------------------------------------------------------
    # The planner threads WorldState through fork/step. To keep the REAL recurrent
    # state (h AND z) alive across imagined steps — instead of rebuilding h from a
    # scalar and zeroing z every call — we pack the full (h, z) into the latent
    # tuple: [feature_dim values of h] + [latent_dim values of z]. This is what
    # lets an observation-corrected posterior actually propagate into planning.

    def pack_latent(self, sim_state: SimulatorState) -> tuple[float, ...]:
        h = sim_state.h[0].detach().cpu().tolist()
        z = sim_state.z[0].detach().cpu().tolist()
        return tuple(h) + tuple(z)

    def _unpack_latent(self, latent: tuple[float, ...], device) -> SimulatorState | None:
        """Recover (h, z) from a packed latent. Returns None if not packed
        (e.g. a fixture latent of the wrong width) so the caller can seed fresh."""
        expected = self.feature_dim + self.latent_dim
        if len(latent) != expected:
            return None
        h = torch.tensor([latent[:self.feature_dim]], dtype=torch.float32, device=device)
        z = torch.tensor([latent[self.feature_dim:]], dtype=torch.float32, device=device)
        return SimulatorState(h=h, z=z, game_id="", model_version="")

    def predict(self, state: WorldState, action: Action) -> Prediction:
        """Dynamics.predict() — advance one imagined step from WorldState.

        Uses the FULL recurrent state (h, z) packed in state.latent when present,
        preserving posterior-corrected state across imagined steps. Falls back to
        a zero-seeded state only when the latent is not a packed recurrent state
        (e.g. fixtures), rather than silently discarding z every call.
        """
        _require_torch()
        device = next(self.parameters()).device
        tokens, mask = self._grid_to_tensors(state.grid, device)
        with torch.no_grad():
            _, spatial, hi_res = self.encoder(tokens, mask)

        recovered = self._unpack_latent(state.latent, device)
        if recovered is not None:
            sim_state = SimulatorState(h=recovered.h, z=recovered.z,
                                       game_id=state.game, model_version=state.model_version)
        else:
            # Not a packed recurrent state — seed a fresh zeroed state.
            sim_state = SimulatorState(
                h=torch.zeros(1, self.feature_dim, device=device),
                z=torch.zeros(1, self.latent_dim, device=device),
                game_id=state.game, model_version=state.model_version,
            )

        new_sim = self.imagine_step(action, sim_state)
        with torch.no_grad():
            outputs = self.decode(
                new_sim, spatial, (len(state.grid), len(state.grid[0])),
                prev_tokens=tokens, hi_res_skip=hi_res,
            )
        return self._finish_prediction(outputs, new_sim, state)

    def _finish_prediction(self, outputs, new_sim, state) -> Prediction:
        """Build a Prediction from decoder outputs, carrying the full (h,z)."""
        next_state = self._make_world_state(outputs, state)
        next_state = replace(next_state, latent=self.pack_latent(new_sim))
        return self._make_prediction(outputs, next_state)

    # No checkpoint in this architecture has learned a hypothesis input. Keep
    # the interface, but never fabricate support by shifting decoded pixels.
    supports_conditioning = False
    transition_version = "posterior-action-v2"

    def predict_conditioned(
        self,
        state: WorldState,
        action: Action,
        condition: SimulatorCondition,
    ) -> Prediction:
        """Reject untrained conditioning rather than inventing a mechanic."""
        if condition.mechanic_hypotheses:
            raise NotImplementedError("Checkpoint does not support learned mechanic conditioning")
        pred = self.predict(state, action)
        if pred.state.game != condition.game_id:
            raise ValueError(
                f"Prediction game {pred.state.game!r} != condition game_id {condition.game_id!r}"
            )
        if pred.state.model_version != condition.model_version:
            raise ValueError(
                f"Prediction model_version {pred.state.model_version!r} != "
                f"condition model_version {condition.model_version!r}"
            )
        return pred

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, *, model_version: str) -> CheckpointDescriptor:
        """Save weights to path and return a CheckpointDescriptor."""
        import torch
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "transition_version": self.transition_version,
                "model_version": model_version,
                "architecture": {
                    "feature_dim": self.feature_dim,
                    "latent_dim": self.latent_dim,
                    "action_dim": self.action_dim,
                    "num_actions": self.num_actions,
                },
                "schema_version": SCHEMA_VERSION,
            },
            path,
        )
        # Compute SHA-256 of saved file
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        device = str(next(self.parameters()).device)
        return CheckpointDescriptor(
            model_name="arc-simulator",
            model_version=model_version,
            role="simulator",
            parameter_count=self.parameter_count(),
            quantization="fp32",
            device=device,
            artifact_hash=digest.hexdigest(),
        )

    @classmethod
    def load_checkpoint(cls, path: Path) -> tuple["ArcSimulator", CheckpointDescriptor]:
        """Load weights from a checkpoint file. Returns (model, descriptor)."""
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        arch = ckpt.get("architecture", {})
        model = cls(
            feature_dim=arch.get("feature_dim", FEATURE_DIM),
            latent_dim=arch.get("latent_dim", LATENT_DIM),
            action_dim=arch.get("action_dim", ACTION_DIM),
            num_actions=arch.get("num_actions", NUM_ACTIONS),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.checkpoint_transition_version = ckpt.get("transition_version", "legacy-v1")
        model.eval()
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        descriptor = CheckpointDescriptor(
            model_name="arc-simulator",
            model_version=ckpt.get("model_version", "unknown"),
            role="simulator",
            parameter_count=model.parameter_count(),
            quantization="fp32",
            device="cpu",
            artifact_hash=digest.hexdigest(),
        )
        return model, descriptor

    def initial_state(self, *, game_id: str, device: "torch.device | None" = None) -> SimulatorState:
        """Return a zeroed initial SimulatorState for a new game."""
        if device is None:
            device = next(self.parameters()).device
        return SimulatorState(
            h=torch.zeros(1, self.feature_dim, device=device),
            z=torch.zeros(1, self.latent_dim, device=device),
            game_id=game_id,
            model_version=self.VERSION,
        )

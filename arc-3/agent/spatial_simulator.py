"""Spatial-dynamics simulator variant (fix for the movement root cause).

Step 1/2 proved the vector-state ArcSimulator cannot learn position-dependent,
coordinate-free mechanics (movement/collision appear-half = 0): its recurrent
dynamics operate on a GLOBALLY-POOLED vector, so nothing can compute an object's
NEW position as a spatial function of the action.

This module replaces the recurrent core with a SPATIAL one that runs at FULL
grid resolution (not the encoder's H/8 map — that is only 2x2 for a 12x12 grid,
which cannot represent single-cell movement):
  - the hidden state is a full-resolution feature MAP h (B, Ch, H, W),
  - dynamics are a ConvGRU conditioned on the action broadcast across the map,
  - a convolution at full resolution lets a cell's next value depend on its
    immediate NEIGHBOURS — exactly what "the agent moves one cell" needs.

A light full-resolution frame encoder (embedding + same-resolution convs, NO
downsampling) produces the observation map. The change-mask decode reads the
full-resolution state directly, so both the dynamics AND the output are at cell
granularity.

It is SEPARATELY VERSIONED and does not touch ArcSimulator or the production
checkpoint. It is validated on the controlled worlds before any integration.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from agent.simulator import (
    ActionEncoder,
    COLOR_CATEGORIES, COLOR_EMBED_DIM, ACTION_DIM, NUM_ACTIONS,
)

# Spatial-state channel widths. Kept modest to stay in the small-param budget.
SPATIAL_H_CH = 48   # hidden feature-map channels
SPATIAL_Z_CH = 16   # stochastic latent-map channels


class FullResEncoder(nn.Module if _TORCH else object):
    """Full-resolution frame encoder: keeps H x W (no downsampling).

    Color embedding -> two same-resolution conv layers. Preserves single-cell
    position, which the H/8 downsampling encoder destroys.
    Output: obs map (B, out_ch, H, W).
    """

    def __init__(self, color_embed_dim: int = COLOR_EMBED_DIM, out_ch: int = SPATIAL_H_CH):
        _require_torch()
        super().__init__()
        self.color_embed = nn.Embedding(COLOR_CATEGORIES + 1, color_embed_dim)
        self.conv1 = nn.Conv2d(color_embed_dim, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, grid_tokens, validity_mask):
        masked = grid_tokens.clone()
        masked[~validity_mask] = COLOR_CATEGORIES
        emb = self.color_embed(masked).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        x = F.relu(self.conv1(emb))
        return F.relu(self.conv2(x))                                # (B, out_ch, H, W)


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for the spatial simulator.")


@dataclass
class SpatialState:
    """Recurrent state as feature MAPS (not pooled vectors).

    h: (B, SPATIAL_H_CH, Hs, Ws)   spatial hidden map
    z: (B, SPATIAL_Z_CH, Hs, Ws)   spatial stochastic latent map
    """
    h: "torch.Tensor"
    z: "torch.Tensor"
    game_id: str
    model_version: str

    def copy(self) -> "SpatialState":
        return SpatialState(
            h=self.h.detach().clone(),
            z=self.z.detach().clone(),
            game_id=self.game_id,
            model_version=self.model_version,
        )


class ConvGRUCell(nn.Module if _TORCH else object):
    """A convolutional GRU cell over a feature map.

    Convolution is the point: it lets the update at a cell depend on its
    neighbours, so an action can SHIFT content by one cell — the operation a
    pooled-vector GRU cannot express.
    """

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3):
        _require_torch()
        super().__init__()
        p = kernel // 2
        self.update_gate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, padding=p)
        self.reset_gate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, padding=p)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, padding=p)

    def forward(self, x: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.update_gate(xh))
        r = torch.sigmoid(self.reset_gate(xh))
        xrh = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.candidate(xrh))
        return (1 - z) * n + z * h


class SpatialPosterior(nn.Module if _TORCH else object):
    """q(z | h, spatial-obs) as a conv over the concatenated maps -> (mu, logvar)."""

    def __init__(self, h_ch: int, obs_ch: int, z_ch: int):
        _require_torch()
        super().__init__()
        self.body = nn.Conv2d(h_ch + obs_ch, h_ch, 3, padding=1)
        self.mu = nn.Conv2d(h_ch, z_ch, 3, padding=1)
        self.logvar = nn.Conv2d(h_ch, z_ch, 3, padding=1)

    def forward(self, h, obs_spatial):
        x = F.relu(self.body(torch.cat([h, obs_spatial], dim=1)))
        return self.mu(x), self.logvar(x)


class SpatialPrior(nn.Module if _TORCH else object):
    """p(z | h, action-map) as a conv -> (mu, logvar)."""

    def __init__(self, h_ch: int, action_ch: int, z_ch: int):
        _require_torch()
        super().__init__()
        self.body = nn.Conv2d(h_ch + action_ch, h_ch, 3, padding=1)
        self.mu = nn.Conv2d(h_ch, z_ch, 3, padding=1)
        self.logvar = nn.Conv2d(h_ch, z_ch, 3, padding=1)

    def forward(self, h, action_map):
        x = F.relu(self.body(torch.cat([h, action_map], dim=1)))
        return self.mu(x), self.logvar(x)


def reparameterise(mu, logvar):
    std = (0.5 * logvar).exp()
    return mu + torch.randn_like(std) * std


class SpatialArcSimulator(nn.Module if _TORCH else object):
    """Simulator with a spatial recurrent core. Same encoder/decoder as ArcSimulator.

    Interface mirrors ArcSimulator where the training/eval harness needs it:
      encoder(tokens, mask) -> (feature, spatial, hi_res)   [reused, unchanged]
      initial_state(game_id, device, hw)
      update_posterior(obs_spatial, state)
      imagine_step(action, state)
      decode(state, spatial_skip, target_hw, prev_tokens, hi_res_skip)
    """

    VERSION = "sim-spatial-v1"

    def __init__(
        self,
        color_embed_dim: int = COLOR_EMBED_DIM,
        h_ch: int = SPATIAL_H_CH,
        z_ch: int = SPATIAL_Z_CH,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
    ):
        _require_torch()
        super().__init__()
        self.h_ch = h_ch
        self.z_ch = z_ch
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.color_categories = COLOR_CATEGORIES
        self.keep_logit_scale = 12.0
        self._frozen = False

        # Full-resolution encoder (NO downsampling) -> obs map (B, h_ch, H, W).
        self.encoder = FullResEncoder(color_embed_dim, h_ch)
        self.action_encoder = ActionEncoder(num_actions, action_dim)

        # ConvGRU input at each step = [z_map ; action_map]. Action is broadcast
        # across the full-resolution map as a constant-per-channel plane.
        self.dynamics = ConvGRUCell(z_ch + action_dim, h_ch, kernel=3)
        self.posterior = SpatialPosterior(h_ch, h_ch, z_ch)
        self.prior = SpatialPrior(h_ch, action_dim, z_ch)

        # Change-mask heads read the full-resolution state directly.
        self.head_body = nn.Conv2d(h_ch + z_ch, h_ch, 3, padding=1)
        self.update_head = nn.Conv2d(h_ch, COLOR_CATEGORIES, 1)
        self.gate_head = nn.Conv2d(h_ch, 1, 1)
        # Outcome heads operate on the globally-pooled state.
        self.progress_head = nn.Linear(h_ch, 1)
        self.failure_head = nn.Linear(h_ch, 1)
        self.avail_head = nn.Linear(h_ch, num_actions)

    # ---- freeze contract (parity with ArcSimulator) ----
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # ---- helpers ----
    @staticmethod
    def _grid_to_tensors(grid, device):
        H = len(grid)
        W = len(grid[0]) if H else 0
        tokens = torch.zeros(1, H, W, dtype=torch.long, device=device)
        mask = torch.ones(1, H, W, dtype=torch.bool, device=device)
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                tokens[0, r, c] = val
        return tokens, mask

    def _action_map(self, action, hw, device, batch=1):
        """Encode the action to a vector, broadcast to a (B, action_dim, Hs, Ws) map."""
        coord_valid = 1.0 if getattr(action, "x", None) is not None else 0.0
        x_norm = action.x / 63.0 if getattr(action, "x", None) is not None else 0.0
        y_norm = action.y / 63.0 if getattr(action, "y", None) is not None else 0.0
        aid = torch.tensor([action.id], dtype=torch.long, device=device)
        ax = torch.tensor([x_norm], dtype=torch.float32, device=device)
        ay = torch.tensor([y_norm], dtype=torch.float32, device=device)
        av = torch.tensor([coord_valid], dtype=torch.float32, device=device)
        enc = self.action_encoder(aid, ax, ay, av)           # (1, action_dim)
        Hs, Ws = hw
        return enc.view(1, self.action_dim, 1, 1).expand(batch, self.action_dim, Hs, Ws)

    def _action_map_from_batch(self, a_ids, a_x, a_y, a_v, hw, device):
        enc = self.action_encoder(a_ids, a_x, a_y, a_v)      # (B, action_dim)
        B = enc.shape[0]
        Hs, Ws = hw
        return enc.view(B, self.action_dim, 1, 1).expand(B, self.action_dim, Hs, Ws)

    def initial_state(self, *, game_id: str, device=None, hw) -> SpatialState:
        """hw is the FULL grid (H, W): the recurrent state runs at cell resolution."""
        device = device or next(self.parameters()).device
        H, W = hw
        return SpatialState(
            h=torch.zeros(1, self.h_ch, H, W, device=device),
            z=torch.zeros(1, self.z_ch, H, W, device=device),
            game_id=game_id, model_version=self.VERSION,
        )

    def encode_obs(self, tokens, mask):
        """Full-resolution obs map (B, h_ch, H, W)."""
        return self.encoder(tokens, mask)

    def update_posterior(self, obs_map, state: SpatialState) -> SpatialState:
        mu, logvar = self.posterior(state.h, obs_map)
        z = reparameterise(mu, logvar) if self.training else mu
        # The posterior ALSO refreshes h from the observation so the hidden map
        # holds the current scene at full resolution (the thing to be moved).
        return SpatialState(h=obs_map, z=z, game_id=state.game_id,
                            model_version=state.model_version)

    def imagine_step(self, action, state: SpatialState) -> SpatialState:
        device = state.h.device
        Hs, Ws = state.h.shape[2], state.h.shape[3]
        amap = self._action_map(action, (Hs, Ws), device, batch=state.h.shape[0])
        x = torch.cat([state.z, amap], dim=1)
        new_h = self.dynamics(x, state.h)
        mu_p, logvar_p = self.prior(new_h, amap)
        z = reparameterise(mu_p, logvar_p) if self.training else mu_p
        return SpatialState(h=new_h, z=z, game_id=state.game_id,
                            model_version=state.model_version)

    def _heads(self, state: SpatialState, target_hw, prev_tokens):
        """Full-resolution change-mask decode + pooled outcome heads."""
        H, W = target_hw
        feat = F.relu(self.head_body(torch.cat([state.h, state.z], dim=1)))  # (B, h_ch, Hs, Ws)
        if feat.shape[2] != H or feat.shape[3] != W:
            feat = F.interpolate(feat, size=(H, W), mode="nearest")
        update_logits = self.update_head(feat)               # (B, C, H, W)
        gate = torch.sigmoid(self.gate_head(feat))           # (B, 1, H, W)
        if prev_tokens is not None:
            C = self.color_categories
            prev = prev_tokens.clamp(0, C - 1).long()
            keep = F.one_hot(prev, C).permute(0, 3, 1, 2).float() * self.keep_logit_scale
            frame_logits = gate * update_logits + (1.0 - gate) * keep
        else:
            frame_logits = update_logits
        pooled = state.h.mean(dim=(2, 3))                    # (B, h_ch)
        return {
            "frame_logits": frame_logits,
            "change_gate": gate,
            "progress": torch.sigmoid(self.progress_head(pooled)),
            "failure": torch.sigmoid(self.failure_head(pooled)),
            "avail_logits": self.avail_head(pooled),
        }

    def decode(self, state: SpatialState, spatial_skip, target_hw,
               prev_tokens=None, hi_res_skip=None):
        # spatial_skip / hi_res_skip are accepted for interface parity but the
        # full-resolution state already carries the scene; the heads read it.
        return self._heads(state, target_hw, prev_tokens)

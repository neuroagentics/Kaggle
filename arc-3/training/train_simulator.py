"""Simulator training scaffold — R10 implementation.

Defines loss functions, the trajectory data-loader contract, and the training
loop for the ArcSimulator. No real training data is loaded here; the DataLoader
contract specifies what the collector (G2) must supply.

Loss functions (blueprint §4):
  1. masked_frame_loss:   categorical cross-entropy on next-frame logits,
                          reporting changed and unchanged regions separately.
  2. multistep_loss:      aggregates 1/2/4-step prediction losses.
  3. kl_loss:             KL divergence between posterior and prior.
  4. outcome_loss:        supervised BCE on progress/failure/avail where labels
                          are actually observed (not fabricated).

Training contract:
  - Splits are frozen before fitting (SplitManifest.frozen must be True).
  - No cross-split trajectories; holdout games/families are never used for tuning.
  - Adapter updates: one small adapter per decision; validated on disjoint slice;
    rolled back on non-finite or degraded loss.
  - All hyperparameters are published in the training config; none are tuned on
    private scores.

Imports PyTorch; skips gracefully if unavailable (contract tests use stubs).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Protocol, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training. "
            "Install it with the project's training dependencies."
        )


# ---------------------------------------------------------------------------
# Trajectory batch contract
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryBatch:
    """One batch of real observed trajectories.

    All tensors have shape (B, T, ...) where B=batch size, T=trajectory length.
    Grids are (B, T, H, W) int64. Masks are (B, T, H, W) bool.

    Fields
    ------
    grid_tokens     : (B, T, H, W) int64 — observed color categories 0..15
    validity_mask   : (B, T, H, W) bool  — True for valid (non-padded) cells
    action_ids      : (B, T) int64
    action_x        : (B, T) float32, normalised 0..1; -1 if N/A
    action_y        : (B, T) float32, normalised 0..1; -1 if N/A
    coord_valid     : (B, T) float32, 0 or 1
    progress_labels : (B, T) float32, in [0,1]; -1 where unobserved
    failure_labels  : (B, T) float32, 0 or 1;  -1 where unobserved
    avail_labels    : (B, T, NUM_ACTIONS) float32, 0 or 1; -1 where unobserved
    game_ids        : list[str] of length B
    level_ids       : (B, T) int64
    """
    grid_tokens: "torch.Tensor"       # (B, T, H, W) int64
    validity_mask: "torch.Tensor"     # (B, T, H, W) bool
    action_ids: "torch.Tensor"        # (B, T) int64
    action_x: "torch.Tensor"         # (B, T) float32
    action_y: "torch.Tensor"         # (B, T) float32
    coord_valid: "torch.Tensor"      # (B, T) float32
    progress_labels: "torch.Tensor"  # (B, T) float32
    failure_labels: "torch.Tensor"   # (B, T) float32
    avail_labels: "torch.Tensor"     # (B, T, NUM_ACTIONS) float32
    game_ids: list[str]
    level_ids: "torch.Tensor"        # (B, T) int64

    def to(self, device: "torch.device") -> "TrajectoryBatch":
        """Move all tensors to device."""
        return TrajectoryBatch(
            grid_tokens=self.grid_tokens.to(device),
            validity_mask=self.validity_mask.to(device),
            action_ids=self.action_ids.to(device),
            action_x=self.action_x.to(device),
            action_y=self.action_y.to(device),
            coord_valid=self.coord_valid.to(device),
            progress_labels=self.progress_labels.to(device),
            failure_labels=self.failure_labels.to(device),
            avail_labels=self.avail_labels.to(device),
            game_ids=self.game_ids,
            level_ids=self.level_ids.to(device),
        )


class TrajectoryDataset(Protocol):
    """Contract for the trajectory data provider (G2/R09 collector output).

    Implementations must:
      - Yield only trajectories from the assigned split (train/dev/holdout).
      - Never include cross-split or holdout trajectories.
      - Report game_id, split, and mechanic_family_tags for each trajectory.
    """

    def __iter__(self) -> Iterator[TrajectoryBatch]: ...
    def __len__(self) -> int: ...
    def split(self) -> str: ...               # 'train', 'dev', or 'holdout'
    def num_games(self) -> int: ...


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_frame_loss(
    frame_logits: "torch.Tensor",      # (B, 16, H, W)
    target_tokens: "torch.Tensor",     # (B, H, W) int64
    validity_mask: "torch.Tensor",     # (B, H, W) bool
    prev_tokens: "torch.Tensor | None" = None,  # (B, H, W) int64 for changed/unchanged split
    changed_weight: float = 1.0,       # up-weight cells that actually change
) -> dict[str, "torch.Tensor"]:
    """Categorical cross-entropy on predicted next-frame logits.

    Reports total, changed-region, and unchanged-region losses separately.
    Padding cells (validity_mask=False) are excluded from all losses.

    changed_weight > 1 up-weights cells whose value differs from prev_tokens, so
    the (minority) changed cells are not drowned out by the ~75% unchanged cells.
    Diagnosis showed the simulator learned static structure well but was weak on
    changed regions precisely because the unchanged majority dominated the gradient.

    blueprint §4: masked categorical frame loss with changed/unchanged regions
    reported separately.
    """
    _require_torch()
    B, C, H, W = frame_logits.shape

    # Flatten spatial dims for cross-entropy
    logits_flat = frame_logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, 16)
    target_flat = target_tokens.reshape(-1)                          # (B*H*W,)
    valid_flat = validity_mask.reshape(-1)                           # (B*H*W,)

    # Per-cell cross-entropy (no reduction)
    loss_per_cell = F.cross_entropy(logits_flat, target_flat, reduction="none")  # (B*H*W,)
    loss_per_cell = loss_per_cell * valid_flat.float()

    if prev_tokens is not None and changed_weight != 1.0:
        # Build per-cell weights: changed cells get `changed_weight`, else 1.0.
        changed_flat_w = (target_tokens != prev_tokens).reshape(-1).float()
        weights = 1.0 + (changed_weight - 1.0) * changed_flat_w
        weighted = loss_per_cell * weights
        denom = (weights * valid_flat.float()).sum().clamp(min=1)
        total_loss = weighted.sum() / denom
    else:
        total_valid = valid_flat.float().sum().clamp(min=1)
        total_loss = loss_per_cell.sum() / total_valid

    result = {"total": total_loss, "changed": total_loss, "unchanged": total_loss}

    if prev_tokens is not None:
        changed_mask = (target_tokens != prev_tokens) & validity_mask  # (B, H, W)
        unchanged_mask = (target_tokens == prev_tokens) & validity_mask

        changed_flat = changed_mask.reshape(-1).float()
        unchanged_flat = unchanged_mask.reshape(-1).float()

        n_changed = changed_flat.sum().clamp(min=1)
        n_unchanged = unchanged_flat.sum().clamp(min=1)

        result["changed"] = (loss_per_cell * changed_flat).sum() / n_changed
        result["unchanged"] = (loss_per_cell * unchanged_flat).sum() / n_unchanged

    return result


def kl_loss(
    mu_q: "torch.Tensor",
    logvar_q: "torch.Tensor",
    mu_p: "torch.Tensor",
    logvar_p: "torch.Tensor",
) -> "torch.Tensor":
    """KL divergence KL(q || p) for diagonal Gaussians.

    blueprint §4: posterior/prior KL regularisation.
    Returns mean KL over the batch.
    """
    _require_torch()
    # KL(N(mu_q, sig_q) || N(mu_p, sig_p))
    # = 0.5 * sum(logvar_p - logvar_q - 1 + sig_q^2/sig_p^2 + (mu_q-mu_p)^2/sig_p^2)
    kl = 0.5 * (
        logvar_p - logvar_q - 1.0
        + (logvar_q.exp()) / (logvar_p.exp() + 1e-8)
        + (mu_q - mu_p).pow(2) / (logvar_p.exp() + 1e-8)
    )
    return kl.sum(dim=-1).mean()


def outcome_loss(
    progress_pred: "torch.Tensor",   # (B, 1) or (B,) in [0,1]
    failure_pred: "torch.Tensor",    # (B, 1) or (B,) in [0,1]
    avail_pred: "torch.Tensor",      # (B, NUM_ACTIONS)
    progress_labels: "torch.Tensor", # (B,) float32; -1 where unobserved
    failure_labels: "torch.Tensor",  # (B,) float32; -1 where unobserved
    avail_labels: "torch.Tensor",    # (B, NUM_ACTIONS) float32; -1 where unobserved
) -> dict[str, "torch.Tensor"]:
    """Supervised BCE on outcome heads, only where labels are actually observed.

    blueprint §4: supervised outcome/availability losses where labels are
    actually observed. Do not use fabricated goal labels or hidden state.
    """
    _require_torch()
    progress_pred = progress_pred.squeeze(-1)  # (B,)
    failure_pred = failure_pred.squeeze(-1)    # (B,)

    def _masked_bce(pred: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        observed = labels >= 0.0
        if not observed.any():
            return torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        return F.binary_cross_entropy(
            pred[observed], labels[observed], reduction="mean"
        )

    progress_l = _masked_bce(progress_pred, progress_labels)
    failure_l = _masked_bce(failure_pred, failure_labels)

    # Per-action availability loss
    avail_pred_sig = torch.sigmoid(avail_pred)  # (B, NUM_ACTIONS)
    avail_observed = avail_labels >= 0.0         # (B, NUM_ACTIONS)
    if avail_observed.any():
        avail_l = F.binary_cross_entropy(
            avail_pred_sig[avail_observed],
            avail_labels[avail_observed],
            reduction="mean",
        )
    else:
        avail_l = torch.zeros(1, device=avail_pred.device, dtype=avail_pred.dtype).squeeze()

    return {"progress": progress_l, "failure": failure_l, "avail": avail_l}


def multistep_loss(
    per_step_losses: Sequence["torch.Tensor"],
    steps: Sequence[int] = (1, 2, 4),
    weights: Sequence[float] = (1.0, 0.5, 0.25),
) -> "torch.Tensor":
    """Weighted sum of per-step frame losses.

    blueprint §4: multi-step prediction loss (1/2/4 steps).
    per_step_losses should have one entry per step in steps.
    """
    _require_torch()
    if len(per_step_losses) != len(steps):
        raise ValueError(
            f"per_step_losses length {len(per_step_losses)} != steps length {len(steps)}"
        )
    if len(weights) != len(steps):
        raise ValueError(
            f"weights length {len(weights)} != steps length {len(steps)}"
        )
    total = sum(w * l for w, l in zip(weights, per_step_losses))
    return total


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Published training hyperparameters.

    All values here are the defaults; none may be tuned on private scores.
    Record this config alongside every checkpoint for reproducibility.
    """
    # Model
    feature_dim: int = 256
    latent_dim: int = 64
    action_dim: int = 32
    num_actions: int = 8
    color_embed_dim: int = 16

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    max_epochs: int = 100
    grad_clip_norm: float = 5.0

    # Loss weights
    frame_loss_weight: float = 1.0
    kl_weight: float = 0.1
    outcome_loss_weight: float = 0.5
    multistep_weights: tuple = (1.0, 0.5, 0.25)
    changed_cell_weight: float = 5.0   # up-weight changed cells in frame loss

    # Multi-step
    rollout_steps: tuple = (1, 2, 4)

    # Adapter
    adapter_replay_min_samples: int = 16
    adapter_max_updates_per_decision: int = 1
    adapter_validation_fraction: float = 0.2

    # Stopping / budgets
    early_stop_patience: int = 10
    max_gpu_hours: float = 48.0

    # Provenance
    config_version: str = "1.0"
    notes: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "TrainingConfig":
        d = json.loads(text)
        d.pop("config_version", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def compute_losses(
    model: "ArcSimulator",
    batch: "TrajectoryBatch",
    config: TrainingConfig,
    device: "torch.device",
) -> tuple["torch.Tensor", dict[str, float]]:
    """Forward pass over a trajectory batch.

    Returns (total_loss_tensor, logging_dict). The tensor retains the autograd
    graph so the caller can call .backward() on it. The dict holds detached
    float values for logging only.

    blueprint §4:
      - masked categorical frame loss with changed/unchanged reported separately
      - multi-step prediction loss (1/2/4 steps)
      - posterior/prior KL regularisation
      - supervised outcome/availability losses where labels are actually observed
    """
    _require_torch()
    from agent.simulator import ArcSimulator, reparameterise

    batch = batch.to(device)
    B, T, H, W = batch.grid_tokens.shape

    # Initialise recurrent state
    h = torch.zeros(B, model.feature_dim, device=device)
    total_frame_loss = torch.zeros(1, device=device)
    total_kl = torch.zeros(1, device=device)
    total_outcome = torch.zeros(1, device=device)
    step_frame_losses: list[torch.Tensor] = []

    for t in range(T - 1):
        # --- Encode current observation ---
        tokens_t = batch.grid_tokens[:, t]       # (B, H, W)
        mask_t = batch.validity_mask[:, t]        # (B, H, W)
        obs_feat, spatial, hi_res = model.encoder(tokens_t, mask_t)

        # --- Posterior: q(z | h, obs) ---
        mu_q, logvar_q = model.posterior(h, obs_feat)

        # --- Encode action ---
        a_ids = batch.action_ids[:, t]
        a_x = batch.action_x[:, t]
        a_y = batch.action_y[:, t]
        a_v = batch.coord_valid[:, t]
        action_enc = model.action_encoder(a_ids, a_x, a_y, a_v)

        # --- Prior: p(z | h, a) ---
        mu_p, logvar_p = model.prior(h, action_enc)

        # --- Sample z from posterior (training) ---
        z = reparameterise(mu_q, logvar_q)

        # --- Advance GRU ---
        h_new = model.dynamics(h, z, action_enc)

        # --- Decode to next-frame prediction ---
        tokens_tp1 = batch.grid_tokens[:, t + 1]
        mask_tp1 = batch.validity_mask[:, t + 1]
        next_mu, next_logvar = model.prior(h_new, action_enc)
        next_z = reparameterise(next_mu, next_logvar)
        outputs = model.decoder(h_new, next_z, spatial, (H, W),
                                prev_tokens=tokens_t, hi_res_skip=hi_res)

        # --- Frame loss ---
        frame_losses = masked_frame_loss(
            outputs["frame_logits"], tokens_tp1, mask_tp1,
            prev_tokens=tokens_t,
            changed_weight=config.changed_cell_weight,
        )
        total_frame_loss = total_frame_loss + frame_losses["total"]
        # Keep the graph: multi-step loss must be differentiable.
        step_frame_losses.append(frame_losses["total"])

        # --- KL loss ---
        kl = kl_loss(mu_q, logvar_q, mu_p, logvar_p)
        total_kl = total_kl + kl

        # --- Outcome loss ---
        out_losses = outcome_loss(
            outputs["progress"], outputs["failure"], outputs["avail_logits"],
            batch.progress_labels[:, t + 1],
            batch.failure_labels[:, t + 1],
            batch.avail_labels[:, t + 1],
        )
        step_outcome = out_losses["progress"] + out_losses["failure"] + out_losses["avail"]
        total_outcome = total_outcome + step_outcome

        h = h_new

    steps = float(max(T - 1, 1))
    avg_frame = (total_frame_loss / steps).squeeze()
    avg_kl = (total_kl / steps).squeeze()
    avg_outcome = (total_outcome / steps).squeeze()

    # GENUINE multi-step rollout loss: from a real anchor, roll the model forward
    # using its OWN predicted state (prior only, NO posterior correction) and
    # penalise the frame error at each imagined horizon against the real future.
    # This penalises errors ACCUMULATED THROUGH IMAGINED FUTURES — the property
    # the planner depends on — unlike the previous term which just reused
    # teacher-forced one-step losses at different positions.
    ms = _imagined_rollout_loss(model, batch, config, device)

    total = (
        config.frame_loss_weight * avg_frame
        + config.kl_weight * avg_kl
        + config.outcome_loss_weight * avg_outcome
        + config.frame_loss_weight * ms
    )

    logs = {
        "total": float(total.item()),
        "frame": float(avg_frame.item()),
        "kl": float(avg_kl.item()),
        "outcome": float(avg_outcome.item()),
        "multistep": float(ms.item()),
    }
    return total, logs


def training_step(
    model: "ArcSimulator",
    batch: "TrajectoryBatch",
    config: TrainingConfig,
    device: "torch.device",
) -> dict[str, float]:
    """Compute losses and return the logging dict only (no backward).

    Retained for callers that only need metrics (e.g. validation). For training
    that updates weights, use compute_losses() and call .backward() on the tensor,
    or use the train() loop below.
    """
    _, logs = compute_losses(model, batch, config, device)
    return logs


def _imagined_rollout_loss(model, batch, config, device):
    """Genuine multi-step rollout loss over IMAGINED futures.

    From a real anchor frame, prime the recurrent state via the posterior, then
    roll the model forward using its OWN prior-sampled latent for each subsequent
    action — NO posterior correction from intermediate real frames. Decode at
    each horizon k in config.rollout_steps and penalise the frame error against
    the REAL frame at anchor+k. This is what makes the model's imagined rollouts
    (which the planner relies on) accurate, rather than only its teacher-forced
    one-step predictions.

    Returns a scalar tensor (weighted sum over horizons). Zero when the
    trajectory is too short for the requested horizons.
    """
    _require_torch()
    from agent.simulator import reparameterise

    B, T, H, W = batch.grid_tokens.shape
    steps = tuple(config.rollout_steps)
    weights = tuple(config.multistep_weights)
    max_k = max(steps) if steps else 0
    max_k = min(max_k, T - 1)
    if max_k < 1:
        return torch.zeros((), device=device)

    anchor = 0
    # Prime recurrent state on the anchor observation (posterior-corrected).
    tokens0 = batch.grid_tokens[:, anchor]
    mask0 = batch.validity_mask[:, anchor]
    obs_feat, spatial, hi_res = model.encoder(tokens0, mask0)
    h = torch.zeros(B, model.feature_dim, device=device)
    mu_q, _ = model.posterior(h, obs_feat)
    z = mu_q

    total = torch.zeros((), device=device)
    wsum = 0.0
    cur_h, cur_z = h, z
    horizon_set = set(steps)
    # prev_frame for change-mask decoding: for k=1 the real anchor, thereafter the
    # model's OWN previous imagined frame (never the real intermediate frame, so
    # the rollout stays a genuine imagined future).
    prev_frame = tokens0
    for k in range(1, max_k + 1):
        # Roll forward one imagined step using the PRIOR (own prediction), driven
        # by the real action taken at that step — but never re-reading the frame.
        a_ids = batch.action_ids[:, anchor + k - 1]
        a_x = batch.action_x[:, anchor + k - 1]
        a_y = batch.action_y[:, anchor + k - 1]
        a_v = batch.coord_valid[:, anchor + k - 1]
        action_enc = model.action_encoder(a_ids, a_x, a_y, a_v)
        cur_h = model.dynamics(cur_h, cur_z, action_enc)
        mu_p, logvar_p = model.prior(cur_h, action_enc)
        cur_z = reparameterise(mu_p, logvar_p)
        outputs = model.decoder(cur_h, cur_z, spatial, (H, W),
                                prev_tokens=prev_frame, hi_res_skip=hi_res)

        if k in horizon_set:
            target = batch.grid_tokens[:, anchor + k]
            tmask = batch.validity_mask[:, anchor + k]
            prev = batch.grid_tokens[:, anchor + k - 1]
            fl = masked_frame_loss(
                outputs["frame_logits"], target, tmask,
                prev_tokens=prev, changed_weight=config.changed_cell_weight,
            )
            w = weights[steps.index(k)] if k in steps else 1.0
            total = total + w * fl["total"]
            wsum += w
        # Runtime re-encodes the imagined frame, not the real intermediate frame.
        imagined_tokens = outputs["frame_logits"].argmax(dim=1).detach()
        _, spatial, hi_res = model.encoder(imagined_tokens, mask0)
        prev_frame = imagined_tokens  # next step preserves from our own prediction

    return total / wsum if wsum > 0 else total


# ---------------------------------------------------------------------------
# Adapter update with rollback
# ---------------------------------------------------------------------------

def adapter_update(
    model: "ArcSimulator",
    replay_batch: "TrajectoryBatch",
    val_batch: "TrajectoryBatch",
    config: TrainingConfig,
    device: "torch.device",
) -> dict[str, object]:
    """Attempt one bounded adapter update; roll back if degraded.

    blueprint §5: propose at most one small adapter update per decision, after
    enough real samples exist. Validate on a disjoint recent real replay slice;
    roll back non-finite or degraded updates.

    Returns a dict with keys: 'accepted' (bool), 'train_loss', 'val_loss_before',
    'val_loss_after'.
    """
    _require_torch()
    import copy

    if replay_batch.grid_tokens.shape[0] < config.adapter_replay_min_samples:
        return {
            "accepted": False,
            "reason": f"insufficient samples ({replay_batch.grid_tokens.shape[0]} "
                      f"< {config.adapter_replay_min_samples})",
        }

    # Snapshot weights before update
    state_before = copy.deepcopy(model.state_dict())

    # Baseline val loss
    model.eval()
    with torch.no_grad():
        val_before = training_step(model, val_batch, config, device)["total"]

    # Single gradient update — use the live loss tensor so backward works.
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer.zero_grad()
    total_tensor, losses = compute_losses(model, replay_batch, config, device)
    train_loss = losses["total"]

    if not math.isfinite(train_loss):
        model.load_state_dict(state_before)
        return {"accepted": False, "reason": "non-finite train loss", "train_loss": train_loss}

    total_tensor.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_after = training_step(model, val_batch, config, device)["total"]

    # Roll back if degraded or non-finite
    if not math.isfinite(val_after) or val_after > val_before * 1.05:
        model.load_state_dict(state_before)
        return {
            "accepted": False,
            "reason": "val loss degraded",
            "train_loss": train_loss,
            "val_loss_before": val_before,
            "val_loss_after": val_after,
        }

    return {
        "accepted": True,
        "train_loss": train_loss,
        "val_loss_before": val_before,
        "val_loss_after": val_after,
    }


# ---------------------------------------------------------------------------
# Checkpoint save / load helpers
# ---------------------------------------------------------------------------

def save_training_state(
    path: Path,
    *,
    model: "ArcSimulator",
    optimizer_state: dict,
    epoch: int,
    config: TrainingConfig,
    metrics: dict,
) -> None:
    """Save full training state (weights + optimiser + config + metrics)."""
    _require_torch()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


def load_training_state(path: Path) -> dict:
    """Load a training state checkpoint. Returns raw dict; caller reconstructs model."""
    _require_torch()
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: "ArcSimulator",
    train_dataset,
    config: TrainingConfig,
    device: "torch.device",
    *,
    val_dataset=None,
    max_epochs: int | None = None,
    log_every: int = 10,
    on_log=None,
) -> dict:
    """Train the simulator with real backward passes and gradient clipping.

    Returns a history dict of per-epoch train/val metrics. Uses early stopping
    on validation total loss with config.early_stop_patience.
    """
    _require_torch()
    epochs = max_epochs if max_epochs is not None else config.max_epochs
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        step = 0
        epoch_total = 0.0
        n = 0
        for batch in train_dataset:
            optimizer.zero_grad()
            total_tensor, logs = compute_losses(model, batch, config, device)
            if not math.isfinite(logs["total"]):
                # Skip a bad batch rather than corrupt the run.
                continue
            total_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_total += logs["total"]
            n += 1
            step += 1
            if on_log and step % log_every == 0:
                on_log(epoch, step, logs)
        train_mean = epoch_total / max(1, n)
        history["train"].append({"epoch": epoch, "mean_total": train_mean})

        # Validation
        if val_dataset is not None:
            model.eval()
            v_total = 0.0
            v_n = 0
            with torch.no_grad():
                for vbatch in val_dataset:
                    logs = training_step(model, vbatch, config, device)
                    if math.isfinite(logs["total"]):
                        v_total += logs["total"]
                        v_n += 1
            val_mean = v_total / max(1, v_n)
            history["val"].append({"epoch": epoch, "mean_total": val_mean})
            if val_mean < best_val - 1e-4:
                best_val = val_mean
                best_state = copy_state(model)
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stop_patience:
                    history["stopped_early_at"] = epoch
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best_val if best_val != float("inf") else None
    return history


def copy_state(model) -> dict:
    """Deep-copy a model state_dict to CPU (for best-checkpoint retention)."""
    import copy
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})


import math  # noqa: E402  needed by adapter_update and train

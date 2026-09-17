"""Bounded online adapter learning with anti-overfitting safeguards — R12.

Wraps a single small adapter update per decision with the guarantees the
blueprint (§5) and the issue register (R12) require:

  - TEMPORAL validation: the validation slice is the most RECENT real samples,
    disjoint from the (older) training slice, so we measure generalisation to
    new experience rather than fit to the same batch.
  - ROLLBACK: non-finite or degraded validation loss reverts the weights; the
    base model is never corrupted by a bad update.
  - PER-DECISION CAP: at most `max_updates_per_decision` accepted updates per
    real decision, enforced across calls (not just within one call).
  - VERSION TRACKING: every accepted update bumps an adapter version; rejected
    updates do not. The history records accepted/rejected with evidence.

This does NOT invent gradients during imagined branches — it is only called
between real decisions, on real replay. Freezing/imagination isolation is the
simulator's concern (ArcSimulator.freeze); this manager assumes it is called
with real replay only.

torch is imported lazily; the manager's bookkeeping is testable without a model.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for adapter updates.")


@dataclass
class AdapterUpdateResult:
    """Outcome of one attempted adapter update."""
    accepted: bool
    reason: str
    adapter_version: int
    train_loss: float | None = None
    val_loss_before: float | None = None
    val_loss_after: float | None = None
    n_train: int = 0
    n_val: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "adapter_version": self.adapter_version,
            "train_loss": self.train_loss,
            "val_loss_before": self.val_loss_before,
            "val_loss_after": self.val_loss_after,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


def temporal_split(batch, val_fraction: float):
    """Split a TrajectoryBatch into (train, val) by RECENCY along the batch axis.

    The batch is assumed ordered oldest→newest along dim 0 (the collector and
    replay buffer preserve insertion order). The most-recent `val_fraction`
    slice becomes validation; the older remainder is training. This makes the
    validation set temporally disjoint from training.

    Returns (train_batch, val_batch) or (None, None) if either slice is empty.
    """
    _require_torch()
    from training.train_simulator import TrajectoryBatch

    B = batch.grid_tokens.shape[0]
    if B < 2:
        return None, None
    n_val = max(1, int(round(B * val_fraction)))
    n_val = min(n_val, B - 1)  # leave at least 1 for train
    n_train = B - n_val

    def _slice(lo, hi):
        return TrajectoryBatch(
            grid_tokens=batch.grid_tokens[lo:hi],
            validity_mask=batch.validity_mask[lo:hi],
            action_ids=batch.action_ids[lo:hi],
            action_x=batch.action_x[lo:hi],
            action_y=batch.action_y[lo:hi],
            coord_valid=batch.coord_valid[lo:hi],
            progress_labels=batch.progress_labels[lo:hi],
            failure_labels=batch.failure_labels[lo:hi],
            avail_labels=batch.avail_labels[lo:hi],
            game_ids=batch.game_ids[lo:hi],
            level_ids=batch.level_ids[lo:hi],
        )

    return _slice(0, n_train), _slice(n_train, B)


class AdapterManager:
    """Owns adapter version, per-decision cap, and update history.

    Usage (between real decisions, on real replay only):
        mgr = AdapterManager(config)
        mgr.begin_decision()                    # resets the per-decision counter
        result = mgr.maybe_update(model, replay_batch, device)
    """

    def __init__(self, config, *, regression_tolerance: float = 1.05, parameter_names=None):
        self.config = config
        self.parameter_names = set(parameter_names) if parameter_names is not None else None
        if not (regression_tolerance >= 1.0):
            raise ValueError("regression_tolerance must be >= 1.0")
        self.regression_tolerance = regression_tolerance
        self._adapter_version = 0
        self._updates_this_decision = 0
        self._history: list[AdapterUpdateResult] = []
        self._accepted = 0
        self._rejected = 0

    @property
    def adapter_version(self) -> int:
        return self._adapter_version

    def begin_decision(self) -> None:
        """Reset the per-decision update counter. Call once per real decision."""
        self._updates_this_decision = 0

    def history(self) -> list[dict]:
        return [r.to_dict() for r in self._history]

    def stats(self) -> dict:
        return {
            "adapter_version": self._adapter_version,
            "accepted": self._accepted,
            "rejected": self._rejected,
            "total_attempts": len(self._history),
        }

    def _record(self, result: AdapterUpdateResult) -> AdapterUpdateResult:
        self._history.append(result)
        if result.accepted:
            self._accepted += 1
        else:
            self._rejected += 1
        return result

    def maybe_update(self, model, replay_batch, device, *, deadline=float("inf")) -> AdapterUpdateResult:
        """Exception-safe game-local update; preserve caller RNG, mode and gradients."""
        import time
        if time.monotonic() >= deadline:
            return self._record(AdapterUpdateResult(False, "deadline exhausted", self._adapter_version))
        state = copy.deepcopy(model.state_dict())
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        was_training = model.training
        devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        try:
            if self.parameter_names is not None:
                if not self.parameter_names.issubset(flags):
                    raise ValueError("Unknown adapter parameter")
                for name, p in model.named_parameters():
                    p.requires_grad_(name in self.parameter_names)
            with torch.random.fork_rng(devices=devices):
                return self._maybe_update_impl(model, replay_batch, device, deadline=deadline)
        except Exception as exc:
            model.load_state_dict(state)
            return self._record(AdapterUpdateResult(False, f"exception rolled back: {exc}", self._adapter_version))
        finally:
            model.train(was_training)
            for name, p in model.named_parameters():
                p.requires_grad_(flags[name])
                p.grad = None

    def _maybe_update_impl(self, model, replay_batch, device, *, deadline) -> AdapterUpdateResult:
        """Attempt one bounded, validated adapter update. Never corrupts the model.

        Enforces:
          - per-decision cap (across calls within the same decision),
          - minimum replay samples,
          - temporal train/val split,
          - rollback on non-finite or degraded validation loss.
        """
        _require_torch()
        from training.train_simulator import compute_losses, training_step

        cap = self.config.adapter_max_updates_per_decision
        if self._updates_this_decision >= cap:
            return self._record(AdapterUpdateResult(
                False, f"per-decision cap reached ({cap})", self._adapter_version,
            ))

        B = replay_batch.grid_tokens.shape[0]
        if B < self.config.adapter_replay_min_samples:
            return self._record(AdapterUpdateResult(
                False,
                f"insufficient samples ({B} < {self.config.adapter_replay_min_samples})",
                self._adapter_version,
            ))

        train_batch, val_batch = temporal_split(
            replay_batch, self.config.adapter_validation_fraction
        )
        if train_batch is None:
            return self._record(AdapterUpdateResult(
                False, "cannot form disjoint temporal split", self._adapter_version,
            ))

        n_train = train_batch.grid_tokens.shape[0]
        n_val = val_batch.grid_tokens.shape[0]

        # Snapshot weights before update — the safety net for rollback.
        state_before = copy.deepcopy(model.state_dict())
        self._updates_this_decision += 1  # attempts, not just accepted updates

        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            val_before = training_step(model, val_batch, self.config, device)["total"]
        if not math.isfinite(val_before):
            return self._record(AdapterUpdateResult(False, "non-finite validation baseline", self._adapter_version))

        # One gradient step on the (older) training slice.
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer.zero_grad()
        total_tensor, logs = compute_losses(model, train_batch, self.config, device)
        train_loss = logs["total"]

        if not math.isfinite(train_loss):
            model.load_state_dict(state_before)
            return self._record(AdapterUpdateResult(
                False, "non-finite train loss", self._adapter_version,
                train_loss=train_loss, n_train=n_train, n_val=n_val,
            ))

        total_tensor.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
        optimizer.step()

        # Validate on the temporally-disjoint recent slice.
        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            val_after = training_step(model, val_batch, self.config, device)["total"]

        import time
        if (time.monotonic() >= deadline or not math.isfinite(val_after)
                or val_after > val_before * self.regression_tolerance):
            model.load_state_dict(state_before)
            return self._record(AdapterUpdateResult(
                False, "validation regressed — rolled back", self._adapter_version,
                train_loss=train_loss, val_loss_before=val_before,
                val_loss_after=val_after, n_train=n_train, n_val=n_val,
            ))

        # Accepted: bump version and per-decision counter.
        self._adapter_version += 1
        return self._record(AdapterUpdateResult(
            True, "accepted", self._adapter_version,
            train_loss=train_loss, val_loss_before=val_before,
            val_loss_after=val_after, n_train=n_train, n_val=n_val,
        ))

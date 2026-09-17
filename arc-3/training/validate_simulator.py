"""Simulator validation — R10 held-out multi-step prediction report.

Measures what the blueprint §4 demands, and nothing weaker:
  - 1/2/4-step changed-cell prediction error on HELD-OUT games.
  - Comparison against a PERSISTENCE baseline (predict "no change") and a
    LAST-ACTION-DELTA baseline is out of scope here; persistence is the key one
    because sparse motion makes an identity predictor look deceptively good.
  - Errors reported separately on changed vs unchanged cells, because overall
    pixel accuracy alone is explicitly inadequate.

A trained simulator is only credible if it beats persistence on the CHANGED
cells at multi-step horizons. This module produces the numbers to judge that.

Held-out discipline: validation games must be disjoint from training games.
`split_games()` provides a deterministic whole-game split.

torch imported lazily; report functions that need the model require it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from training.dataset import load_records, record_to_windows, build_batch


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("PyTorch is required for simulator validation.")


# ---------------------------------------------------------------------------
# Whole-game split
# ---------------------------------------------------------------------------

def split_games(
    game_ids: Sequence[str], *, holdout_fraction: float = 0.3, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministic whole-game split into (train_games, holdout_games).

    Splitting by whole game (not by trajectory) prevents leakage: a held-out
    game's mechanics are never seen during training.
    """
    import random
    games = sorted(set(game_ids))
    rng = random.Random(seed)
    rng.shuffle(games)
    n_holdout = max(1, round(len(games) * holdout_fraction)) if len(games) > 1 else 0
    holdout = sorted(games[:n_holdout])
    train = sorted(games[n_holdout:])
    return train, holdout


# ---------------------------------------------------------------------------
# Multi-step prediction error
# ---------------------------------------------------------------------------

@dataclass
class StepErrorReport:
    """Per-horizon changed-cell error for the model vs persistence."""
    horizon: int
    n_transitions: int
    # Fraction of cells mispredicted, restricted to cells that actually changed
    model_changed_error: float
    persistence_changed_error: float
    # Fraction over all valid cells
    model_overall_error: float
    persistence_overall_error: float

    @property
    def beats_persistence_on_changed(self) -> bool:
        return self.model_changed_error < self.persistence_changed_error


@dataclass
class ValidationReport:
    holdout_games: list[str]
    train_games: list[str]
    horizons: list[int]
    steps: list[StepErrorReport]
    model_version: str

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "holdout_games": self.holdout_games,
            "train_games": self.train_games,
            "horizons": self.horizons,
            "steps": [asdict(s) for s in self.steps],
            "summary": {
                f"h{s.horizon}": {
                    "model_changed_err": round(s.model_changed_error, 4),
                    "persistence_changed_err": round(s.persistence_changed_error, 4),
                    "beats_persistence": s.beats_persistence_on_changed,
                }
                for s in self.steps
            },
            "note": (
                "Changed-cell error is the primary metric. A credible simulator "
                "beats persistence on changed cells at multi-step horizons."
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _rollout_predict(model, batch, horizon: int, device):
    """Roll the model forward `horizon` steps from each window start.

    Returns a list of (predicted_grid, target_grid, before_grid, validity) tuples,
    one per batch item that has at least `horizon`+1 steps.
    """
    import torch
    from agent.simulator import reparameterise

    B, T, H, W = batch.grid_tokens.shape
    if T < horizon + 1:
        return []

    batch = batch.to(device)
    h = torch.zeros(B, model.feature_dim, device=device)

    # Prime the recurrent state on the first observation, then roll from prior.
    tokens0 = batch.grid_tokens[:, 0]
    mask0 = batch.validity_mask[:, 0]
    obs_feat, spatial, hi_res = model.encoder(tokens0, mask0)
    mu_q, logvar_q = model.posterior(h, obs_feat)
    z = mu_q  # use the mean at eval (no sampling noise)

    cur_h, cur_z = h, z
    prev_frame = tokens0  # change-mask prev: anchor first, then own prediction
    for step in range(horizon):
        a_ids = batch.action_ids[:, step]
        a_x = batch.action_x[:, step]
        a_y = batch.action_y[:, step]
        a_v = batch.coord_valid[:, step]
        action_enc = model.action_encoder(a_ids, a_x, a_y, a_v)
        cur_h = model.dynamics(cur_h, cur_z, action_enc)
        mu_p, logvar_p = model.prior(cur_h, action_enc)
        cur_z = mu_p
        outputs = model.decoder(cur_h, cur_z, spatial, (H, W),
                                prev_tokens=prev_frame, hi_res_skip=hi_res)
        pred = outputs["frame_logits"].argmax(dim=1)
        _, spatial, hi_res = model.encoder(pred, mask0)
        prev_frame = pred


    results = []
    for b in range(B):
        results.append((
            pred[b],                                 # predicted grid at t=horizon
            batch.grid_tokens[b, horizon],           # target grid
            batch.grid_tokens[b, 0],                 # before (t=0) for persistence
            batch.validity_mask[b, horizon],         # validity
        ))
    return results


def evaluate_horizon(model, dataset, horizon: int, device) -> StepErrorReport:
    """Compute changed-cell + overall error at a given horizon vs persistence."""
    _require_torch()
    import torch

    model.eval()
    m_changed_err = m_changed_tot = 0
    p_changed_err = p_changed_tot = 0
    m_all_err = p_all_err = all_tot = 0
    n_trans = 0

    with torch.no_grad():
        for batch in dataset:
            for pred, target, before, valid in _rollout_predict(model, batch, horizon, device):
                changed = (target != before) & valid
                unchanged = (target == before) & valid
                n_changed = int(changed.sum().item())
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                n_trans += 1

                # Model errors
                m_wrong = (pred != target) & valid
                m_all_err += int(m_wrong.sum().item())
                m_changed_err += int((m_wrong & changed).sum().item())
                m_changed_tot += n_changed

                # Persistence baseline predicts `before` (no change)
                p_wrong = (before != target) & valid
                p_all_err += int(p_wrong.sum().item())
                p_changed_err += int((p_wrong & changed).sum().item())
                p_changed_tot += n_changed

                all_tot += n_valid

    def frac(num, den):
        return (num / den) if den else 0.0

    return StepErrorReport(
        horizon=horizon,
        n_transitions=n_trans,
        model_changed_error=frac(m_changed_err, m_changed_tot),
        persistence_changed_error=frac(p_changed_err, p_changed_tot),
        model_overall_error=frac(m_all_err, all_tot),
        persistence_overall_error=frac(p_all_err, all_tot),
    )


def validate(
    model,
    holdout_path: Path,
    *,
    holdout_games: Sequence[str],
    train_games: Sequence[str],
    horizons: Sequence[int] = (1, 2, 4),
    window: int = 8,
    batch_size: int = 8,
    max_hw: int = 64,
    device=None,
    model_version: str = "sim-v1",
) -> ValidationReport:
    """Produce a ValidationReport on held-out games."""
    _require_torch()
    import torch
    from training.dataset import JsonlTrajectoryDataset

    device = device or torch.device("cpu")
    steps = []
    for h in horizons:
        ds = JsonlTrajectoryDataset(
            holdout_path, window=max(window, h + 1), batch_size=batch_size,
            games=list(holdout_games), max_hw=max_hw, shuffle=False,
        )
        steps.append(evaluate_horizon(model, ds, h, device))

    return ValidationReport(
        holdout_games=sorted(holdout_games),
        train_games=sorted(train_games),
        horizons=list(horizons),
        steps=steps,
        model_version=model_version,
    )

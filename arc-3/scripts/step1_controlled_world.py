"""Step 1 harness — prove the simulator can LEARN a small controlled world.

Milestone: trustworthy internal game. Before spending GPU-hours on the real ARC-3
games, this trains the REAL ArcSimulator (via the real training.train_simulator
path) on tiny controlled worlds whose transition function is EXACTLY known, one
mechanic at a time: movement, collision, click, delayed-consequence.

For each mechanic it reports, on HELD-OUT trajectories, at horizons k=1/2/4:
  - frame_acc        : fraction of valid cells predicted correctly (all cells)
  - changed_acc      : accuracy on cells that ACTUALLY changed vs the anchor
                       (the mechanic's effect — the thing that matters)
  - false_change_rate: fraction of truly-unchanged cells the model WRONGLY changed
                       (damage to the background — the current model's main failure)
  - baseline_persist : frame_acc of a do-nothing "predict the anchor" baseline
                       (a mechanic is only "learned" if the model beats persistence
                        on changed cells while keeping false-change low)

Honesty: multi-step predictions roll the model forward on its OWN predictions
(prior only, no posterior correction on intermediate frames), so k=2/4 are genuine
imagined futures, not teacher-forced peeks. Nothing here is submitted or scored on
the real competition; these are diagnostic worlds only.

Usage (from repo root):
  .venv\\Scripts\\python.exe scripts/step1_controlled_world.py --device cpu \\
      --mechanics movement collision click delayed --epochs 40 --out artifacts/step1_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from agent.simulator import ArcSimulator, SimulatorState
from agent.internal_world import Action
from training.controlled_worlds import generate_dataset
from training.dataset import InMemoryTrajectoryDataset
from training.train_simulator import TrainingConfig, train


ROOT = Path(__file__).resolve().parent.parent


def _grid_tuple(grid_row_major) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid_row_major)


def _rollout_predict_grids(
    model: ArcSimulator,
    traj: dict,
    horizons: tuple[int, ...],
    device: torch.device,
) -> list[dict]:
    """For each valid anchor t in the trajectory, imagine forward and record the
    predicted vs real grids at each horizon k (anchor+k), plus the anchor grid.

    Rolls forward on the model's OWN prior predictions (no posterior correction on
    intermediate frames) — genuine imagined futures.
    """
    model.eval()
    steps = traj["steps"]
    results: list[dict] = []
    max_k = max(horizons)

    for t in range(len(steps) - max_k):
        # Anchor observation at t; actions a_t..a_{t+max_k-1} lead to frames t+1..t+max_k
        anchor_grid = _grid_tuple(steps[t]["grid"])
        H, W = len(anchor_grid), len(anchor_grid[0])

        # Prime: encode anchor, update posterior once from the real anchor frame.
        sim_state = model.initial_state(game_id=traj["game_id"], device=device)
        feat, spatial, hi_res = model.encoder(*model._grid_to_tensors(anchor_grid, device))
        sim_state = model.update_posterior(feat, sim_state)

        # Roll forward on the prior using the real action sequence.
        preds_by_k: dict[int, tuple] = {}
        state = sim_state
        # change-mask prev: real anchor first, then the model's OWN prediction.
        prev_tokens = model._grid_to_tensors(anchor_grid, device)[0][0]  # (H, W)
        for k in range(1, max_k + 1):
            step = steps[t + k - 1]
            aid = step.get("action_id")
            if aid is None:
                break
            act = Action(
                id=int(aid),
                x=step.get("action_x"),
                y=step.get("action_y"),
            )
            state = model.imagine_step(act, state)
            out = model.decode(state, spatial, (H, W),
                               prev_tokens=prev_tokens.unsqueeze(0), hi_res_skip=hi_res)
            pred_grid = out["frame_logits"][0].argmax(dim=0)  # (H, W)
            prev_tokens = pred_grid.detach()
            if k in horizons:
                preds_by_k[k] = _grid_tuple(pred_grid.detach().cpu().tolist())

        for k in horizons:
            if k in preds_by_k and (t + k) < len(steps):
                results.append({
                    "anchor": anchor_grid,
                    "k": k,
                    "pred": preds_by_k[k],
                    "real": _grid_tuple(steps[t + k]["grid"]),
                })
    return results


def _score(rollouts: list[dict], horizons: tuple[int, ...]) -> dict:
    """Aggregate per-horizon frame/changed/false-change metrics."""
    report: dict[str, dict] = {}
    for k in horizons:
        subset = [r for r in rollouts if r["k"] == k]
        if not subset:
            report[f"k={k}"] = {"n": 0}
            continue
        cells_total = correct_total = 0
        changed_total = changed_correct = 0
        unchanged_total = false_changed = 0
        persist_correct = 0  # do-nothing baseline: predict anchor
        for r in subset:
            anchor, pred, real = r["anchor"], r["pred"], r["real"]
            H, W = len(real), len(real[0])
            for y in range(H):
                for x in range(W):
                    rv = real[y][x]
                    pv = pred[y][x]
                    av = anchor[y][x]
                    cells_total += 1
                    if pv == rv:
                        correct_total += 1
                    if av == rv:
                        persist_correct += 1
                    if rv != av:  # cell actually changed
                        changed_total += 1
                        if pv == rv:
                            changed_correct += 1
                    else:          # cell truly unchanged
                        unchanged_total += 1
                        if pv != av:  # model wrongly changed it
                            false_changed += 1
        report[f"k={k}"] = {
            "n": len(subset),
            "frame_acc": round(correct_total / max(1, cells_total), 4),
            "changed_acc": round(changed_correct / max(1, changed_total), 4),
            "changed_cells": changed_total,
            "false_change_rate": round(false_changed / max(1, unchanged_total), 4),
            "baseline_persist_acc": round(persist_correct / max(1, cells_total), 4),
        }
    return report


def run_mechanic(
    mechanic: str,
    *,
    device: torch.device,
    n_train: int,
    n_eval: int,
    steps: int,
    epochs: int,
    grid: int,
    seed: int,
) -> dict:
    horizons = (1, 2, 4)
    # Distinct seed spaces so train and eval trajectories never coincide.
    train_recs = generate_dataset(mechanic, n_trajectories=n_train, steps=steps,
                                  seed=seed, h=grid, w=grid)
    eval_recs = generate_dataset(mechanic, n_trajectories=n_eval, steps=steps,
                                 seed=seed + 999, h=grid, w=grid)

    window = min(steps, 8)
    dataset = InMemoryTrajectoryDataset(train_recs, window=window, batch_size=16,
                                        max_hw=grid, shuffle=True, seed=seed)

    model = ArcSimulator().to(device)
    config = TrainingConfig(
        learning_rate=1e-3,
        changed_cell_weight=5.0,
        rollout_steps=(1, 2, 4),
    )
    t0 = time.time()
    train(model, dataset, config, device, max_epochs=epochs)
    train_secs = round(time.time() - t0, 2)

    rollouts = []
    for rec in eval_recs:
        rollouts.extend(_rollout_predict_grids(model, rec, horizons, device))
    metrics = _score(rollouts, horizons)

    return {
        "mechanic": mechanic,
        "params": model.parameter_count(),
        "train_trajectories": n_train,
        "eval_trajectories": n_eval,
        "steps_per_traj": steps,
        "epochs": epochs,
        "grid": grid,
        "train_seconds": train_secs,
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1 controlled-world learning test")
    ap.add_argument("--mechanics", nargs="+",
                    default=["movement", "collision", "click", "delayed"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/step1_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    results = []
    for mech in args.mechanics:
        print(f"[step1] training mechanic: {mech} ...", flush=True)
        res = run_mechanic(
            mech, device=device, n_train=args.n_train, n_eval=args.n_eval,
            steps=args.steps, epochs=args.epochs, grid=args.grid, seed=args.seed,
        )
        results.append(res)
        for k, m in res["metrics"].items():
            print(f"    {mech} {k}: {m}", flush=True)

    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"[step1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

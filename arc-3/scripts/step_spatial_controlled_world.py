"""Validate the SPATIAL-dynamics simulator on the controlled worlds.

Trains SpatialArcSimulator (agent/spatial_simulator.py) on the same controlled
worlds as Step 1/2 and reports the same metrics (frame/changed/false-change at
k=1/2/4), plus the movement vanish/appear decomposition — the decisive test of
whether a spatial recurrent core lets the model learn WHERE an object moves TO.

Self-contained trainer (the spatial state has a different interface than
ArcSimulator, so the vector-only train() path does not apply). Reuses the real
loss functions (masked_frame_loss, kl_loss) so the comparison is honest.

Usage (from repo root):
  .venv\\Scripts\\python.exe scripts/step_spatial_controlled_world.py --device cpu \\
      --mechanics movement collision click delayed --epochs 60 \\
      --out artifacts/step7_spatial_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from agent.spatial_simulator import SpatialArcSimulator, reparameterise
from agent.internal_world import Action
from training.controlled_worlds import generate_dataset, AGENT, BG
from training.dataset import InMemoryTrajectoryDataset
from training.train_simulator import masked_frame_loss, kl_loss

ROOT = Path(__file__).resolve().parent.parent


def _kl_spatial(mu_q, logvar_q, mu_p, logvar_p):
    """KL for spatial (B, C, H, W) diag Gaussians: flatten C,H,W as the event dim."""
    B = mu_q.shape[0]
    flat = lambda t: t.reshape(B, -1)
    return kl_loss(flat(mu_q), flat(logvar_q), flat(mu_p), flat(logvar_p))


def train_spatial(model, dataset, *, device, epochs, lr=1e-3, changed_weight=5.0,
                  kl_weight=0.1, rollout=(1, 2, 4)):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        for batch in dataset:
            batch = batch.to(device)
            B, T, H, W = batch.grid_tokens.shape
            # Full-resolution recurrent state: (B, ch, H, W).
            h = torch.zeros(B, model.h_ch, H, W, device=device)
            opt.zero_grad()
            total = torch.zeros((), device=device)
            n = 0
            for t in range(T - 1):
                tok_t = batch.grid_tokens[:, t]
                mask_t = batch.validity_mask[:, t]
                obs_map = model.encoder(tok_t, mask_t)          # (B, h_ch, H, W)
                # posterior z from obs; posterior refreshes h to the current scene
                mu_q, logvar_q = model.posterior(obs_map, obs_map)
                z = reparameterise(mu_q, logvar_q)
                h = obs_map
                # action -> full-res dynamics
                amap = model._action_map_from_batch(
                    batch.action_ids[:, t], batch.action_x[:, t],
                    batch.action_y[:, t], batch.coord_valid[:, t], (H, W), device)
                x = torch.cat([z, amap], dim=1)
                h_new = model.dynamics(x, h)
                mu_p, logvar_p = model.prior(h_new, amap)
                z_next = reparameterise(mu_p, logvar_p)
                from agent.spatial_simulator import SpatialState
                st = SpatialState(h=h_new, z=z_next, game_id="t", model_version=model.VERSION)
                outputs = model._heads(st, (H, W), prev_tokens=tok_t)
                fl = masked_frame_loss(outputs["frame_logits"], batch.grid_tokens[:, t + 1],
                                       batch.validity_mask[:, t + 1],
                                       prev_tokens=tok_t, changed_weight=changed_weight)
                kl = _kl_spatial(mu_q, logvar_q, mu_p, logvar_p)
                total = total + fl["total"] + kl_weight * kl
                n += 1
                h = h_new
            (total / max(1, n)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def _grid_tuple(g):
    return tuple(tuple(int(v) for v in row) for row in g)


def rollout_eval(model, traj, horizons, device):
    model.eval()
    steps = traj["steps"]
    max_k = max(horizons)
    results = []
    for t in range(len(steps) - max_k):
        anchor = _grid_tuple(steps[t]["grid"])
        H, W = len(anchor), len(anchor[0])
        tok, mask = model._grid_to_tensors(anchor, device)
        obs_map = model.encoder(tok, mask)
        st = model.initial_state(game_id=traj["game_id"], device=device, hw=(H, W))
        st = model.update_posterior(obs_map, st)
        prev = tok[0]
        preds = {}
        for k in range(1, max_k + 1):
            step = steps[t + k - 1]
            aid = step.get("action_id")
            if aid is None:
                break
            act = Action(id=int(aid), x=step.get("action_x"), y=step.get("action_y"))
            st = model.imagine_step(act, st)
            out = model._heads(st, (H, W), prev_tokens=prev.unsqueeze(0))
            pg = out["frame_logits"][0].argmax(dim=0)
            prev = pg.detach()
            if k in horizons:
                preds[k] = _grid_tuple(pg.detach().cpu().tolist())
        for k in horizons:
            if k in preds and (t + k) < len(steps):
                results.append({"anchor": anchor, "k": k, "pred": preds[k],
                                "real": _grid_tuple(steps[t + k]["grid"])})
    return results


def score(rollouts, horizons):
    report = {}
    for k in horizons:
        subset = [r for r in rollouts if r["k"] == k]
        if not subset:
            report[f"k={k}"] = {"n": 0}
            continue
        ct = cc = cht = chc = unt = fc = pc = 0
        for r in subset:
            a, p, rl = r["anchor"], r["pred"], r["real"]
            H, W = len(rl), len(rl[0])
            for y in range(H):
                for x in range(W):
                    ct += 1
                    cc += (p[y][x] == rl[y][x])
                    pc += (a[y][x] == rl[y][x])
                    if rl[y][x] != a[y][x]:
                        cht += 1; chc += (p[y][x] == rl[y][x])
                    else:
                        unt += 1; fc += (p[y][x] != a[y][x])
        report[f"k={k}"] = {
            "n": len(subset),
            "frame_acc": round(cc / max(1, ct), 4),
            "changed_acc": round(chc / max(1, cht), 4),
            "changed_cells": cht,
            "false_change_rate": round(fc / max(1, unt), 4),
            "baseline_persist_acc": round(pc / max(1, ct), 4),
        }
    return report


def movement_decomp(model, eval_recs, device):
    model.eval()
    av = ao = pv = po = 0
    for rec in eval_recs:
        s = rec["steps"]
        for t in range(len(s) - 1):
            aid = s[t].get("action_id")
            if aid is None:
                continue
            a = _grid_tuple(s[t]["grid"]); rl = _grid_tuple(s[t + 1]["grid"])
            H, W = len(a), len(a[0])
            tok, mask = model._grid_to_tensors(a, device)
            obs_map = model.encoder(tok, mask)
            st = model.initial_state(game_id="m", device=device, hw=(H, W))
            st = model.update_posterior(obs_map, st)
            st = model.imagine_step(Action(id=int(aid)), st)
            out = model._heads(st, (H, W), prev_tokens=tok[0].unsqueeze(0))
            p = out["frame_logits"][0].argmax(dim=0).tolist()
            for y in range(H):
                for x in range(W):
                    if rl[y][x] == a[y][x]:
                        continue
                    if a[y][x] == AGENT and rl[y][x] == BG:
                        av += 1; ao += (p[y][x] == rl[y][x])
                    elif a[y][x] == BG and rl[y][x] == AGENT:
                        pv += 1; po += (p[y][x] == rl[y][x])
    return {"vanish": f"{ao}/{av}={round(ao/max(1,av),3)}",
            "appear": f"{po}/{pv}={round(po/max(1,pv),3)}"}


def run_mechanic(mech, *, device, n_train, n_eval, steps, epochs, grid, seed):
    horizons = (1, 2, 4)
    tr = generate_dataset(mech, n_trajectories=n_train, steps=steps, seed=seed, h=grid, w=grid)
    ev = generate_dataset(mech, n_trajectories=n_eval, steps=steps, seed=seed + 999, h=grid, w=grid)
    ds = InMemoryTrajectoryDataset(tr, window=min(steps, 8), batch_size=16, max_hw=grid, shuffle=True, seed=seed)
    model = SpatialArcSimulator().to(device)
    t0 = time.time()
    train_spatial(model, ds, device=device, epochs=epochs)
    secs = round(time.time() - t0, 2)
    rollouts = []
    for rec in ev:
        rollouts.extend(rollout_eval(model, rec, horizons, device))
    res = {"mechanic": mech, "params": model.parameter_count(), "epochs": epochs,
           "grid": grid, "train_seconds": secs, "metrics": score(rollouts, horizons)}
    if mech in ("movement", "collision"):
        res["decomposition"] = movement_decomp(model, ev, device)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanics", nargs="+", default=["movement", "collision", "click", "delayed"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/step7_spatial_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    results = []
    for mech in args.mechanics:
        print(f"[spatial] training mechanic: {mech} ...", flush=True)
        res = run_mechanic(mech, device=device, n_train=args.n_train, n_eval=args.n_eval,
                           steps=args.steps, epochs=args.epochs, grid=args.grid, seed=args.seed)
        results.append(res)
        for k, m in res["metrics"].items():
            print(f"    {mech} {k}: {m}", flush=True)
        if "decomposition" in res:
            print(f"    {mech} decomposition: {res['decomposition']}", flush=True)

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"[spatial] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

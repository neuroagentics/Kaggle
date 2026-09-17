"""Full simulator training launcher — R10 production run (GPU).

One command to produce the trained simulator checkpoint the agent needs:

    python scripts/train_full.py --device cuda:0 --out models/simulator

Pipeline (all pieces already unit-verified on CPU; this just runs them at scale):
  1. Collect trajectories from every MIT-licensed environment_files game
     (unless --trajectories points at an existing JSONL).
  2. Whole-game split into train / held-out (no leakage).
  3. Train ArcSimulator to convergence with early stopping.
  4. Validate on held-out games: 1/2/4-step changed-cell error vs persistence.
  5. Save the checkpoint (+ CheckpointDescriptor), the training config, the
     split, and the validation report — everything needed to reproduce and to
     point ARC3_SIMULATOR_CHECKPOINT at the result.

This script is intentionally CPU-runnable for a tiny smoke (--smoke) but a real
run wants a CUDA device. It does not download anything and reads only local,
MIT-licensed environments.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "vendor" / "ARC-AGI-3-Agents"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full ARC-3 simulator training")
    parser.add_argument("--device", default="auto",
                        help="cuda:0 / cpu / auto (default: auto)")
    parser.add_argument("--out", type=Path, default=_ROOT / "models" / "simulator",
                        help="Output dir for checkpoint + reports")
    parser.add_argument("--trajectories", type=Path, default=None,
                        help="Existing JSONL; if omitted, collect fresh")
    parser.add_argument("--games", default="", help="Comma-separated game keys; default all")
    parser.add_argument("--per-game", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-hw", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-version", default="sim-v1")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny fast run for wiring checks (small dims/epochs)")
    args = parser.parse_args(argv)

    import torch
    from agent.device import resolve_device
    from agent.simulator import ArcSimulator
    from training.collect_trajectories import discover_environments, collect
    from training.dataset import JsonlTrajectoryDataset, load_records
    from training.train_simulator import TrainingConfig, train, save_training_state
    from training.validate_simulator import split_games, validate

    resolution = resolve_device(args.device)
    device = torch.device(resolution.resolved)
    if resolution.degraded:
        print(f"[train] device degraded: {resolution.reason} -> {resolution.resolved}")
    print(f"[train] device={device}")

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Trajectories -------------------------------------------------------
    if args.trajectories and args.trajectories.is_file():
        traj_path = args.trajectories
        print(f"[train] using existing trajectories: {traj_path}")
    else:
        traj_path = args.out / "trajectories.jsonl"
        specs = discover_environments()
        if args.games:
            wanted = {g.strip() for g in args.games.split(",") if g.strip()}
            specs = [s for s in specs if s.game_key in wanted]
        per_game = 4 if args.smoke else args.per_game
        max_steps = 20 if args.smoke else args.max_steps
        print(f"[train] collecting {len(specs)} games x {per_game} trajectories ...")
        summary = collect(specs, out_path=traj_path, trajectories_per_game=per_game,
                          max_steps=max_steps, base_seed=args.seed)
        print(f"[train] collected {summary['total_trajectories']} trajectories, "
              f"{summary['total_transitions']} transitions, "
              f"{len(summary['failures'])} failures")

    # 2. Split --------------------------------------------------------------
    all_games = {r["game_id"] for r in load_records(traj_path)}
    train_games, holdout_games = split_games(
        all_games, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    print(f"[train] train games ({len(train_games)}): {train_games}")
    print(f"[train] holdout games ({len(holdout_games)}): {holdout_games}")
    if not train_games or not holdout_games:
        print("[train] ERROR: need at least 1 train and 1 holdout game", file=sys.stderr)
        return 1

    # 3. Train --------------------------------------------------------------
    if args.smoke:
        cfg = TrainingConfig(feature_dim=64, latent_dim=16, action_dim=16,
                             rollout_steps=(1, 2), multistep_weights=(1.0, 0.5),
                             max_epochs=3, learning_rate=2e-3)
    else:
        cfg = TrainingConfig(max_epochs=args.max_epochs)

    model = ArcSimulator(
        feature_dim=cfg.feature_dim, latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim, num_actions=cfg.num_actions,
        color_embed_dim=cfg.color_embed_dim,
    ).to(device)
    model.VERSION = args.model_version

    train_ds = JsonlTrajectoryDataset(
        traj_path, window=args.window, batch_size=args.batch_size,
        games=train_games, max_hw=args.max_hw, seed=args.seed,
    )
    val_ds = JsonlTrajectoryDataset(
        traj_path, window=args.window, batch_size=args.batch_size,
        games=holdout_games, max_hw=args.max_hw, shuffle=False,
    )
    print(f"[train] train windows={train_ds.num_windows()} "
          f"holdout windows={val_ds.num_windows()}")

    t0 = time.time()

    def _log(epoch, step, logs):
        print(f"[train] epoch {epoch} step {step}: total={logs['total']:.4f} "
              f"frame={logs['frame']:.3f} kl={logs['kl']:.3f}", flush=True)

    history = train(model, train_ds, cfg, device,
                    val_dataset=val_ds, max_epochs=cfg.max_epochs, on_log=_log)
    print(f"[train] done in {time.time() - t0:.1f}s; best_val={history.get('best_val')}")

    # 4. Validate -----------------------------------------------------------
    report = validate(model, traj_path, holdout_games=holdout_games,
                      train_games=train_games, horizons=(1, 2, 4),
                      window=args.window, batch_size=args.batch_size,
                      max_hw=args.max_hw, device=device, model_version=args.model_version)
    print("[train] held-out validation summary:")
    print(json.dumps(report.to_dict()["summary"], indent=2))

    # 5. Save ---------------------------------------------------------------
    ckpt_path = args.out / "simulator.pt"
    descriptor = model.save_checkpoint(ckpt_path, model_version=args.model_version)
    (args.out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out / "validation_report.json").write_text(report.to_json(), encoding="utf-8")
    (args.out / "split.json").write_text(json.dumps(
        {"train_games": train_games, "holdout_games": holdout_games,
         "holdout_fraction": args.holdout_fraction, "seed": args.seed}, indent=2), encoding="utf-8")
    (args.out / "checkpoint_descriptor.json").write_text(json.dumps({
        "model_name": descriptor.model_name,
        "model_version": descriptor.model_version,
        "role": descriptor.role,
        "parameter_count": descriptor.parameter_count,
        "quantization": descriptor.quantization,
        "device": descriptor.device,
        "artifact_hash": descriptor.artifact_hash,
    }, indent=2), encoding="utf-8")

    beats = [s.beats_persistence_on_changed for s in report.steps]
    print(f"[train] checkpoint: {ckpt_path}")
    print(f"[train] params: {descriptor.parameter_count:,} | sha256: {descriptor.artifact_hash[:16]}...")
    print(f"[train] beats persistence on changed cells (h1,h2,h4): {beats}")
    print("[train] To use: set ARC3_SIMULATOR_CHECKPOINT=" + str(ckpt_path))
    print("[train] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

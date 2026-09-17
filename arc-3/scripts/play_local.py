"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it — exactly what the Kaggle
gateway does, just in-process.

Usage:
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200
    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-action logs and print only game summaries.")
    p.add_argument("--telemetry-json", type=Path, default=None,
                   help="Write per-game summaries and bounded transition traces.")
    args = p.parse_args()

    print("RUN_MODE=local_normal: development score; not a verified competition-mode run")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # NORMAL = local execution; game source is downloaded on first call into
    # ./environment_files/ and cached for subsequent runs.
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Resolve which games to play. `arc.make()` accepts the short game id
    # (e.g. "ls20") even though the EnvironmentInfo.game_id includes the
    # version suffix ("ls20-9607627b"), so we normalize to short ids.
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; playing all {len(game_ids)} games "
              f"(this is what Kaggle does in competition rerun).\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)

    per_game = []
    telemetry_games = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
        env = arc.make(game_id, render_mode=args.render)
        if env is None:
            print(f"  could not create env for {game_id!r}, skipping")
            continue

        agent = MyAgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev"],
        )
        agent.main()

        final = agent.frames[-1]
        per_game.append((game_id, final.state, final.levels_completed,
                         agent.action_counter))
        summary = agent.telemetry_summary()
        summary.update({
            "final_state": final.state.name,
            "levels_completed": final.levels_completed,
            "framework_action_count": agent.action_counter,
        })
        telemetry_games.append({
            "summary": summary,
            "transitions": agent.transition_events(),
        })
        # Keep console output ASCII-safe on the default Windows code page.
        print(f"  -> state={final.state}, levels_completed={final.levels_completed}, "
              f"actions={agent.action_counter}")

    sc = arc.get_scorecard()
    print("\n========= SUMMARY =========")
    for gid, state, levels, actions in per_game:
        print(f"  {gid:8} levels={levels:3}  actions={actions:5}  state={state}")
    score_val = sc.score if hasattr(sc, "score") else sc
    print(f"\nAggregate scorecard score: {score_val}")

    if args.telemetry_json is not None:
        completed_levels = sum(row[2] for row in per_game)
        total_actions = sum(row[3] for row in per_game)
        report = {
            "schema_version": 1,
            "run_mode": "local_normal",
            "competition_mode_verified": False,
            "aggregate": {
                "games": len(per_game),
                "levels_completed": completed_levels,
                "actions": total_actions,
                "actions_per_level": (
                    total_actions / completed_levels if completed_levels else None
                ),
                "score": score_val,
            },
            "games": telemetry_games,
        }
        args.telemetry_json.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"Telemetry: {args.telemetry_json.resolve()}")


if __name__ == "__main__":
    main()

"""Bounded LOCAL_NORMAL integration diagnostic; never uploads or submits."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game", default="cd82")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--actions", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from scripts.run_ablations import AblationAgentFactory, ArcEnvironmentFactory, load_env_classes
    from scripts.evaluate import run_game, CONDITION_NO_DELIBERATOR
    factory = AblationAgentFactory(str(args.checkpoint), None, args.device)
    agent = factory.make(args.game, CONDITION_NO_DELIBERATOR, 0, "diagnostic-v2")
    result = run_game(agent, args.game, CONDITION_NO_DELIBERATOR, 0,
                      max_actions=args.actions, max_wall_seconds=60,
                      model_version="world-model-v0", code_hash="see artifact_hashes",
                      data_hash="local-public-environment",
                      env_factory=ArcEnvironmentFactory(load_env_classes([args.game])))
    ctrl = agent._agent._controller
    digest = hashlib.sha256()
    for folder in ("agent", "scripts", "training"):
        for path in sorted((ROOT / folder).glob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    with args.checkpoint.open("rb") as stream:
        weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "purpose": "Local runtime diagnostic, no real deliberator, not competition evidence",
        "artifact_hashes": {"source": digest.hexdigest(), "checkpoint": weights_hash},
        "result": result.to_dict(), "controller": ctrl.stats(),
        "agent": agent._agent.world_model_summary(),
        "nonzero_live_h": ctrl._sim_state is not None and bool(torch.any(ctrl._sim_state.h != 0)),
        "checkpoint_transition_version": agent._agent._simulator.checkpoint_transition_version,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("result", "agent")}, indent=2))
    print(json.dumps({k:v for k,v in report["result"].items() if k != "steps"}, indent=2))
    return int(result.crashed or result.timed_out)


if __name__ == "__main__":
    raise SystemExit(main())

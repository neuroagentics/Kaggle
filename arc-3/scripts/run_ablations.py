"""Ablation runner — R14 equal-budget component study on held-out games.

Runs the required ablation conditions (delivery plan §Measurement) against the
REAL MIT-licensed environments, with equal action/wall budgets, and writes an
EvaluationReport with paired per-condition metrics and confidence intervals.

    python scripts/run_ablations.py --sim models/simulator/simulator.pt \
        --games holdout1,holdout2 --seeds 0,1,2 --max-actions 200 --out report.json

Conditions wired here map to blueprint mechanisms:
  - full                    : trained simulator + planner + memory + deliberator
  - no_deliberator          : same, deliberator disabled (stub, no hypotheses)
  - no_imagined_planning    : planner horizon forced to 1 (no look-ahead)
  - no_memory               : working memory disabled (no within-game learning)
  - persistence_baseline    : predict-no-change agent (lower bound)
  - simple_motion_baseline  : cycle non-coordinate actions (naive control)

Honesty: each agent is a WorldModelAgent variant EXCEPT the two baselines, which
are deliberately weak controls. The report is LOCAL_NORMAL and never equated with
Kaggle scores. A condition whose agent cannot be built (e.g. no simulator) is
recorded as a crash, counted in the denominator, not hidden.

A real trained simulator (--sim) is required for the world-model conditions to be
meaningful; without it they fall back to baseline behaviour and the report says so.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "vendor" / "ARC-AGI-3-Agents"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.evaluate import (  # noqa: E402
    EvaluationManifest,
    evaluate_all,
    CONDITION_FULL,
    CONDITION_NO_DELIBERATOR,
    CONDITION_NO_PLANNING,
    CONDITION_NO_MEMORY,
    CONDITION_PERSISTENCE,
    CONDITION_SIMPLE_MOTION,
)

ABLATION_CONDITIONS = [
    CONDITION_FULL,
    CONDITION_NO_DELIBERATOR,
    CONDITION_NO_PLANNING,
    CONDITION_NO_MEMORY,
    CONDITION_PERSISTENCE,
    CONDITION_SIMPLE_MOTION,
]


# ---------------------------------------------------------------------------
# Real environment transport (stateful, reset-once / step-once)
# ---------------------------------------------------------------------------

class ArcEnvironment:
    """A single real MIT-licensed game as an evaluate.Environment.

    reset() starts the game once; step(action_id, x, y) applies EXACTLY one
    action and returns the resulting settled frame. Click coordinates from the
    agent are passed through verbatim (no (32,32) substitution). No re-reads,
    no double execution — the current frame is whatever the last call returned.
    """

    def __init__(self, game_cls):
        self._game_cls = game_cls
        self._game = None

    def _to_step(self, frame):
        from scripts.evaluate import EnvStep
        from arcengine import GameState, GameAction
        grid = [[int(v) for v in row] for row in frame.frame[-1]] if frame.frame else [[0]]
        terminal = frame.state in (GameState.WIN, GameState.GAME_OVER)
        success = frame.state is GameState.WIN
        avail = tuple(int(a) for a in frame.available_actions
                      if int(a) != GameAction.RESET.value)
        return EnvStep(grid, int(frame.levels_completed), terminal, success, avail)

    def reset(self):
        from arcengine import ActionInput, GameAction
        self._game = self._game_cls()
        return self._to_step(self._game.perform_action(ActionInput(id=GameAction.RESET)))

    def step(self, action_id, x, y):
        from arcengine import ActionInput, GameAction
        from scripts.evaluate import EnvStep
        if self._game is None:
            raise RuntimeError("Environment must be initialized before stepping")
        from agent.internal_world import Action
        Action(action_id, x, y)  # validate; never silently substitute RESET
        action = GameAction.from_id(int(action_id))
        data = {}
        if int(action_id) == GameAction.ACTION6.value:
            # Pass the agent's real coordinates through; default to centre only
            # if the agent genuinely supplied none.
            data = {"x": x, "y": y}
        frame = self._game.perform_action(ActionInput(id=action, data=data))
        return self._to_step(frame)


class ArcEnvironmentFactory:
    """evaluate.EnvironmentFactory over the discovered game classes."""

    def __init__(self, env_specs: dict):
        self._classes = env_specs

    def make(self, game_id: str):
        cls = self._classes.get(game_id)
        if cls is None:
            raise ValueError(f"Unknown evaluation game: {game_id}")
        return ArcEnvironment(cls)


# ---------------------------------------------------------------------------
# Condition-aware agent factory
# ---------------------------------------------------------------------------

class _PersistenceBaseline:
    """Weak control: always repeats the first available action (no adaptation).

    A genuine lower bound — it never varies behaviour, so any full-system gain
    over it reflects doing SOMETHING beyond a fixed action.
    """
    def __init__(self, actions):
        self._actions = list(actions) or [1]

    def decide(self, obs):
        if not obs.available_action_ids:
            raise ValueError("No legal action")
        aid = obs.available_action_ids[0]
        return _baseline_decision(aid, obs)

    def correct(self, obs_after, decision):
        pass


class _SimpleMotionBaseline:
    """Weak control: cycles through the available non-click actions in order.

    Distinct from persistence — it moves, but blindly and without any model,
    memory, or planning. Isolates 'structured motion' from 'reasoned motion'.
    """
    def __init__(self, actions):
        self._actions = [a for a in actions if a != 6] or list(actions) or [1]
        self._i = 0

    def decide(self, obs):
        avail = [a for a in obs.available_action_ids if a != 6] or list(obs.available_action_ids) or [1]
        aid = avail[self._i % len(avail)]
        self._i += 1
        return _baseline_decision(aid, obs)

    def correct(self, obs_after, decision):
        pass


def _baseline_decision(aid, obs):
    from agent.internal_world import Action
    if aid == 6:
        h, w = len(obs.grid), len(obs.grid[0]) if obs.grid else 0
        act = Action(6, min(w - 1, w // 2), min(h - 1, h // 2))
    else:
        act = Action(aid if 0 <= aid <= 7 else 1)
    return _Decision(act)


class AblationAgentFactory:
    """Builds one condition's agent, isolating EXACTLY ONE mechanism vs 'full'.

    Control structure (each ablation = full MINUS one mechanism):
      full                   : simulator + planning + memory + deliberator
      no_deliberator         : full − deliberator (planning + memory kept)
      no_imagined_planning   : full − look-ahead (deliberator + memory kept)
      no_memory              : full − within-run memory (deliberator + planning kept)
      persistence_baseline   : fixed-action control (distinct, no world model)
      simple_motion_baseline : action-cycling control (distinct, no world model)

    The prior version confounded everything by enabling the deliberator ONLY for
    'full' and using the same explorer for both baselines. Here the deliberator is
    on for every world-model condition except no_deliberator, and the baselines
    are genuinely different controls.
    """

    def __init__(self, sim_checkpoint: str | None, deliberator_path: str | None, device: str):
        self.sim_checkpoint = sim_checkpoint
        self.deliberator_path = deliberator_path
        self.device = device

    def make(self, game_id: str, condition: str, seed: int, model_version: str):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from agent.world_model_agent import (
            WorldModelAgent, SIMULATOR_CHECKPOINT_ENV, DELIBERATOR_PATH_ENV,
        )

        # Distinct weak baselines (no world model at all).
        if condition == CONDITION_PERSISTENCE:
            return _PersistenceBaseline((1, 2, 3, 4, 5, 6))
        if condition == CONDITION_SIMPLE_MOTION:
            return _SimpleMotionBaseline((1, 2, 3, 4, 5))

        # World-model conditions share identical config EXCEPT the one ablated
        # mechanism. Simulator is present for all of them.
        os.environ["ARC3_DEVICE"] = self.device
        if self.sim_checkpoint:
            os.environ[SIMULATOR_CHECKPOINT_ENV] = self.sim_checkpoint
        else:
            os.environ.pop(SIMULATOR_CHECKPOINT_ENV, None)

        # Deliberator is ON for every world-model condition EXCEPT no_deliberator,
        # so it is not a hidden confound in the planning/memory comparisons.
        if condition != CONDITION_NO_DELIBERATOR and self.deliberator_path:
            os.environ[DELIBERATOR_PATH_ENV] = self.deliberator_path
        else:
            os.environ.pop(DELIBERATOR_PATH_ENV, None)

        agent = WorldModelAgent(card_id="ablation", game_id=game_id, agent_name=condition,
                                ROOT_URL="local", record=False, arc_env=None)

        ctrl = getattr(agent, "_controller", None)
        if ctrl is None:
            raise RuntimeError("World-model condition cannot use a baseline fallback")
        if condition != CONDITION_NO_DELIBERATOR and agent._deliberator_backend == "stub":
            raise RuntimeError("Condition requires a loaded real deliberator")
        if ctrl is not None:
            if condition == CONDITION_NO_DELIBERATOR:
                ctrl.deliberator = None
            if condition == CONDITION_NO_PLANNING:
                # Disable look-ahead: horizon 1, single branch.
                ctrl.planning_enabled = False
            if condition == CONDITION_NO_MEMORY:
                # GENUINELY disable within-run memory: a no-op memory that records
                # nothing and returns nothing, so no adaptation/procedures accrue.
                ctrl.memory = _NullMemory(game_id)
        return FrameworkAgentAdapter(agent)


def _with(config, **changes):
    """Return a copy of a frozen ControllerConfig with fields replaced."""
    from dataclasses import replace
    return replace(config, **changes)


def _NullMemory(game_id):
    """A WorkingMemory that records and returns nothing — genuine 'no memory'.

    Subclasses WorkingMemory so isinstance/type checks and the controller's calls
    still work, but add_episode is a no-op and all queries are empty, so the
    no_memory ablation truly has no within-run learning (unlike the prior
    'shrink to cap=1' which still adapted within a step).
    """
    from agent.memory import WorkingMemory

    class _NM(WorkingMemory):
        def add_episode(self, **kwargs):
            return None
        def query_episodes(self, **kwargs):
            return []
        def query_beliefs(self, **kwargs):
            return []
        def query_goals(self, **kwargs):
            return []
        def query_procedures(self, **kwargs):
            return []
        def add_procedure(self, **kwargs):
            return None
        def promote_procedure_diverse(self, *a, **k):
            return None

    return _NM(game_id, episode_cap=1, mechanic_cap=1, goal_cap=1, procedure_cap=1)


class _Decision:
    """Minimal decision record exposing the fields evaluate.run_game reads."""
    __slots__ = ("action", "predicted_state", "plan_was_empty",
                 "plan_was_ambiguous", "imagined_steps", "plan_length")

    def __init__(self, action):
        self.action = action
        self.predicted_state = None
        self.plan_was_empty = False
        self.plan_was_ambiguous = False
        self.imagined_steps = 0
        self.plan_length = 0


class FrameworkAgentAdapter:
    """Adapts a framework Agent (choose_action/is_done) to evaluate.run_game's
    decide()/correct() interface, converting Observation <-> FrameData.

    evaluate.run_game speaks the Controller API (decide/correct); the framework
    agents speak choose_action(frames, latest_frame)->GameAction. This adapter
    bridges them so the same ablation harness drives both WorldModelAgent variants
    and the baseline controls without special-casing.
    """

    def __init__(self, agent):
        self._agent = agent
        self._frames: list = []

    def start_game(self, deadline: float) -> None:
        if self._agent._controller is not None:
            self._agent._controller.start_game(deadline)

    def _obs_to_frame(self, obs):
        from arcengine import FrameData, GameState
        state = GameState.WIN if obs.actual_success and obs.is_terminal else (
            GameState.GAME_OVER if obs.is_terminal else GameState.NOT_FINISHED
        )
        return FrameData(
            game_id=obs.game_id,
            frame=[[list(row) for row in obs.grid]],
            state=state,
            levels_completed=obs.level_id,
            available_actions=list(obs.available_action_ids),
        )

    @staticmethod
    def _to_internal_action(game_action):
        """Convert a framework GameAction to internal_world.Action (.id/.x/.y)."""
        from agent.internal_world import Action
        aid = int(game_action.value)
        if aid == 6:
            data = getattr(game_action, "action_data", None)
            x = int(getattr(data, "x", 32)) if data is not None else 32
            y = int(getattr(data, "y", 32)) if data is not None else 32
            return Action(6, x, y)
        # Map RESET(0)/ACTION7 etc.: Action requires 0..7; reasoning unused here.
        return Action(aid if 0 <= aid <= 7 else 0)

    def decide(self, obs) -> _Decision:
        frame = self._obs_to_frame(obs)
        game_action = self._agent.choose_action(self._frames, frame)
        self._frames.append(frame)
        # Convert framework GameAction -> internal Action exposing .id/.x/.y.
        action = self._to_internal_action(game_action)
        d = _Decision(action)
        pending = self._agent._pending_decision
        if pending is not None:
            d.predicted_state = pending.predicted_state
        reasoning = getattr(game_action, "reasoning", None)
        if isinstance(reasoning, dict):
            d.plan_was_empty = bool(reasoning.get("plan_was_empty", False))
            d.imagined_steps = int(reasoning.get("imagined_steps", 0) or 0)
            d.plan_length = int(reasoning.get("plan_length", 0) or 0)
        return d

    def correct(self, obs_after, decision) -> None:
        self._agent.observe_outcome(self._obs_to_frame(obs_after))
        self._agent.action_counter += 1


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------

def load_env_classes(game_keys: list[str]) -> dict:
    from training.collect_trajectories import discover_environments, load_game_class
    specs = {s.game_key: s for s in discover_environments()}
    classes = {}
    for key in game_keys:
        spec = specs.get(key)
        if spec is None:
            print(f"[ablation] WARNING: game {key!r} not found", file=sys.stderr)
            continue
        classes[key] = load_game_class(spec)
    return classes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R14 ablation study on held-out games")
    parser.add_argument("--sim", default=None, help="Path to trained simulator checkpoint")
    parser.add_argument("--deliberator", default=None, help="Path to HF deliberator dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games", required=True, help="Comma-separated held-out game keys")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (>=3)")
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=_ROOT / "artifacts" / "ablation_report.json")
    parser.add_argument("--split", type=Path, required=True,
                        help="Reviewed JSON with train_games and holdout_games")
    args = parser.parse_args(argv)

    game_keys = [g.strip() for g in args.games.split(",") if g.strip()]
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if (set(split["train_games"]) & set(split["holdout_games"])
            or not set(game_keys).issubset(split["holdout_games"])):
        raise ValueError("Evaluation games must be disjoint held-out games from reviewed split")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 3:
        print("[ablation] ERROR: at least 3 seeds required", file=sys.stderr)
        return 1

    manifest = EvaluationManifest(
        manifest_id="ablation-v1",
        game_ids=game_keys,
        holdout_game_ids=game_keys,
        seeds=seeds,
        conditions=ABLATION_CONDITIONS,
        max_actions_per_game=args.max_actions,
        max_wall_seconds_per_game=args.max_wall_seconds,
    )

    env_classes = load_env_classes(game_keys)
    env_factory = ArcEnvironmentFactory(env_classes)
    factory = AblationAgentFactory(args.sim, args.deliberator, args.device)
    import hashlib
    def file_hash(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    code = hashlib.sha256()
    for folder in ("agent", "scripts", "training"):
        for path in sorted((_ROOT / folder).glob("*.py")):
            code.update(str(path.relative_to(_ROOT)).encode())
            code.update(path.read_bytes())
    artifacts = {"split": file_hash(args.split)}
    if args.sim:
        artifacts["simulator"] = file_hash(args.sim)
    if args.deliberator:
        for path in sorted(Path(args.deliberator).glob("*")):
            if path.is_file() and path.suffix in (".json", ".safetensors"):
                artifacts["deliberator/" + path.name] = file_hash(path)
    for key, cls in env_classes.items():
        artifacts["environment/" + key] = file_hash(inspect.getfile(cls))

    if not args.sim:
        print("[ablation] NOTE: no --sim checkpoint; world-model conditions run in "
              "baseline-fallback mode and the report will reflect that.")

    def _progress(msg):
        print(f"[ablation] {msg}", flush=True)

    report = evaluate_all(
        manifest, factory,
        model_version="world-model-v0",
        code_hash=code.hexdigest(), data_hash=artifacts["split"],
        env_factory=env_factory, progress_callback=_progress,
    )
    report.artifact_hashes = artifacts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_json(), encoding="utf-8")
    print(f"\n[ablation] report -> {args.out}")
    for ab in report.ablations:
        s = ab.summary()
        print(f"  {s['condition']:24s} completion={s['completion_rate']:.2f} "
              f"levels={s['mean_levels_completed']:.2f} crashes={s['crash_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

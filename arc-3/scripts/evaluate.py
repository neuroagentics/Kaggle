"""Per-game evaluation and ablation framework — G7 prerequisite.

Produces the measurement report required by ARC3_LEAN_DELIVERY_PLAN.md §Measurement:
  - Per-game outcomes (completed levels, actions, latency, crashes)
  - Aggregate statistics with confidence intervals
  - Ablation conditions (no deliberator, no imagined planning, no memory,
    no procedural chaining, persistence baseline, simple-motion baseline)
  - Raw report in JSON (never summarised away from failures)

Design constraints:
  - Evaluation manifest frozen before tuning; held-out games never used for
    parameter selection.
  - Equal wall-clock and action budgets for all conditions.
  - At least 3 fixed seeds where stochastic behaviour matters.
  - Local NORMAL-mode scores are explicitly labelled; not equated with Kaggle scores.
  - No percent-complete claims based on pass counts.
  - All crashes and timeouts counted in the denominator.

Usage:
    python scripts/evaluate.py --manifest eval_manifest.json --output report.json

The script does not run inference directly; it expects a callable agent factory
(supplied via --agent-module) that returns a Controller-compatible object with
decide() and correct() methods.  This decouples evaluation logic from model loading.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Outcome of one decide/correct step."""
    step: int
    action_id: int
    action_x: int | None
    action_y: int | None
    level_before: int
    level_after: int
    level_delta: int
    latency_ms: float
    plan_was_empty: bool
    plan_was_ambiguous: bool
    imagined_steps: int
    discrepancy_score: float   # prediction error fraction; -1 if no prediction
    crashed: bool
    crash_detail: str


@dataclass
class GameResult:
    """Outcome of one complete game run under one ablation condition."""
    run_id: str
    game_id: str
    condition: str             # e.g. 'full', 'no_deliberator', 'persistence_baseline'
    seed: int
    mode: str                  # 'LOCAL_NORMAL' or 'COMPETITION'
    model_version: str
    code_hash: str
    data_hash: str

    # Outcome
    levels_completed: int
    total_actions: int
    total_steps: int
    crashed: bool
    timed_out: bool
    crash_detail: str

    # Timing
    elapsed_seconds: float
    peak_latency_ms: float
    mean_latency_ms: float

    # Prediction quality
    mean_discrepancy: float    # average prediction error over non-crash steps
    steps_with_prediction: int

    # Planning
    total_imagined_steps: int
    probe_decisions: int       # steps where plan was empty → probe used

    # Step trace (bounded; omit if too large for report)
    steps: list[StepResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class AblationReport:
    """Comparison of one ablation condition vs the full system on the same games."""
    condition: str
    game_ids: list[str]
    seeds: list[int]
    results: list[GameResult]

    @property
    def completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.levels_completed > 0 for r in self.results) / len(self.results)

    @property
    def mean_levels_completed(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.levels_completed for r in self.results) / len(self.results)

    @property
    def mean_actions(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.total_actions for r in self.results) / len(self.results)

    @property
    def crash_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.crashed for r in self.results) / len(self.results)

    def confidence_interval(self, values: list[float], confidence: float = 0.95) -> tuple[float, float]:
        """Bootstrap mean CI (simple normal approximation for n >= 3)."""
        n = len(values)
        if n == 0:
            return (0.0, 0.0)
        mean = sum(values) / n
        if n == 1:
            return (mean, mean)
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        # 95% CI: z=1.96; 99%: z=2.576
        z = 1.96 if confidence == 0.95 else 2.576
        margin = z * std / math.sqrt(n)
        return (mean - margin, mean + margin)

    def summary(self) -> dict:
        levels = [r.levels_completed for r in self.results]
        actions = [r.total_actions for r in self.results]
        disc = [r.mean_discrepancy for r in self.results if r.steps_with_prediction > 0]
        return {
            "condition": self.condition,
            "n_runs": len(self.results),
            "completion_rate": self.completion_rate,
            "mean_levels_completed": self.mean_levels_completed,
            "levels_ci_95": self.confidence_interval(levels),
            "mean_actions": self.mean_actions,
            "actions_ci_95": self.confidence_interval(actions),
            "crash_rate": self.crash_rate,
            "mean_discrepancy": sum(disc) / len(disc) if disc else None,
            "note": "LOCAL_NORMAL scores; not equivalent to Kaggle scores",
        }


@dataclass
class EvaluationReport:
    """Top-level evaluation report.

    Includes all conditions, raw game results, and aggregate summaries.
    Never hides failures or timeouts.
    """
    report_id: str
    timestamp: str
    evaluation_manifest_hash: str
    mode: str               # 'LOCAL_NORMAL'
    conditions: list[str]
    seeds: list[int]
    game_ids: list[str]
    ablations: list[AblationReport]
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "artifact_hashes": self.artifact_hashes,
            "timestamp": self.timestamp,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "mode": self.mode,
            "conditions": self.conditions,
            "seeds": self.seeds,
            "game_ids": self.game_ids,
            "summaries": [a.summary() for a in self.ablations],
            "raw_results": [
                r.to_dict()
                for ablation in self.ablations
                for r in ablation.results
            ],
            "note": (
                "LOCAL_NORMAL mode. Scores are not official Kaggle scores. "
                "All crashes and timeouts are counted in the denominator."
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Evaluation manifest
# ---------------------------------------------------------------------------

@dataclass
class EvaluationManifest:
    """Frozen evaluation specification.

    Must be committed before any tuning begins.  The manifest hash is recorded
    in every report so results can be traced back to the exact evaluation setup.
    """
    manifest_id: str
    game_ids: list[str]              # Games to evaluate (no holdout leakage)
    holdout_game_ids: list[str]      # Must not appear in game_ids
    seeds: list[int]                 # Fixed seeds; at least 3
    conditions: list[str]            # Ablation condition names
    max_actions_per_game: int        # Equal budget for all conditions
    max_wall_seconds_per_game: float
    mode: str = "LOCAL_NORMAL"
    schema_version: str = "1.0"

    _REQUIRED_CONDITIONS = frozenset({
        "full",
        "no_deliberator",
        "no_imagined_planning",
        "no_memory",
        "persistence_baseline",
        "simple_motion_baseline",
    })

    def __post_init__(self):
        if not self.game_ids:
            raise ValueError("EvaluationManifest game_ids must not be empty")
        if set(self.game_ids) & set(self.holdout_game_ids):
            raise ValueError("game_ids and holdout_game_ids must be disjoint")
        if len(self.seeds) < 3:
            raise ValueError("At least 3 seeds required for stochastic evaluation")
        if self.max_actions_per_game < 1:
            raise ValueError("max_actions_per_game must be positive")
        if self.max_wall_seconds_per_game <= 0:
            raise ValueError("max_wall_seconds_per_game must be positive")
        missing = self._REQUIRED_CONDITIONS - set(self.conditions)
        if missing:
            raise ValueError(
                f"EvaluationManifest missing required ablation conditions: {sorted(missing)}"
            )

    def hash(self) -> str:
        """Deterministic SHA-256 of the manifest content."""
        import hashlib
        content = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "EvaluationManifest":
        d = json.loads(text)
        d.pop("schema_version", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationManifest":
        return cls.from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Agent factory protocol
# ---------------------------------------------------------------------------

class AgentFactory(Protocol):
    """Contract for plugging agent implementations into the evaluator.

    The factory receives condition and seed and returns an agent-like object
    that exposes decide() → DecisionRecord and correct() compatible methods.
    """
    def make(
        self,
        game_id: str,
        condition: str,
        seed: int,
        model_version: str,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Environment transport contract
# ---------------------------------------------------------------------------

@dataclass
class EnvStep:
    """Result of one environment reset or step.

    grid              : settled grid (list[list[int]])
    level_id          : levels completed so far
    terminal          : True if WIN or GAME_OVER
    success           : True only on WIN
    available_actions : action IDs the environment actually offers (excl. RESET);
                        defaults to all of 1..6 when the env does not report them.
    """
    grid: list
    level_id: int
    terminal: bool
    success: bool
    available_actions: tuple = (1, 2, 3, 4, 5, 6)


class Environment(Protocol):
    """Stateful environment transport.

    A correct environment is reset ONCE per game, then stepped EXACTLY ONCE per
    decision. `step` applies the action and returns the resulting frame. There is
    no separate "read current frame" call — the current frame is whatever the last
    reset/step returned. This prevents the earlier defects: RESET-on-read,
    double action execution, and coordinate replacement.
    """
    def reset(self) -> EnvStep: ...
    def step(self, action_id: int, x: int | None, y: int | None) -> EnvStep: ...


class EnvironmentFactory(Protocol):
    """Returns a fresh Environment for a given game_id (one per game run)."""
    def make(self, game_id: str) -> Environment: ...


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

# Canonical condition names used in reports and comparisons
CONDITION_FULL = "full"
CONDITION_NO_DELIBERATOR = "no_deliberator"
CONDITION_NO_PLANNING = "no_imagined_planning"
CONDITION_NO_MEMORY = "no_memory"
CONDITION_NO_CHAINING = "no_procedural_chaining"
CONDITION_PERSISTENCE = "persistence_baseline"
CONDITION_SIMPLE_MOTION = "simple_motion_baseline"

ALL_CONDITIONS = [
    CONDITION_FULL,
    CONDITION_NO_DELIBERATOR,
    CONDITION_NO_PLANNING,
    CONDITION_NO_MEMORY,
    CONDITION_PERSISTENCE,
    CONDITION_SIMPLE_MOTION,
]


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_game(
    agent,
    game_id: str,
    condition: str,
    seed: int,
    *,
    max_actions: int,
    max_wall_seconds: float,
    model_version: str,
    code_hash: str,
    data_hash: str,
    env_factory: "EnvironmentFactory | None" = None,
    started_at: float | None = None,
) -> GameResult:
    """Run one game under one condition and return a GameResult.

    Transport contract (fixes prior defects): the environment is reset ONCE, then
    stepped EXACTLY ONCE per decision with the decision's real action (including
    its real click coordinates). Prediction error is computed from that single
    real next frame — the environment is never re-read or double-stepped.

    Parameters
    ----------
    agent
        Object with decide(obs) -> DecisionRecord and correct(obs, decision) methods.
    game_id : str
    condition : str
        Ablation condition label.
    seed : int
    max_actions : int
        Hard action budget (equal for all conditions).
    max_wall_seconds : float
        Wall-clock budget per game.
    model_version : str
    code_hash : str
    data_hash : str
    env_factory : EnvironmentFactory | None
        Returns a fresh stateful Environment for the game. When None, a trivial
        no-op environment is used so the runner contract can be tested without a
        real game.

    Returns
    -------
    GameResult
        All fields populated; never raises.  Crashes are recorded in the result.
    """
    run_id = uuid.uuid4().hex
    t_start = time.monotonic() if started_at is None else started_at
    t_deadline = t_start + max_wall_seconds

    steps: list[StepResult] = []
    levels_completed = 0
    total_actions = 0
    crashed = False
    timed_out = False
    crash_detail = ""
    latencies: list[float] = []
    discrepancies: list[float] = []
    total_imagined = 0
    probe_decisions = 0
    current_level = 0

    # Default trivial environment: 1×1 grid, never terminal, no progress.
    class _NoopEnv:
        def reset(self):
            return EnvStep([[0]], 0, False, False)
        def step(self, action_id, x, y):
            return EnvStep([[0]], 0, False, False)

    class _NoopFactory:
        def make(self, gid):
            return _NoopEnv()

    factory = env_factory if env_factory is not None else _NoopFactory()

    try:
        if hasattr(agent, "start_game"):
            agent.start_game(t_deadline)

        from agent.controller import Observation  # local import; evaluator is script-level

        env = factory.make(game_id)
        cur = env.reset()          # reset ONCE per game
        step_idx = 0

        while total_actions < max_actions and time.monotonic() < t_deadline:
            t_step = time.monotonic()

            if cur.terminal:
                break

            obs = Observation(
                observation_id=f"{run_id}-{step_idx:06d}",
                game_id=game_id,
                level_id=cur.level_id,
                grid=tuple(tuple(row) for row in cur.grid),
                available_action_ids=tuple(cur.available_actions),
                is_terminal=cur.terminal,
                actual_success=cur.success,
                model_version=model_version,
            )

            # Decide
            try:
                decision = agent.decide(obs)
            except Exception as exc:
                crashed = True
                crash_detail = f"decide() raised at step {step_idx}: {exc}\n{traceback.format_exc()}"
                break

            latency_ms = (time.monotonic() - t_step) * 1000.0
            latencies.append(latency_ms)
            if time.monotonic() >= t_deadline:
                timed_out = True
                break
            if decision.action.id not in cur.available_actions:
                raise ValueError("Agent selected an unavailable action")

            # Step the environment EXACTLY ONCE with the decision's real action
            # (including its real click coordinates). This single next frame is
            # both the correction target and the prediction-error reference.
            try:
                total_actions += 1  # count attempted dispatch even on transport/correction failure
                nxt = env.step(decision.action.id, decision.action.x, decision.action.y)
            except Exception as exc:
                crashed = True
                crash_detail = f"env.step() raised at step {step_idx}: {exc}"
                break

            # Prediction error against the SINGLE real next frame. Report overall
            # error (incl. damage to unchanged cells) — honest, not changed-only.
            pred_grid = getattr(getattr(decision, "predicted_state", None), "grid", None)
            discrepancy = -1.0
            if pred_grid is not None:
                shape_matches = len(pred_grid) == len(nxt.grid) and all(
                    len(pg) == len(ng) for pg, ng in zip(pred_grid, nxt.grid)
                )
                total_cells = max(1, sum(len(r) for r in nxt.grid))
                errors = sum(a != b for pg, ng in zip(pred_grid, nxt.grid)
                             for a, b in zip(pg, ng))
                discrepancy = errors / total_cells if shape_matches else 1.0
                discrepancies.append(discrepancy)

            level_delta = max(0, nxt.level_id - cur.level_id)
            levels_completed += level_delta
            # Correct
            try:
                obs_after = Observation(
                    observation_id=f"{run_id}-{step_idx + 1:06d}",
                    game_id=game_id,
                    level_id=nxt.level_id,
                    grid=tuple(tuple(row) for row in nxt.grid),
                    available_action_ids=tuple(nxt.available_actions),
                    is_terminal=nxt.terminal,
                    actual_success=(nxt.success or level_delta > 0),
                    model_version=model_version,
                )
                agent.correct(obs_after, decision)
            except Exception as exc:
                crashed = True
                crash_detail = f"correct() raised at step {step_idx}: {exc}"
                break

            total_imagined += getattr(decision, "imagined_steps", 0)
            probe_decisions += int(getattr(decision, "plan_was_empty", False))

            steps.append(StepResult(
                step=step_idx,
                action_id=decision.action.id,
                action_x=decision.action.x,
                action_y=decision.action.y,
                level_before=cur.level_id,
                level_after=nxt.level_id,
                level_delta=level_delta,
                latency_ms=latency_ms,
                plan_was_empty=getattr(decision, "plan_was_empty", False),
                plan_was_ambiguous=getattr(decision, "plan_was_ambiguous", False),
                imagined_steps=getattr(decision, "imagined_steps", 0),
                discrepancy_score=discrepancy,
                crashed=False,
                crash_detail="",
            ))

            cur = nxt              # advance; no re-read
            step_idx += 1

            if nxt.terminal:
                break
            if time.monotonic() >= t_deadline:
                timed_out = True
                break

    except Exception as exc:
        crashed = True
        crash_detail = f"Unhandled exception: {exc}\n{traceback.format_exc()}"

    elapsed = time.monotonic() - t_start
    timed_out = timed_out or time.monotonic() >= t_deadline
    peak_lat = max(latencies) if latencies else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    mean_disc = sum(discrepancies) / len(discrepancies) if discrepancies else 0.0

    return GameResult(
        run_id=run_id,
        game_id=game_id,
        condition=condition,
        seed=seed,
        mode="LOCAL_NORMAL",
        model_version=model_version,
        code_hash=code_hash,
        data_hash=data_hash,
        levels_completed=levels_completed,
        total_actions=total_actions,
        total_steps=len(steps),
        crashed=crashed,
        timed_out=timed_out,
        crash_detail=crash_detail,
        elapsed_seconds=elapsed,
        peak_latency_ms=peak_lat,
        mean_latency_ms=mean_lat,
        mean_discrepancy=mean_disc,
        steps_with_prediction=len(discrepancies),
        total_imagined_steps=total_imagined,
        probe_decisions=probe_decisions,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

def evaluate_all(
    manifest: EvaluationManifest,
    factory: AgentFactory,
    *,
    model_version: str,
    code_hash: str,
    data_hash: str,
    env_factory: "EnvironmentFactory | None" = None,
    progress_callback: Callable[[str], None] | None = None,
) -> EvaluationReport:
    """Run all conditions × games × seeds and return an EvaluationReport.

    Crashes in one run do not stop others.  All runs are counted.
    """
    ablation_map: dict[str, list[GameResult]] = {c: [] for c in manifest.conditions}

    for condition in manifest.conditions:
        for game_id in manifest.game_ids:
            for seed in manifest.seeds:
                if progress_callback:
                    progress_callback(f"condition={condition} game={game_id} seed={seed}")
                try:
                    started_at = time.monotonic()
                    agent = factory.make(game_id, condition, seed, model_version)
                except Exception as exc:
                    # Agent construction failed — record as a crash
                    result = GameResult(
                        run_id=uuid.uuid4().hex,
                        game_id=game_id,
                        condition=condition,
                        seed=seed,
                        mode="LOCAL_NORMAL",
                        model_version=model_version,
                        code_hash=code_hash,
                        data_hash=data_hash,
                        levels_completed=0,
                        total_actions=0,
                        total_steps=0,
                        crashed=True,
                        timed_out=False,
                        crash_detail=f"factory.make() failed: {exc}",
                        elapsed_seconds=time.monotonic() - started_at,
                        peak_latency_ms=0.0,
                        mean_latency_ms=0.0,
                        mean_discrepancy=0.0,
                        steps_with_prediction=0,
                        total_imagined_steps=0,
                        probe_decisions=0,
                    )
                    ablation_map[condition].append(result)
                    continue

                result = run_game(
                    agent,
                    game_id,
                    condition,
                    seed,
                    max_actions=manifest.max_actions_per_game,
                    max_wall_seconds=manifest.max_wall_seconds_per_game,
                    model_version=model_version,
                    code_hash=code_hash,
                    data_hash=data_hash,
                    env_factory=env_factory,
                    started_at=started_at,
                )
                ablation_map[condition].append(result)

    ablations = [
        AblationReport(
            condition=condition,
            game_ids=manifest.game_ids,
            seeds=manifest.seeds,
            results=ablation_map[condition],
        )
        for condition in manifest.conditions
    ]

    return EvaluationReport(
        report_id=uuid.uuid4().hex,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evaluation_manifest_hash=manifest.hash(),
        mode="LOCAL_NORMAL",
        conditions=manifest.conditions,
        seeds=manifest.seeds,
        game_ids=manifest.game_ids,
        ablations=ablations,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ARC-3 per-game evaluation and ablation report"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to evaluation manifest JSON",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for evaluation report JSON",
    )
    parser.add_argument(
        "--model-version", default="sim-v1",
        help="Model version label (default: sim-v1)",
    )
    parser.add_argument(
        "--code-hash", default="unknown",
        help="SHA-256 of the agent code (for traceability)",
    )
    parser.add_argument(
        "--data-hash", default="unknown",
        help="SHA-256 of the data split (for traceability)",
    )
    parser.add_argument(
        "--agent-module", default=None,
        help="Python module path exposing an AgentFactory as 'factory'",
    )
    args = parser.parse_args(argv)

    try:
        manifest = EvaluationManifest.from_file(args.manifest)
    except Exception as exc:
        print(f"ERROR: cannot load manifest: {exc}", file=sys.stderr)
        return 1

    # Load agent factory from module if specified
    if args.agent_module:
        import importlib
        try:
            mod = importlib.import_module(args.agent_module)
            factory = mod.factory
        except Exception as exc:
            print(f"ERROR: cannot load agent module: {exc}", file=sys.stderr)
            return 1
    else:
        # Stub factory for dry-run / CI
        class _StubFactory:
            def make(self, game_id, condition, seed, model_version):
                from agent.controller import Controller, ControllerConfig
                from agent.memory import WorkingMemory
                from agent.internal_world import Action, WorldState, Prediction

                class _TrivialDynamics:
                    def predict(self, state, action):
                        return Prediction(
                            replace(state, imagined=True),
                            utility=0.0, uncertainty=0.5, risk=0.0,
                        )

                from dataclasses import replace
                cfg = ControllerConfig(game_id=game_id, model_version=model_version)
                mem = WorkingMemory(game_id)
                ctrl = Controller(cfg, _TrivialDynamics(), None, mem)
                return ctrl

        factory = _StubFactory()

    def _progress(msg: str) -> None:
        print(f"  {msg}", flush=True)

    print("Running evaluation …")
    report = evaluate_all(
        manifest,
        factory,
        model_version=args.model_version,
        code_hash=args.code_hash,
        data_hash=args.data_hash,
        progress_callback=_progress,
    )

    args.output.write_text(report.to_json(), encoding="utf-8")
    print(f"\nReport written to {args.output}")
    for ablation in report.ablations:
        s = ablation.summary()
        print(
            f"  {s['condition']:30s}  "
            f"completion={s['completion_rate']:.2f}  "
            f"levels={s['mean_levels_completed']:.2f}  "
            f"crashes={s['crash_rate']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

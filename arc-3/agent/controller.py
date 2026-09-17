"""MRLA controller — G5 wiring layer.

Implements the single Measure → Reason → Look-ahead → Act loop that connects
every component built so far into one decision cycle:

    Real frame/status
        └─ observe()        encode observation, update posterior, record episode
        └─ reason()         call deliberator when warranted → SimulatorCondition
        └─ look_ahead()     GoalDirectedPlanner beam search through imagined states
        └─ act()            validate + dispatch first action of plan
        └─ correct()        compare prediction to outcome, update memory/beliefs

Design constraints (blueprint §3, §6):
  - ONE controller instance per game; new instance for each new game (isolation).
  - Controller never holds a reference to the official environment; it receives
    observations and returns Action objects.  Real dispatch is the caller's job.
  - Weights are frozen during look-ahead; never during correct().
  - Deadline budget: ≥10 % of overall run reserved for startup/recovery/final;
    deliberation capped at 20 % of measured inference time; adaptation at 10 %.
  - Hard backend cancellation of a hung model call remains an open requirement
    (R04); the deadline check after every real action is the current mitigation.
  - No ARC-2 solver, no graph traversal, no action-frequency fallback.
    If the plan is empty (calibration requested) the controller probes with
    an uncertainty-directed action, not a random button cycle.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from typing import Callable, Sequence

from agent.internal_world import (
    Action,
    EpisodeMemory,
    GoalDirectedPlan,
    GoalDirectedPlanner,
    GoalEvaluator,
    GoalSpec,
    InternalGame,
    Plan,
    RecursivePlanner,
    WorldState,
)
from agent.memory import WorkingMemory
from agent.model import (
    GoalSignal,
    SimulatorCondition,
    validate_prediction_against_condition,
)
from agent.deliberator import (
    DeliberatorInput,
    DeliberatorInterface,
    DeliberatorOutput,
)


# ---------------------------------------------------------------------------
# Controller configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerConfig:
    """Run-time constants for one controller instance.

    All values are the blueprint §6 defaults; measure actual tradeoffs before
    changing caps and record justification.
    """
    game_id: str
    model_version: str

    # Planning
    horizon: int = 4
    beam: int = 8
    max_predictions: int = 128
    uncertainty_limit: float = 0.35
    risk_weight: float = 4.0
    action_cost: float = 0.01

    # Time budget fractions of the overall run deadline
    deliberation_fraction: float = 0.20   # max share of inference time for deliberator
    adaptation_fraction: float = 0.10    # max share for adapter updates
    deadline_reserve_fraction: float = 0.10  # keep at least this fraction for cleanup

    # Deliberator call frequency (blueprint §6)
    stall_threshold: int = 8   # actions with zero level progress → stalled

    def __post_init__(self):
        if not self.game_id:
            raise ValueError("ControllerConfig game_id must not be empty")
        if not self.model_version:
            raise ValueError("ControllerConfig model_version must not be empty")
        if any(type(v) is not int or v < 1 for v in (self.horizon, self.beam, self.max_predictions)):
            raise ValueError("Planning budgets must be positive integers")
        for name, val in (
            ("uncertainty_limit", self.uncertainty_limit),
            ("deliberation_fraction", self.deliberation_fraction),
            ("adaptation_fraction", self.adaptation_fraction),
            ("deadline_reserve_fraction", self.deadline_reserve_fraction),
        ):
            if not isfinite(val) or not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if any(not isfinite(v) or v < 0 for v in (self.risk_weight, self.action_cost)):
            raise ValueError("risk_weight and action_cost must be finite and nonneg")


# ---------------------------------------------------------------------------
# Decision record — one full MRLA cycle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionRecord:
    """Immutable record of one complete MRLA cycle.

    Saved before the action is dispatched so the prediction is always
    available for correct() even if the model state changes.
    """
    decision_id: str
    game_id: str
    level_id: int
    observation_id: str
    model_version: str

    # The action selected
    action: Action

    # The prediction saved before dispatch
    predicted_state: WorldState | None   # None when plan was empty (probe)

    # Plan metadata
    plan_value: float
    plan_length: int
    plan_was_empty: bool         # True → calibration/probe requested
    plan_was_ambiguous: bool     # True → competing goals disagreed

    # Deliberator call metadata
    deliberator_called: bool
    hypotheses_count: int
    goals_count: int

    # Budget consumed
    imagined_steps: int
    wall_seconds: float


# ---------------------------------------------------------------------------
# Observation bundle — caller supplies this each step
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One real frame observation from the official environment.

    The controller never touches the official environment directly.
    Callers (my_agent.py integration or test harnesses) construct this.
    """
    observation_id: str          # Unique ID for this frame; used for evidence lineage
    game_id: str
    level_id: int
    grid: tuple[tuple[int, ...], ...]
    available_action_ids: tuple[int, ...]
    is_terminal: bool
    actual_success: bool         # Whether the *previous* action succeeded (level complete)
    model_version: str

    def __post_init__(self):
        if not self.observation_id:
            raise ValueError("Observation observation_id must not be empty")
        if not self.game_id:
            raise ValueError("Observation game_id must not be empty")
        if not isinstance(self.level_id, int) or self.level_id < 0:
            raise ValueError("Observation level_id must be a nonneg int")
        if not self.grid:
            raise ValueError("Observation grid must not be empty")
        if type(self.actual_success) is not bool:
            raise ValueError("actual_success must be a real bool")
        if not self.model_version:
            raise ValueError("Observation model_version must not be empty")

    def to_actions(self) -> tuple[Action, ...]:
        """Preserve API availability; expand clicks into bounded candidate points.

        Candidates cover the visible grid and each observed color. They are
        probes, not assumptions about which coordinates have an effect.
        """
        actions = []
        for aid in dict.fromkeys(self.available_action_ids):
            if aid == 6:
                height, width = len(self.grid), len(self.grid[0])
                points = {(min(width - 1, (2*x+1)*width//8),
                           min(height - 1, (2*y+1)*height//8))
                          for y in range(4) for x in range(4)}
                colors = set()
                for y, row in enumerate(self.grid):
                    for x, color in enumerate(row):
                        if color not in colors:
                            colors.add(color)
                            points.add((x, y))
                actions.extend(Action(6, x, y) for x, y in sorted(points))
                continue
            try:
                actions.append(Action(aid))
            except ValueError:
                pass  # Skip invalid IDs; controller uses what the environment reports
        return tuple(actions)


# ---------------------------------------------------------------------------
# Conditioned-dynamics adapter
# ---------------------------------------------------------------------------

class _ConditionedDynamics:
    """Presents a Dynamics.predict() that routes through predict_conditioned().

    Lets the existing planner (which calls dynamics.predict) drive the
    hypothesis-conditioned pathway without modifying InternalGame or the planner.
    Forwards freeze/unfreeze so weight-freezing during planning still works.
    """

    def __init__(self, dynamics, condition):
        self._dynamics = dynamics
        self._condition = condition

    def predict(self, state, action):
        return self._dynamics.predict_conditioned(state, action, self._condition)

    def freeze(self):
        f = getattr(self._dynamics, "freeze", None)
        if f:
            f()

    def unfreeze(self):
        u = getattr(self._dynamics, "unfreeze", None)
        if u:
            u()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:
    """One-game MRLA loop.

    Usage
    -----
    ctrl = Controller(config, dynamics, deliberator, memory)
    ctrl.start_game(run_deadline)          # set the per-run wall-clock deadline
    for each real frame:
        obs = Observation(...)
        decision = ctrl.decide(obs)        # MRLA cycle → DecisionRecord
        real_action = decision.action      # dispatch this
        ctrl.correct(obs_after, decision)  # record outcome
    """

    def __init__(
        self,
        config: ControllerConfig,
        dynamics,                          # Dynamics or HypothesisConditionedDynamics
        deliberator: DeliberatorInterface | None,
        memory: WorkingMemory,
        clock: Callable[[], float] = monotonic,
    ):
        if not isinstance(config, ControllerConfig):
            raise TypeError("config must be a ControllerConfig")
        if not isinstance(memory, WorkingMemory):
            raise TypeError("memory must be a WorkingMemory")
        if memory.game_id != config.game_id:
            raise ValueError(
                f"Memory game_id {memory.game_id!r} != config game_id {config.game_id!r}"
            )

        self.config = config
        self.dynamics = dynamics
        self.deliberator = deliberator
        self.memory = memory
        self._clock = clock

        # Run-level state
        self._run_deadline: float = float("inf")
        self._actions_since_deliberator: int = config.stall_threshold  # allow first call

        # Step-level state
        self._last_world_state: WorldState | None = None
        self._last_decision: DecisionRecord | None = None
        self._current_condition: SimulatorCondition | None = None
        self._current_goals: tuple[GoalSpec, ...] = ()
        self._level_id: int = 0
        self._actions_this_level: int = 0
        self._stall_counter: int = 0
        # Persistent recurrent internal-world state (h, z). Maintained across
        # real steps and CORRECTED from each observation via the posterior, so
        # planning advances a genuinely observed internal world rather than a
        # state rebuilt from scratch each decision.
        self._sim_state = None
        self.planning_enabled = True
        self._root_predictions = {}
        self._prediction_deadline = float("inf")
        self._corrected_decisions: set[str] = set()
        self._last_measured_observation: str | None = None
        # Optional online-learning hooks, connected by the agent when a trainable
        # simulator + replay source exist. Left None for pure planning/eval.
        self._adapter_manager = None    # training.adapter.AdapterManager | None
        self._adapter_hook = None       # Callable[[AdapterManager], None] | None
        self._used_procedure_id = None
        self._procedure_reuses = 0
        # Track supporting episode IDs of the most recent successful transitions,
        # so a promotion can cite real evidence spanning distinct contexts.
        self._recent_success_ids: list[tuple[int, str]] = []  # (level_id, episode_id)

        # Diagnostics
        self._total_decisions: int = 0
        self._total_deliberator_calls: int = 0
        self._total_imagined_steps: int = 0
        self._total_probe_decisions: int = 0
        self._total_corrections: int = 0
        self._last_inference_seconds: float = 0.1  # seed so fraction maths works

    # ------------------------------------------------------------------
    # Game / level lifecycle
    # ------------------------------------------------------------------

    def start_game(self, run_deadline: float) -> None:
        """Set the overall run deadline.  Must be called before decide()."""
        if not isfinite(run_deadline):
            raise ValueError("run_deadline must be finite")
        self._run_deadline = run_deadline

    def _planning_deadline(self) -> float:
        """Compute a per-decision planning deadline from the run deadline."""
        now = self._clock()
        remaining = self._run_deadline - now
        # Reserve at least deadline_reserve_fraction of total remaining
        usable = remaining * (1.0 - self.config.deadline_reserve_fraction)
        # Cap deliberation at deliberation_fraction of last measured inference time
        delib_budget = self._last_inference_seconds * self.config.deliberation_fraction
        # Planning gets the rest up to a per-decision maximum (usable / 4 is conservative)
        per_decision = min(usable / max(1, self._total_decisions + 1), usable / 4)
        return now + max(0.01, per_decision - delib_budget)

    def _on_level_change(self, new_level: int) -> None:
        """Reset per-level trackers on level advance."""
        self._level_id = new_level
        self._actions_this_level = 0
        self._stall_counter = 0
        self.memory.clear_branches()
        # A new level is a new internal-world context: reseed the recurrent state
        # on the next observation rather than carrying the old level's state.
        self._sim_state = None
        # Force a deliberator call at the start of the new level
        self._actions_since_deliberator = self.config.stall_threshold

    # ------------------------------------------------------------------
    # Measure — encode the observation into a WorldState
    # ------------------------------------------------------------------

    def _measure(self, obs: Observation) -> WorldState:
        """Convert a real Observation into a WorldState, correcting the internal
        world from the observation via the posterior.

        Maintains a persistent recurrent SimulatorState (h, z). On each real
        frame it encodes the observation and runs the posterior update so z (and
        thus the internal world) reflects what was actually seen — the central
        'progressively complete and correct the internal game' requirement. The
        corrected (h, z) is packed into WorldState.latent so the planner forks
        the real observed state.

        Degrades gracefully when the dynamics is a plain fixture without the
        posterior/encoder API (used by unit tests): it just carries the prior
        latent through, exactly as before.
        """
        grid = tuple(tuple(row) for row in obs.grid)

        encode = getattr(self.dynamics, "encode_observation", None)
        update_posterior = getattr(self.dynamics, "update_posterior", None)
        pack_latent = getattr(self.dynamics, "pack_latent", None)
        initial_state = getattr(self.dynamics, "initial_state", None)

        latent: tuple[float, ...]
        if encode and update_posterior and pack_latent and initial_state:
            # Seed a persistent recurrent state on the first real frame.
            if self._sim_state is None:
                self._sim_state = initial_state(game_id=obs.game_id)
            try:
                if self._last_measured_observation == obs.observation_id:
                    corrected = self._sim_state
                else:
                    obs_feat, _spatial, _hi_res = encode(grid, self._sim_state)
                    corrected = update_posterior(obs_feat, self._sim_state)
                self._sim_state = corrected
                self._last_measured_observation = obs.observation_id
                latent = pack_latent(corrected)
            except Exception:
                # Never let internal-world maintenance crash the decision loop;
                # fall back to the previous latent.
                latent = (self._last_world_state.latent
                          if self._last_world_state is not None else (0.0,))
        else:
            latent = (self._last_world_state.latent
                      if self._last_world_state is not None else (0.0,))

        return WorldState(
            game=obs.game_id,
            level=obs.level_id,
            latent=latent,
            grid=grid,
            actions=obs.to_actions(),
            model_version=obs.model_version,
            imagined=False,
            terminal=obs.is_terminal,
        )

    # ------------------------------------------------------------------
    # Reason — call deliberator when warranted → update condition
    # ------------------------------------------------------------------

    def _reason(
        self,
        obs: Observation,
        world_state: WorldState,
        *,
        is_initial: bool,
        planning_deadline: float,
    ) -> None:
        """Optionally invoke deliberator and update SimulatorCondition + GoalSpecs."""
        if self.deliberator is None:
            return

        stalled = self._stall_counter >= self.config.stall_threshold
        should = self.deliberator.should_call(
            self._actions_since_deliberator,
            is_initial=is_initial,
            stalled_progress=stalled,
        )
        if not should:
            return

        # Build structured input (no raw chat history)
        recent_grids = []
        recent_actions: list[tuple] = []
        recent_outcomes: list[bool] = []
        recent_errors: list[float | None] = []
        for ep in reversed(self.memory.query_episodes(
            level_id=obs.level_id, limit=4
        )):
            recent_grids.append(ep.before_grid())
            recent_actions.append((ep.action_id, ep.action_x, ep.action_y))
            recent_outcomes.append(ep.actual_success)
            recent_errors.append(ep.discrepancy_score if ep.prediction_available else None)

        belief_summaries = tuple(
            b.hypothesis for b in self.memory.query_beliefs(limit=4)
        )
        goal_summaries = tuple(
            g.description for g in self.memory.query_goals(
                status=None, level_context=obs.level_id, limit=4
            )
        )

        delib_input = DeliberatorInput(
            game_id=obs.game_id,
            level_id=obs.level_id,
            recent_grids=tuple(recent_grids[-4:]),
            recent_actions=tuple(recent_actions[-4:]),
            recent_outcomes=tuple(recent_outcomes[-4:]),
            active_belief_summaries=belief_summaries,
            active_goal_summaries=goal_summaries,
            prediction_errors=tuple(recent_errors[-4:]),
            real_actions_since_last_call=self._actions_since_deliberator,
            current_grid=tuple(tuple(row) for row in obs.grid),
        )

        delib_deadline = min(
            planning_deadline,
            self._clock() + self.deliberator.config.max_wall_seconds,
        )

        output: DeliberatorOutput = self.deliberator.call(
            delib_input, deadline=delib_deadline
        )

        self._total_deliberator_calls += 1
        self._actions_since_deliberator = 0

        if not output.mechanic_hypotheses and not output.goal_signals:
            return  # empty output: retain existing condition

        # Convert GoalSignals to GoalSpecs for the planner (no circular import)
        new_goals = tuple(
            GoalSpec(
                goal_id=gs.goal_id,
                description=gs.description,
                target_color=gs.target_color,
                target_region=gs.target_region,
                priority=gs.priority,
                status=gs.status,
            )
            for gs in output.goal_signals
            if gs.status in ("provisional", "active")
        )
        if new_goals:
            self._current_goals = new_goals

        self._current_condition = output.to_simulator_condition(
            obs.game_id, obs.level_id, obs.model_version
        )

    # ------------------------------------------------------------------
    # Look-ahead — plan through imagined states
    # ------------------------------------------------------------------

    def _look_ahead(
        self,
        world_state: WorldState,
        *,
        planning_deadline: float,
    ) -> tuple[GoalDirectedPlan | Plan, int]:
        """Run beam search; return (plan, imagined_steps_used)."""
        # Route planning through the hypothesis-conditioned dynamics when a
        # SimulatorCondition is active and the dynamics supports conditioning, so
        # the deliberator's mechanic hypotheses actually shape the imagined world.
        planning_dynamics = self.dynamics
        cond = self._current_condition
        pc = getattr(self.dynamics, "predict_conditioned", None)
        if (cond is not None and callable(pc) and cond.mechanic_hypotheses
                and getattr(self.dynamics, "supports_conditioning", True)):
            planning_dynamics = _ConditionedDynamics(self.dynamics, cond)
        backend = planning_dynamics
        cache = self._root_predictions
        class CachedRoot:
            def predict(self, state, action):
                if state == world_state:
                    if action not in cache:
                        cache[action] = backend.predict(state, action)
                    return cache[action]
                return backend.predict(state, action)
        planning_dynamics = CachedRoot()
        world = InternalGame(planning_dynamics, world_state)

        # Freeze weights during planning (no-op if dynamics lacks the method)
        freeze = getattr(self.dynamics, "freeze", None)
        unfreeze = getattr(self.dynamics, "unfreeze", None)
        if freeze:
            freeze()

        try:
            if self._current_goals:
                evaluator = GoalEvaluator(list(self._current_goals))
                planner = GoalDirectedPlanner(
                    evaluator,
                    horizon=self.config.horizon,
                    beam=self.config.beam,
                    max_predictions=self.config.max_predictions,
                    uncertainty_limit=self.config.uncertainty_limit,
                    risk_weight=self.config.risk_weight,
                    action_cost=self.config.action_cost,
                    clock=self._clock,
                )
                plan = planner.plan(world, deadline=planning_deadline)
                steps = planner.predictions_made
            else:
                planner_base = RecursivePlanner(
                    horizon=self.config.horizon,
                    beam=self.config.beam,
                    max_predictions=self.config.max_predictions,
                    uncertainty_limit=self.config.uncertainty_limit,
                    risk_weight=self.config.risk_weight,
                    action_cost=self.config.action_cost,
                    clock=self._clock,
                )
                plan = planner_base.plan(world, deadline=planning_deadline)
                steps = planner_base.predictions_made
        finally:
            if unfreeze:
                unfreeze()

        self._total_imagined_steps += steps
        return plan, steps

    # ------------------------------------------------------------------
    # Act — validate and select the first action
    # ------------------------------------------------------------------

    def _select_action(
        self,
        plan: GoalDirectedPlan | Plan,
        world_state: WorldState,
    ) -> tuple[Action, WorldState | None, bool, bool]:
        """Return (action, predicted_next_state, plan_was_empty, plan_was_ambiguous).

        When the plan is empty (calibration requested) or ambiguous, fall back
        to an uncertainty-directed probe: the action whose predicted uncertainty
        is highest (most informative).  Never a random button cycle.
        """
        ambiguous = getattr(plan, "ambiguous", False)
        empty = len(plan.actions) == 0
        self._used_procedure_id = None
        if not self.planning_enabled:
            if not world_state.actions:
                raise ValueError("No legal action reported by environment")
            return world_state.actions[0], None, True, False

        if not empty and not ambiguous:
            action = plan.actions[0]
            predicted = plan.predictions[0].state if plan.predictions else None
            return action, predicted, False, False

        # Reuse only a promoted, real-success action in an identical observable
        # context, and only when the current simulator agrees it is low risk.
        for procedure in self.memory.query_procedures(status="active", limit=16):
            for eid in procedure.supporting_ids:
                episode = self.memory.get_episode(eid)
                if (episode is None or episode.before_grid() != world_state.grid
                        or episode.model_version != world_state.model_version):
                    continue
                action = Action(episode.action_id, episode.action_x, episode.action_y)
                if action not in world_state.actions or self._clock() >= self._run_deadline:
                    continue
                prediction = self.dynamics.predict(world_state, action)
                if prediction.risk <= 0.1 and prediction.uncertainty <= self.config.uncertainty_limit:
                    self._used_procedure_id = procedure.procedure_id
                    self._procedure_reuses += 1
                    return action, prediction.state, empty, ambiguous

        # Probe: pick the action with highest uncertainty (most to learn)
        best_action: Action | None = None
        best_prediction: WorldState | None = None
        best_uncertainty = -1.0
        for candidate in world_state.actions:
            try:
                pred = self._root_predictions.get(candidate)
                if pred is None:
                    if self._clock() >= min(self._run_deadline, self._prediction_deadline):
                        continue
                    pred = self.dynamics.predict(world_state, candidate)
                if pred.uncertainty > best_uncertainty:
                    best_uncertainty = pred.uncertainty
                    best_action = candidate
                    best_prediction = pred.state
            except Exception:
                pass

        if best_action is None:
            # Absolute fallback: first available action
            if not world_state.actions:
                raise ValueError("No legal action reported by environment")
            best_action = world_state.actions[0]

        self._total_probe_decisions += 1
        return best_action, best_prediction, empty, ambiguous

    # ------------------------------------------------------------------
    # Main MRLA cycle
    # ------------------------------------------------------------------

    def decide(self, obs: Observation) -> DecisionRecord:
        """Run one complete Measure → Reason → Look-ahead → Act cycle.

        Returns a DecisionRecord that must be passed to correct() after the
        real action has been dispatched and the next frame received.
        """
        t0 = self._clock()

        # Level change detection
        if obs.level_id != self._level_id:
            self._on_level_change(obs.level_id)

        is_initial = self._total_decisions == 0

        # --- Measure ---
        world_state = self._measure(obs)

        # --- Reason ---
        planning_dl = self._planning_deadline()
        self._reason(obs, world_state, is_initial=is_initial, planning_deadline=planning_dl)
        self._root_predictions = {}
        self._prediction_deadline = planning_dl

        # --- Look-ahead ---
        hyp_count = len(self._current_condition.mechanic_hypotheses) if self._current_condition else 0
        goal_count = len(self._current_condition.goal_signals) if self._current_condition else 0

        try:
            if self.planning_enabled:
                plan, imagined_steps = self._look_ahead(world_state, planning_deadline=planning_dl)
            else:
                plan, imagined_steps = Plan((), (), 0.0), 0
        except Exception:
            # Dynamics raised during planning — treat as empty plan (probe requested)
            plan = Plan((), (), 0.0)
            imagined_steps = 0

        # Store imagined branches in working memory (cleared at next plan)
        self.memory.clear_branches()

        # --- Act ---
        action, predicted_state, was_empty, was_ambiguous = self._select_action(
            plan, world_state
        )

        # Validate predicted state version if we have one
        if predicted_state is not None and self._current_condition is not None:
            try:
                from agent.model import validate_prediction_against_condition
                from agent.internal_world import Prediction
                # Wrap into Prediction for the validator
                # (utility/uncertainty/risk are irrelevant here; only game/version checked)
                dummy_pred = plan.predictions[0] if plan.predictions else None
                if dummy_pred is not None:
                    validate_prediction_against_condition(dummy_pred, self._current_condition)
            except (ValueError, IndexError):
                pass  # Log but don't block; mismatches handled in correct()

        # Record the decision before dispatching
        t1 = self._clock()
        self._last_inference_seconds = max(0.001, t1 - t0)

        decision = DecisionRecord(
            decision_id=uuid.uuid4().hex,
            game_id=obs.game_id,
            level_id=obs.level_id,
            observation_id=obs.observation_id,
            model_version=obs.model_version,
            action=action,
            predicted_state=predicted_state,
            plan_value=plan.value,
            plan_length=len(plan.actions),
            plan_was_empty=was_empty,
            plan_was_ambiguous=was_ambiguous,
            deliberator_called=(self._actions_since_deliberator == 0 and is_initial is False)
                or (self._total_deliberator_calls > 0 and self._actions_since_deliberator == 0),
            hypotheses_count=hyp_count,
            goals_count=goal_count,
            imagined_steps=imagined_steps,
            wall_seconds=self._last_inference_seconds,
        )

        self._last_world_state = world_state
        self._last_decision = decision
        self._total_decisions += 1
        self._actions_since_deliberator += 1
        self._actions_this_level += 1

        return decision

    # ------------------------------------------------------------------
    # Correct — compare prediction to outcome, update memory
    # ------------------------------------------------------------------

    def correct(self, obs_after: Observation, decision: DecisionRecord) -> None:
        """Record real outcome; update memory, beliefs, and stall tracking.

        Must be called after every decide() once the outcome frame is available.
        Feedback is recorded before the next decision (blueprint §5).
        """
        if decision.decision_id in self._corrected_decisions:
            return
        if obs_after.game_id != self.config.game_id:
            raise ValueError(
                f"correct() obs game_id {obs_after.game_id!r} != "
                f"controller game_id {self.config.game_id!r}"
            )

        if decision.game_id != obs_after.game_id:
            raise ValueError("Feedback decision belongs to another game")
        if self._last_decision is not None and decision.decision_id != self._last_decision.decision_id:
            raise ValueError("Feedback must match the pending decision")
        self._corrected_decisions.add(decision.decision_id)
        # Only real execution advances the live recurrent state. Imagined branches
        # never mutate it. The next measurement corrects this prior with reality.
        if self._sim_state is not None and hasattr(self.dynamics, "imagine_step"):
            self._sim_state = self.dynamics.imagine_step(decision.action, self._sim_state)

        # Level progress tracking
        if obs_after.level_id > decision.level_id:
            self._stall_counter = 0
        else:
            self._stall_counter += 1

        # Build predicted grid for the episode record. HONESTY: when the decision
        # made no genuine prediction (an uncertainty probe), do NOT store the
        # actual outcome as the "prediction" — that manufactures zero error.
        # Record with prediction_available=False so metrics treat it correctly.
        observed_grid = tuple(tuple(row) for row in obs_after.grid)
        if decision.predicted_state is not None:
            predicted_grid = decision.predicted_state.grid
            prediction_available = True
        else:
            predicted_grid = observed_grid  # placeholder bytes only; flagged absent
            prediction_available = False

        before_grid = observed_grid
        if self._last_world_state is not None:
            before_grid = self._last_world_state.grid

        # Record the episode in WorkingMemory
        episode = None
        try:
            episode = self.memory.add_episode(
                level_id=decision.level_id,
                observation_id=decision.observation_id,
                model_version=decision.model_version,
                action_id=decision.action.id,
                action_x=decision.action.x,
                action_y=decision.action.y,
                before_grid=before_grid,
                predicted_grid=predicted_grid,
                observed_grid=observed_grid,
                actual_success=(obs_after.actual_success or obs_after.level_id > decision.level_id),
                prediction_available=prediction_available,
            )
        except Exception:
            pass  # Memory full or invalid; don't crash the game loop

        # Wire same-run learning to the live loop (was previously disconnected).
        if episode is not None:
            self._maybe_learn(episode, replace(obs_after, actual_success=episode.actual_success))

        self._total_corrections += 1

    def set_adapter(self, adapter_manager, adapter_hook) -> None:
        """Connect an online adapter. adapter_hook(manager) performs one bounded
        update using the caller's replay source; the controller only triggers it
        with the correct per-decision cadence. Optional; safe to never call."""
        self._adapter_manager = adapter_manager
        self._adapter_hook = adapter_hook

    def _promote_success_procedure(self, episode) -> None:
        """On a real success, record/strengthen a contextual procedure with
        diverse-context promotion (uses the memory's R12 utilities that were
        previously disconnected from the live loop)."""
        eid = getattr(episode, "record_id", None)
        if eid is None:
            return
        self._recent_success_ids.append((episode.level_id, eid))
        self._recent_success_ids = self._recent_success_ids[-16:]

        import hashlib
        signature = episode.before_grid_bytes + repr((episode.action_id, episode.action_x, episode.action_y)).encode()
        name = "progress:" + hashlib.sha256(signature).hexdigest()[:24]
        existing = [p for p in self.memory.query_procedures(limit=64) if p.name == name]
        if not existing:
            self.memory.add_procedure(
                name=name,
                preconditions=("observable_context:" + hashlib.sha256(episode.before_grid_bytes).hexdigest(),),
                steps=(repr((episode.action_id, episode.action_x, episode.action_y)),),
                postconditions=("observed_level_progress",),
                supporting_ids=(eid,),
            )
        else:
            # Diverse-context promotion: only strengthens to 'active' when the
            # supporting successes span >=2 level contexts (R12 anti-overfitting).
            try:
                self.memory.promote_procedure_diverse(
                    existing[0].procedure_id, new_supporting_id=eid
                )
            except Exception:
                pass

    def _maybe_learn(self, episode, obs_after: Observation) -> None:
        """Connect adapter updates and procedural learning to the live loop.

        - Adapter: after enough real episodes, attempt one bounded, validated
          online update via the AdapterManager (rolls back on regression).
        - Procedures: on real level progress, promote/strengthen a contextual
          procedure from the recent successful episodes; on failure, revoke.
        All guarded so a learning failure never crashes the game loop.
        """
        if episode is None:
            return
        if self._used_procedure_id is not None and not episode.actual_success:
            self.memory.revoke_procedure_on_contradiction(
                self._used_procedure_id, failing_episode_id=episode.record_id
            )
        # Procedural learning: a real success (level progress) is the signal.
        try:
            if obs_after.actual_success:
                self._promote_success_procedure(episode)
        except Exception:
            pass
        # Adapter learning: delegate cadence/validation to the manager if present.
        try:
            if self._adapter_manager is not None and self._adapter_hook is not None:
                self._adapter_manager.begin_decision()
                self._adapter_hook(self._adapter_manager)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a JSON-safe stats dict for telemetry and evaluation."""
        delib_stats = self.deliberator.stats() if self.deliberator else {}
        return {
            "game_id": self.config.game_id,
            "model_version": self.config.model_version,
            "total_decisions": self._total_decisions,
            "total_deliberator_calls": self._total_deliberator_calls,
            "total_imagined_steps": self._total_imagined_steps,
            "total_probe_decisions": self._total_probe_decisions,
            "total_corrections": self._total_corrections,
            "adapter": self._adapter_manager.stats() if self._adapter_manager else None,
            "procedure_reuses": self._procedure_reuses,
            "mechanic_conditioning_supported": getattr(self.dynamics, "supports_conditioning", False),
            "last_inference_seconds": self._last_inference_seconds,
            "memory": self.memory.stats(),
            "deliberator": delib_stats,
        }

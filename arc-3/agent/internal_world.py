"""Internal-game simulation and recursive planning contracts.

This is the control layer, not trained dynamics. Production requires a learned
Dynamics implementation; no action-frequency or identity fallback is provided.
No component here receives an official environment or can send real actions.

R08 additions: GoalEvaluator, Subgoal, milestone-conditioned GoalDirectedPlanner
that handles competing falsifiable goals and delayed-credit attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Action:
    id: int
    x: int | None = None
    y: int | None = None

    def __post_init__(self):
        if type(self.id) is not int or not 0 <= self.id <= 7:
            raise ValueError("ARC-3 action ID must be 0..7")
        if self.id == 6:
            if any(type(v) is not int or not 0 <= v <= 63 for v in (self.x, self.y)):
                raise ValueError("ACTION6 requires integer x,y in 0..63")
        elif self.x is not None or self.y is not None:
            raise ValueError("Only ACTION6 accepts coordinates")


@dataclass(frozen=True)
class WorldState:
    game: str
    level: int
    latent: tuple[float, ...]
    grid: tuple[tuple[int, ...], ...]
    actions: tuple[Action, ...]
    model_version: str
    imagined: bool = False
    terminal: bool = False

    def __post_init__(self):
        # frozen dataclasses do not freeze caller-owned nested lists.
        object.__setattr__(self, "latent", tuple(self.latent))
        object.__setattr__(self, "grid", tuple(tuple(row) for row in self.grid))
        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.game or not self.model_version:
            raise ValueError("Game and model version are required")
        if type(self.level) is not int or self.level < 0:
            raise ValueError("Level must be a nonnegative integer")
        if any(not isinstance(a, Action) for a in self.actions):
            raise ValueError("World actions must be validated Action objects")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("Duplicate world actions")
        if not self.latent or not all(isfinite(v) for v in self.latent):
            raise ValueError("A finite latent state is required")
        if not self.grid or not 1 <= len(self.grid) <= 64 or not 1 <= len(self.grid[0]) <= 64:
            raise ValueError("ARC-3 grids must be 1..64 cells on each axis")
        if any(len(row) != len(self.grid[0]) for row in self.grid):
            raise ValueError("Grid must be rectangular")
        if any(type(c) is not int or not 0 <= c <= 15 for row in self.grid for c in row):
            raise ValueError("ARC-3 colors must be integer categories 0..15")


@dataclass(frozen=True)
class Prediction:
    state: WorldState
    utility: float
    uncertainty: float
    risk: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.utility, self.uncertainty, self.risk)):
            raise ValueError("Prediction scores must be finite")
        if not 0 <= self.uncertainty <= 1 or not 0 <= self.risk <= 1:
            raise ValueError("Uncertainty and risk must lie in [0,1]")


class Dynamics(Protocol):
    """Implement with pretrained weights plus game-local adaptation state."""
    def predict(self, state: WorldState, action: Action) -> Prediction: ...


class InternalGame:
    def __init__(self, dynamics: Dynamics, state: WorldState):
        self.dynamics = dynamics
        self.state = state

    def fork(self) -> InternalGame:
        # Immutable states make branch isolation explicit. Predictor must be pure;
        # online weight updates occur between plans, never inside imagined steps.
        return InternalGame(self.dynamics, self.state)

    def step(self, action: Action) -> Prediction:
        if self.state.terminal:
            raise ValueError("Cannot advance a terminal imagined world")
        if action not in self.state.actions:
            raise ValueError("Action is unavailable in this world state")
        prediction = self.dynamics.predict(self.state, action)
        if (prediction.state.game, prediction.state.model_version) != (
            self.state.game, self.state.model_version
        ):
            raise ValueError("Prediction crossed a game or model-version boundary")
        prediction = replace(prediction, state=replace(prediction.state, imagined=True))
        self.state = prediction.state
        return prediction


@dataclass(frozen=True)
class Plan:
    actions: tuple[Action, ...]
    predictions: tuple[Prediction, ...]
    value: float


class RecursivePlanner:
    """Bounded beam search through predicted states, not recorded graph edges.

Utility is supplied by grounded goal hypotheses. Terminal does not imply a win.
Uncertain branches shorten the horizon; an empty plan requests calibration.
"""
    def __init__(self, *, horizon=4, beam=8, max_predictions=128,
                 uncertainty_limit=0.35, risk_weight=4.0, action_cost=0.01,
                 clock: Callable[[], float] = monotonic):
        if any(type(v) is not int or v < 1 for v in (horizon, beam, max_predictions)):
            raise ValueError("Planning budgets must be positive integers")
        if not isfinite(uncertainty_limit) or not 0 <= uncertainty_limit <= 1:
            raise ValueError("Uncertainty limit must lie in [0,1]")
        if any(not isfinite(v) or v < 0 for v in (risk_weight, action_cost)):
            raise ValueError("Planning costs must be finite and nonnegative")
        self.horizon, self.beam, self.max_predictions = horizon, beam, max_predictions
        self.uncertainty_limit, self.risk_weight = uncertainty_limit, risk_weight
        self.action_cost, self.clock = action_cost, clock
        self.predictions_made = 0

    def plan(self, world: InternalGame, *, deadline: float) -> Plan:
        if deadline != deadline:
            raise ValueError("Deadline must not be NaN")
        empty = Plan((), (), 0.0)
        frontier = [(world.fork(), empty)]
        best = empty
        self.predictions_made = 0
        for _depth in range(self.horizon):
            expanded = []
            for branch, parent in frontier:
                if branch.state.terminal:
                    continue
                for action in branch.state.actions:
                    if self.clock() >= deadline or self.predictions_made >= self.max_predictions:
                        return best
                    child = branch.fork()
                    prediction = child.step(action)
                    self.predictions_made += 1
                    # Do not select a prediction returned after the deadline.
                    # Backend cancellation is still required to bound a hung call.
                    if self.clock() >= deadline:
                        return best
                    if prediction.uncertainty > self.uncertainty_limit:
                        continue
                    value = parent.value + prediction.utility - self.action_cost
                    value -= self.risk_weight * prediction.risk
                    candidate = Plan(parent.actions + (action,),
                                     parent.predictions + (prediction,), value)
                    expanded.append((child, candidate))
                    if not best.actions or candidate.value > best.value:
                        best = candidate
            frontier = sorted(expanded, key=lambda item: item[1].value, reverse=True)[:self.beam]
            if not frontier:
                break
        return best


@dataclass(frozen=True)
class Experience:
    before: WorldState
    action: Action
    predicted: WorldState
    observed: WorldState
    changed_cell_error: int
    actual_success: bool


class EpisodeMemory:
    """Bounded observed experience available immediately in this same game.

Stores validation targets, not hidden reasoning traces. A future learner consumes
these records to revise dynamics and compile contextual procedures.
"""
    def __init__(self, game: str, capacity: int = 512):
        if capacity < 1:
            raise ValueError("Memory capacity must be positive")
        self.game, self.capacity = game, capacity
        self.records: list[Experience] = []

    def record(self, before, action, predicted, observed, *, actual_success=False):
        if before.imagined or observed.imagined or not predicted.imagined:
            raise ValueError("Learning requires real endpoints and a prior prediction")
        if any(s.game != self.game for s in (before, predicted, observed)):
            raise ValueError("Experience crossed the game-memory boundary")
        if predicted.model_version != before.model_version:
            raise ValueError("Prediction model version differs from its source")
        if type(actual_success) is not bool:
            raise ValueError("Success must be a real boolean observation")
        if action not in before.actions:
            raise ValueError("Recorded action was not available")
        if tuple(map(len, predicted.grid)) != tuple(map(len, observed.grid)):
            error = max(sum(map(len, predicted.grid)), sum(map(len, observed.grid)))
        else:
            error = sum(a != b for ra, rb in zip(predicted.grid, observed.grid)
                        for a, b in zip(ra, rb))
        event = Experience(before, action, predicted, observed, error, actual_success)
        self.records.append(event)
        del self.records[:-self.capacity]
        return event


# ---------------------------------------------------------------------------
# R08: Goal inference and delayed-credit planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalSpec:
    """A grounded, observable goal specification passed to the planner.

    Carries the same information as model.GoalSignal but is self-contained:
    no import of agent.model to avoid circular dependencies. The controller
    constructs GoalSpecs from active/provisional GoalSignals.

    Fields
    ------
    goal_id : str
        Opaque ID matching a GoalHypothesis or GoalSignal record.
    description : str
        Human-readable label used only for diagnostics.
    target_color : int | None
        ARC color category (0..15). None = color-agnostic goal.
    target_region : tuple[int,int,int,int] | None
        (x_min, y_min, x_max, y_max) bounding box in grid coords.
    priority : float
        In (0, 1]. Scales how strongly this goal's reward weights the utility.
    status : str
        'provisional' or 'active'. Falsified goals must not be passed here.
    """
    goal_id: str
    description: str
    target_color: int | None
    target_region: tuple[int, int, int, int] | None
    priority: float
    status: str  # 'provisional' | 'active'

    def __post_init__(self):
        if not self.goal_id:
            raise ValueError("GoalSpec goal_id must not be empty")
        if not self.description.strip():
            raise ValueError("GoalSpec description must not be empty")
        if self.target_color is not None:
            if not isinstance(self.target_color, int) or not 0 <= self.target_color <= 15:
                raise ValueError(
                    f"GoalSpec target_color must be 0..15, got {self.target_color!r}"
                )
        if self.target_region is not None:
            x0, y0, x1, y1 = self.target_region
            if not (0 <= x0 <= x1 <= 63 and 0 <= y0 <= y1 <= 63):
                raise ValueError(
                    f"GoalSpec target_region invalid: {self.target_region!r}"
                )
        if not isfinite(self.priority) or not 0.0 < self.priority <= 1.0:
            raise ValueError(
                f"GoalSpec priority must be in (0,1], got {self.priority!r}"
            )
        if self.status not in ("provisional", "active"):
            raise ValueError(
                f"GoalSpec status must be 'provisional' or 'active', got {self.status!r}"
            )


@dataclass(frozen=True)
class Subgoal:
    """A milestone that must be satisfied before pursuing the terminal goal.

    Subgoals form an ordered sequence: each must be reached before the next
    is evaluated. A failed milestone triggers revision (blueprint §5).

    Fields
    ------
    subgoal_id : str
    description : str
    target_color : int | None
    target_region : tuple[int,int,int,int] | None
    required_before : str | None
        goal_id of the next subgoal this must precede; None = terminal goal.
    """
    subgoal_id: str
    description: str
    target_color: int | None
    target_region: tuple[int, int, int, int] | None
    required_before: str | None  # next subgoal_id or None

    def __post_init__(self):
        if not self.subgoal_id:
            raise ValueError("Subgoal subgoal_id must not be empty")
        if not self.description.strip():
            raise ValueError("Subgoal description must not be empty")
        if self.target_color is not None:
            if not isinstance(self.target_color, int) or not 0 <= self.target_color <= 15:
                raise ValueError(
                    f"Subgoal target_color must be 0..15, got {self.target_color!r}"
                )
        if self.target_region is not None:
            x0, y0, x1, y1 = self.target_region
            if not (0 <= x0 <= x1 <= 63 and 0 <= y0 <= y1 <= 63):
                raise ValueError(
                    f"Subgoal target_region invalid: {self.target_region!r}"
                )


class GoalEvaluator:
    """Scores a WorldState against a set of competing GoalSpecs.

    Produces a composite utility signal for the planner by:
      1. Checking whether each goal's observable condition is satisfied in
         the predicted grid (color presence or bounding-box coverage).
      2. Weighting by priority and promotion status (active > provisional).
      3. Applying delayed-credit discounting: partial progress toward a
         goal earns fractional reward proportional to cells matching the
         target color within the target region.

    No fabricated labels, hidden state, or pretrained model calls are made.
    All scoring is deterministic given the grid and goal specs.
    """

    # Status multipliers: active goals contribute more to utility
    _STATUS_WEIGHT = {"active": 1.0, "provisional": 0.4}

    def __init__(self, goals: Sequence[GoalSpec], subgoals: Sequence[Subgoal] = ()):
        if not goals:
            raise ValueError("GoalEvaluator requires at least one GoalSpec")
        for g in goals:
            if not isinstance(g, GoalSpec):
                raise TypeError(f"Expected GoalSpec, got {type(g).__name__}")
        self.goals: tuple[GoalSpec, ...] = tuple(goals)
        self.subgoals: tuple[Subgoal, ...] = tuple(subgoals)

        # Build required_before chain.
        # _subgoal_chain is an ordered list of subgoal_ids following the
        # required_before links from the root (the subgoal with no predecessor).
        # For a linear chain A→B→C: A.required_before='B', B.required_before='C',
        # C.required_before=None.  The chain is [A, B, C].
        # Falls back to insertion order when required_before fields are absent or
        # form no valid chain (e.g. all None).
        self._subgoal_chain: tuple[str, ...] = self._build_chain(self.subgoals)
        # Index into _subgoal_chain for the currently active milestone.
        self._active_subgoal_idx: int = 0

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain(subgoals: tuple["Subgoal", ...]) -> tuple[str, ...]:
        """Order subgoal IDs by following required_before links.

        Algorithm:
          1. Build a map {subgoal_id → Subgoal} and a set of IDs that are
             pointed-to by some required_before (i.e. they are successors).
          2. The root is the subgoal with no predecessor (not in the pointed-to
             set). If there is exactly one such root, follow the chain from it.
          3. If the chain does not consume all subgoals (e.g. a cycle or
             disconnected nodes), append remaining subgoals in insertion order
             so no subgoal is silently lost.
          4. Falls back to insertion order when no valid chain root is found
             (all required_before=None, or multiple roots).

        This means single-root linear chains are fully ordered; everything
        else degrades gracefully to the previous index-based behaviour.
        """
        if not subgoals:
            return ()

        by_id: dict[str, "Subgoal"] = {sg.subgoal_id: sg for sg in subgoals}
        # IDs that appear as a required_before target (i.e. they are successors)
        pointed_to: set[str] = {
            sg.required_before
            for sg in subgoals
            if sg.required_before is not None
        }
        # Roots: subgoals that are not pointed to by any other subgoal
        roots = [sg for sg in subgoals if sg.subgoal_id not in pointed_to]

        if len(roots) != 1:
            # Multiple roots or all roots (all required_before=None): fall back
            # to insertion order, which is the previous behaviour.
            return tuple(sg.subgoal_id for sg in subgoals)

        # Follow the single-root chain
        chain: list[str] = []
        current_id: str | None = roots[0].subgoal_id
        visited: set[str] = set()
        while current_id is not None:
            if current_id in visited or current_id not in by_id:
                # Cycle or dangling reference — stop following
                break
            visited.add(current_id)
            chain.append(current_id)
            current_id = by_id[current_id].required_before

        # Append any subgoals not reached by the chain (safety: no silent loss)
        for sg in subgoals:
            if sg.subgoal_id not in visited:
                chain.append(sg.subgoal_id)

        return tuple(chain)

    # ------------------------------------------------------------------
    # Observable-condition checks
    # ------------------------------------------------------------------

    @staticmethod
    def _cells_in_region(
        grid: tuple[tuple[int, ...], ...],
        color: int | None,
        region: tuple[int, int, int, int] | None,
    ) -> tuple[int, int]:
        """Return (matching_cells, total_region_cells).

        If region is None, the whole grid is the region.
        If color is None, all non-zero cells count as matching.
        """
        if not grid:
            return 0, 0
        h, w = len(grid), len(grid[0])
        if region is not None:
            x0, y0, x1, y1 = region
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
        else:
            x0, y0, x1, y1 = 0, 0, w - 1, h - 1

        total = (x1 - x0 + 1) * (y1 - y0 + 1)
        if total <= 0:
            return 0, 0

        matching = sum(
            1
            for row_idx in range(y0, y1 + 1)
            for col_idx in range(x0, x1 + 1)
            if (color is None and grid[row_idx][col_idx] != 0)
            or (color is not None and grid[row_idx][col_idx] == color)
        )
        return matching, total

    def goal_progress(self, state: WorldState, goal: GoalSpec) -> float:
        """Return a [0, 1] progress score toward this goal in the given state.

        0.0 = no evidence of goal satisfaction.
        1.0 = goal fully satisfied (all target cells match).
        Intermediate values give partial credit for partial coverage.
        """
        matching, total = self._cells_in_region(
            state.grid, goal.target_color, goal.target_region
        )
        if total == 0:
            return 0.0
        return matching / total

    def is_goal_satisfied(self, state: WorldState, goal: GoalSpec, threshold: float = 1.0) -> bool:
        """True when goal_progress >= threshold."""
        return self.goal_progress(state, goal) >= threshold

    # ------------------------------------------------------------------
    # Subgoal milestone tracking
    # ------------------------------------------------------------------

    def current_subgoal(self) -> Subgoal | None:
        """Return the currently active subgoal, or None if all are cleared."""
        if self._active_subgoal_idx < len(self._subgoal_chain):
            current_id = self._subgoal_chain[self._active_subgoal_idx]
            # Look up by id in the subgoals tuple
            for sg in self.subgoals:
                if sg.subgoal_id == current_id:
                    return sg
        return None

    def advance_subgoal(self, state: WorldState, threshold: float = 1.0) -> bool:
        """Check if the current subgoal is satisfied; advance if so.

        Uses the required_before chain order (built in __init__) rather than
        raw insertion order. Returns True when a milestone is advanced.
        A failed precondition or milestone stops execution; the caller is
        responsible for triggering revision (blueprint §5).
        """
        sg = self.current_subgoal()
        if sg is None:
            return False
        matching, total = self._cells_in_region(
            state.grid, sg.target_color, sg.target_region
        )
        if total > 0 and matching / total >= threshold:
            self._active_subgoal_idx += 1
            return True
        return False

    def reset_subgoals(self) -> None:
        """Reset subgoal progress to the chain root (e.g. on level reset)."""
        self._active_subgoal_idx = 0

    # ------------------------------------------------------------------
    # Composite utility
    # ------------------------------------------------------------------

    def utility(self, state: WorldState, *, partial_credit_weight: float = 0.3) -> float:
        """Return a composite utility score for the given WorldState.

        Combines:
          - Full goal satisfaction reward (1.0 * priority * status_weight)
          - Partial progress reward (partial_credit_weight * progress * priority * status_weight)
          - Subgoal milestone bonus (0.5 * status_weight of highest-priority active goal)
            when the current subgoal is satisfied

        Competing goals each contribute independently; the planner sees the sum.
        Falsified goals are excluded by contract (caller must not pass them).
        """
        if not isfinite(partial_credit_weight) or partial_credit_weight < 0:
            raise ValueError("partial_credit_weight must be finite and nonnegative")

        total = 0.0
        for goal in self.goals:
            sw = self._STATUS_WEIGHT.get(goal.status, 0.4)
            progress = self.goal_progress(state, goal)
            if progress >= 1.0:
                # Full satisfaction
                total += goal.priority * sw
            else:
                # Partial / delayed credit
                total += partial_credit_weight * progress * goal.priority * sw

        # Subgoal bonus: reward reaching the current milestone
        sg = self.current_subgoal()
        if sg is not None:
            sg_matching, sg_total = self._cells_in_region(
                state.grid, sg.target_color, sg.target_region
            )
            if sg_total > 0:
                sg_progress = sg_matching / sg_total
                # Use priority of the highest-priority active goal as the base
                max_priority = max((g.priority for g in self.goals), default=0.5)
                total += 0.5 * sg_progress * max_priority

        return total

    def competing_goals_disagree(self, state: WorldState, threshold: float = 0.8) -> bool:
        """True when two or more goals are above threshold simultaneously.

        Indicates an ambiguous state; the planner should probe rather than commit.
        """
        satisfied_count = sum(
            1 for g in self.goals if self.goal_progress(state, g) >= threshold
        )
        return satisfied_count >= 2


@dataclass(frozen=True)
class GoalDirectedPlan:
    """Result of GoalDirectedPlanner.plan(): extends Plan with goal attribution."""
    actions: tuple[Action, ...]
    predictions: tuple[Prediction, ...]
    value: float
    # Which goal_ids received credit at each step (same length as actions, or empty)
    goal_attribution: tuple[tuple[str, ...], ...]
    # Whether any competing goals disagreed during planning
    ambiguous: bool


class GoalDirectedPlanner:
    """Milestone-conditioned beam search with competing-goal utility and delayed credit.

    Extends RecursivePlanner with:
      - A GoalEvaluator that provides utility per predicted state.
      - Milestone checking: if a subgoal is reached during a branch, the evaluator
        is advanced and subsequent steps earn the next subgoal's reward.
      - Goal attribution: each plan step records which goals contributed utility.
      - Ambiguity detection: competing goals above the threshold request probing.

    blueprint §6: horizon 4, beam 8, max 128 predictions; extend toward horizon 8
    only when calibration and time justify it.
    """

    def __init__(
        self,
        evaluator: GoalEvaluator,
        *,
        horizon: int = 4,
        beam: int = 8,
        max_predictions: int = 128,
        uncertainty_limit: float = 0.35,
        risk_weight: float = 4.0,
        action_cost: float = 0.01,
        ambiguity_threshold: float = 0.8,
        clock: Callable[[], float] = monotonic,
    ):
        if not isinstance(evaluator, GoalEvaluator):
            raise TypeError("evaluator must be a GoalEvaluator")
        if any(type(v) is not int or v < 1 for v in (horizon, beam, max_predictions)):
            raise ValueError("Planning budgets must be positive integers")
        if not isfinite(uncertainty_limit) or not 0 <= uncertainty_limit <= 1:
            raise ValueError("Uncertainty limit must lie in [0,1]")
        if any(not isfinite(v) or v < 0 for v in (risk_weight, action_cost)):
            raise ValueError("Planning costs must be finite and nonneg")
        if not isfinite(ambiguity_threshold) or not 0 < ambiguity_threshold <= 1:
            raise ValueError("ambiguity_threshold must be in (0,1]")

        self.evaluator = evaluator
        self.horizon = horizon
        self.beam = beam
        self.max_predictions = max_predictions
        self.uncertainty_limit = uncertainty_limit
        self.risk_weight = risk_weight
        self.action_cost = action_cost
        self.ambiguity_threshold = ambiguity_threshold
        self.clock = clock
        self.predictions_made = 0

    def plan(self, world: InternalGame, *, deadline: float) -> GoalDirectedPlan:
        """Beam-search through predicted states using goal-directed utility.

        Returns an empty GoalDirectedPlan when:
          - No action passes the uncertainty limit (requests calibration).
          - Budget or deadline is exhausted.
          - Competing goals are ambiguous at the root state (requests probing).
        """
        if deadline != deadline:
            raise ValueError("Deadline must not be NaN")

        empty = GoalDirectedPlan((), (), 0.0, (), False)
        self.predictions_made = 0

        # Check root-state ambiguity before spending budget
        if self.evaluator.competing_goals_disagree(
            world.state, self.ambiguity_threshold
        ):
            return GoalDirectedPlan((), (), 0.0, (), ambiguous=True)

        # (branch_world, plan_so_far, evaluator_copy)
        # We snapshot the evaluator's subgoal index per branch so milestones
        # are independent across beam branches.
        frontier: list[tuple[InternalGame, GoalDirectedPlan, int]] = [
            (world.fork(), empty, self.evaluator._active_subgoal_idx)
        ]
        best = empty

        for _depth in range(self.horizon):
            expanded: list[tuple[InternalGame, GoalDirectedPlan, int]] = []
            for branch, parent, sg_idx in frontier:
                if branch.state.terminal:
                    continue
                for action in branch.state.actions:
                    if (
                        self.clock() >= deadline
                        or self.predictions_made >= self.max_predictions
                    ):
                        return best
                    child = branch.fork()
                    prediction = child.step(action)
                    self.predictions_made += 1

                    if self.clock() >= deadline:
                        return best
                    if prediction.uncertainty > self.uncertainty_limit:
                        continue

                    # Evaluate utility for this predicted state, advancing
                    # subgoals on a per-branch copy of the index.
                    child_sg_idx = sg_idx
                    self.evaluator._active_subgoal_idx = child_sg_idx
                    self.evaluator.advance_subgoal(prediction.state)
                    child_sg_idx = self.evaluator._active_subgoal_idx

                    goal_utility = self.evaluator.utility(prediction.state)
                    # Restore evaluator to root state after scoring
                    self.evaluator._active_subgoal_idx = 0

                    step_value = (
                        parent.value
                        + goal_utility
                        - self.action_cost
                        - self.risk_weight * prediction.risk
                    )

                    # Record which goals contributed (progress > 0)
                    contributing = tuple(
                        g.goal_id
                        for g in self.evaluator.goals
                        if self.evaluator.goal_progress(prediction.state, g) > 0
                    )

                    candidate = GoalDirectedPlan(
                        actions=parent.actions + (action,),
                        predictions=parent.predictions + (prediction,),
                        value=step_value,
                        goal_attribution=parent.goal_attribution + (contributing,),
                        ambiguous=False,
                    )
                    expanded.append((child, candidate, child_sg_idx))
                    if not best.actions or candidate.value > best.value:
                        best = candidate

            frontier = sorted(
                expanded, key=lambda item: item[1].value, reverse=True
            )[: self.beam]
            if not frontier:
                break

        # Restore evaluator subgoal index to 0 after planning
        self.evaluator._active_subgoal_idx = 0
        return best


# ---------------------------------------------------------------------------
# R08-A: GoalSignal → GoalSpec adapter and GoalHypothesis → GoalSpec helper
# ---------------------------------------------------------------------------

def goal_signal_to_goal_spec(signal: object) -> "GoalSpec":
    """Convert a GoalSignal (from agent.model) to a GoalSpec for the planner.

    GoalSignal and GoalSpec are structurally isomorphic but kept in separate
    modules to avoid circular imports. This adapter is the single conversion
    point; the controller calls it after every deliberator output.

    Parameters
    ----------
    signal : GoalSignal
        A validated GoalSignal from a DeliberatorOutput or SimulatorCondition.
        Must not have status='falsified' (GoalSignal already forbids it).

    Returns
    -------
    GoalSpec
        Equivalent GoalSpec ready for GoalEvaluator.
    """
    # Import here to avoid top-level circular dependency
    from agent.model import GoalSignal  # type: ignore[attr-defined]
    if not isinstance(signal, GoalSignal):
        raise TypeError(
            f"goal_signal_to_goal_spec expects a GoalSignal, got {type(signal).__name__}"
        )
    return GoalSpec(
        goal_id=signal.goal_id,
        description=signal.description,
        target_color=signal.target_color,
        target_region=signal.target_region,
        priority=signal.priority,
        status=signal.status,
    )


def goal_hypothesis_to_goal_spec(
    hypothesis: object,
    *,
    target_color: int | None = None,
    target_region: tuple[int, int, int, int] | None = None,
    priority: float = 0.5,
) -> "GoalSpec":
    """Build a GoalSpec from a GoalHypothesis record (from agent.memory).

    GoalHypothesis stores goals as natural-language strings (observable_target,
    disconfirming_test). The caller must supply target_color and/or target_region
    if they have been inferred from the deliberator or prior observations; if
    neither is supplied the GoalSpec is color-agnostic and region-agnostic
    (matches any non-zero cell).

    Only 'provisional' and 'active' goals may be converted; falsified goals
    raise ValueError to prevent them from entering the planner.

    Parameters
    ----------
    hypothesis : GoalHypothesis
    target_color : int | None
        ARC color category 0..15 if known; None = color-agnostic.
    target_region : tuple[int,int,int,int] | None
        (x_min, y_min, x_max, y_max) if known; None = whole grid.
    priority : float
        In (0, 1]; defaults to 0.5 for provisional goals.
    """
    # Import here to avoid top-level circular dependency
    from agent.memory import GoalHypothesis  # type: ignore[attr-defined]
    if not isinstance(hypothesis, GoalHypothesis):
        raise TypeError(
            f"goal_hypothesis_to_goal_spec expects a GoalHypothesis, "
            f"got {type(hypothesis).__name__}"
        )
    if hypothesis.status == "falsified":
        raise ValueError(
            f"Falsified goal {hypothesis.goal_id!r} must not be forwarded to the planner"
        )
    return GoalSpec(
        goal_id=hypothesis.goal_id,
        description=hypothesis.description,
        target_color=target_color,
        target_region=target_region,
        priority=priority,
        status=hypothesis.status,  # 'provisional' or 'active'
    )

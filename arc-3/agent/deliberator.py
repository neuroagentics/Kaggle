"""Deliberator interface — R11 implementation.

Defines the contract for the quantized local reasoner (4–8B open-weight model):
  - DeliberatorConfig: one selected backend, token budgets, time limits.
  - DeliberatorInput: structured observation feed (not full chat history).
  - DeliberatorOutput: validated mechanic/goal hypotheses with schema checks.
  - DeliberatorInterface: abstract base class with one bounded repair attempt.
  - StubDeliberator: deterministic contract-exercising stub (no weights loaded).

blueprint §3 and §6 contracts:
  - One selected checkpoint; never silently swap the model.
  - 1024 new tokens per call, 8192 input tokens maximum.
  - At most one call per 8 real actions (except initial orientation).
  - Feed structured observations and evidence, not the entire chat/event history.
  - On malformed output: one bounded repair attempt, then retain last-valid hypotheses.
  - The deliberator proposes hypotheses; it does not certify truth, write
    executable code into the runtime, or send game actions directly.

No model weights are loaded by this module. A real backend implementation
sub-classes DeliberatorInterface and implements _call_model().
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Callable, Sequence

from agent.model import (
    SCHEMA_VERSION,
    MechanicHypothesis,
    GoalSignal,
    SimulatorCondition,
    validate_mechanic_hypothesis_dict,
    validate_goal_signal_dict,
)
from agent.internal_world import WorldState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeliberatorConfig:
    """Static configuration for one deliberator instance.

    Fields
    ------
    model_name : str
        Name of the checkpoint (e.g. 'mistral-7b-instruct-q4').
        Set once at construction; never silently changed.
    model_version : str
        Must match WorldState.model_version and SimulatorCondition.model_version.
    max_new_tokens : int
        Per-call new-token budget. blueprint §6: initial cap 1024.
    max_input_tokens : int
        Per-call input-token budget. blueprint §6: initial cap 8192.
    min_real_actions_between_calls : int
        blueprint §6: at most one call per 8 real actions (except initial).
    max_wall_seconds : float
        Hard per-call time limit. Complements but does not replace backend
        cancellation (hard cancellation remains an open requirement, R04).
    max_repair_attempts : int
        blueprint §3: one bounded repair on malformed output; then retain
        last-valid hypotheses.
    """
    model_name: str
    model_version: str
    max_new_tokens: int = 1024
    max_input_tokens: int = 8192
    min_real_actions_between_calls: int = 8
    max_wall_seconds: float = 30.0
    max_repair_attempts: int = 1

    def __post_init__(self):
        if not self.model_name.strip():
            raise ValueError("DeliberatorConfig model_name must not be empty")
        if not self.model_version.strip():
            raise ValueError("DeliberatorConfig model_version must not be empty")
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive int")
        if not isinstance(self.max_input_tokens, int) or self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be a positive int")
        if (
            not isinstance(self.min_real_actions_between_calls, int)
            or self.min_real_actions_between_calls < 0
        ):
            raise ValueError("min_real_actions_between_calls must be a nonneg int")
        if not isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be finite and positive")
        if not isinstance(self.max_repair_attempts, int) or self.max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be a nonneg int")


# ---------------------------------------------------------------------------
# Deliberator input
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeliberatorInput:
    """Structured observation feed for one deliberator call.

    The deliberator receives structured evidence, not the full chat/event
    history. Each field is bounded and schema-typed.

    Fields
    ------
    game_id : str
    level_id : int
    recent_grids : tuple[tuple[tuple[int,...],...],...] 
        Up to 4 most recent observed grids (oldest first).
    recent_actions : tuple[tuple[int,int|None,int|None],...]
        Corresponding (action_id, x, y) tuples.
    recent_outcomes : tuple[bool, ...]
        Observed success/failure per step.
    active_belief_summaries : tuple[str, ...]
        Human-readable summaries of active/provisional beliefs (not raw IDs).
    active_goal_summaries : tuple[str, ...]
        Human-readable summaries of active/provisional goals.
    prediction_errors : tuple[float, ...]
        Per-step changed-cell error fractions (0..1) for recent steps.
    real_actions_since_last_call : int
        How many real actions have occurred since the last deliberator call.
    schema_version : str
    """
    game_id: str
    level_id: int
    recent_grids: tuple
    recent_actions: tuple
    recent_outcomes: tuple[bool, ...]
    active_belief_summaries: tuple[str, ...]
    active_goal_summaries: tuple[str, ...]
    prediction_errors: tuple[float | None, ...]
    real_actions_since_last_call: int
    schema_version: str = SCHEMA_VERSION
    current_grid: tuple = ()

    _MAX_RECENT = 4

    def __post_init__(self):
        if not self.game_id:
            raise ValueError("DeliberatorInput game_id must not be empty")
        if not isinstance(self.level_id, int) or self.level_id < 0:
            raise ValueError("DeliberatorInput level_id must be a nonneg int")
        if len(self.recent_grids) > self._MAX_RECENT:
            raise ValueError(
                f"recent_grids max {self._MAX_RECENT}, got {len(self.recent_grids)}"
            )
        if not isinstance(self.real_actions_since_last_call, int) or \
                self.real_actions_since_last_call < 0:
            raise ValueError("real_actions_since_last_call must be a nonneg int")
        for e in self.prediction_errors:
            if e is None:
                continue
            if not isfinite(e) or not 0.0 <= e <= 1.0:
                raise ValueError(
                    f"prediction_errors must be finite floats in [0,1], got {e!r}"
                )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )


# ---------------------------------------------------------------------------
# Deliberator output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeliberatorOutput:
    """Validated output from one deliberator call.

    mechanic_hypotheses and goal_signals are fully validated MechanicHypothesis
    and GoalSignal objects. Malformed raw outputs are repaired or discarded before
    reaching this dataclass.

    Fields
    ------
    mechanic_hypotheses : tuple[MechanicHypothesis, ...]
    goal_signals : tuple[GoalSignal, ...]
    raw_tokens_used : int
        Actual new tokens generated (for budget tracking).
    wall_seconds : float
        Actual wall time of the call.
    repair_applied : bool
        True when a repair attempt was needed.
    schema_version : str
    """
    mechanic_hypotheses: tuple[MechanicHypothesis, ...]
    goal_signals: tuple[GoalSignal, ...]
    raw_tokens_used: int
    wall_seconds: float
    repair_applied: bool
    schema_version: str = SCHEMA_VERSION
    input_truncated: bool = False   # True when input was trimmed to fit max_input_tokens

    def __post_init__(self):
        if not isinstance(self.raw_tokens_used, int) or self.raw_tokens_used < 0:
            raise ValueError("raw_tokens_used must be a nonneg int")
        if not isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("wall_seconds must be finite and nonneg")
        for h in self.mechanic_hypotheses:
            if not isinstance(h, MechanicHypothesis):
                raise TypeError(f"Expected MechanicHypothesis, got {type(h).__name__}")
        for g in self.goal_signals:
            if not isinstance(g, GoalSignal):
                raise TypeError(f"Expected GoalSignal, got {type(g).__name__}")

    def to_simulator_condition(
        self, game_id: str, level_id: int, model_version: str
    ) -> SimulatorCondition:
        """Package hypotheses and goals as a SimulatorCondition for the planner."""
        return SimulatorCondition(
            game_id=game_id,
            level_id=level_id,
            model_version=model_version,
            mechanic_hypotheses=self.mechanic_hypotheses,
            goal_signals=self.goal_signals,
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DeliberatorInterface(ABC):
    """Abstract base for the deliberator.

    Implementations must:
      - Sub-class this and implement _call_model().
      - Honour token budgets and wall-clock limits.
      - Never silently swap the underlying model or quantizer.
      - Return structured JSON parseable into mechanic/goal hypotheses.
      - Accept one bounded repair attempt before retaining last-valid output.

    The deliberator does not:
      - Send game actions directly.
      - Write executable code into the runtime.
      - Certify hypothesis truth.
    """

    def __init__(
        self,
        config: DeliberatorConfig,
        clock: Callable[[], float] = monotonic,
    ):
        if not isinstance(config, DeliberatorConfig):
            raise TypeError("config must be a DeliberatorConfig")
        self.config = config
        self._clock = clock
        self._real_actions_since_last_call: int = config.min_real_actions_between_calls
        self._last_valid_output: DeliberatorOutput | None = None
        self._total_calls: int = 0
        self._total_tokens: int = 0
        self._total_repairs: int = 0
        self._total_truncations: int = 0

    # ------------------------------------------------------------------
    # Input token budget
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_input_tokens(inp: "DeliberatorInput") -> int:
        """Rough token estimate for a DeliberatorInput (4 chars ≈ 1 token).

        Counts characters across all text-bearing fields: grid rows serialised
        as space-separated integers, action/outcome tuples, summary strings, and
        prediction errors. The estimate is intentionally conservative (rounds up)
        so the budget check errs toward truncation rather than overflow.
        """
        parts: list[str] = [inp.game_id, str(inp.level_id)]

        for grid in (*inp.recent_grids, inp.current_grid):
            for row in grid:
                parts.append(" ".join(str(c) for c in row))

        for act in inp.recent_actions:
            parts.append(str(act))

        for outcome in inp.recent_outcomes:
            parts.append("1" if outcome else "0")

        parts.extend(inp.active_belief_summaries)
        parts.extend(inp.active_goal_summaries)

        for e in inp.prediction_errors:
            parts.append("unknown" if e is None else f"{e:.4f}")

        total_chars = sum(len(p) for p in parts) + len(parts)  # +separators
        return max(1, (total_chars + 3) // 4)  # ceiling division

    def _check_input_budget(
        self, inp: "DeliberatorInput"
    ) -> tuple["DeliberatorInput", bool]:
        """Return (possibly truncated input, was_truncated).

        If the estimated token count fits within max_input_tokens, returns
        (inp, False) unchanged. Otherwise trims recent_grids and summaries
        from the front (oldest first) until the estimate fits, then returns
        the trimmed input and True.

        Truncation policy (oldest-first, in order):
          1. Drop oldest grids/actions/outcomes (keep at least 1 grid).
          2. Drop oldest belief summaries (keep at least 0).
          3. Drop oldest goal summaries (keep at least 0).
          4. Drop oldest prediction errors (keep at least 0).

        If the input still overflows after all optional fields are dropped,
        the minimum possible input (1 grid, no summaries) is returned with
        truncated=True. A subclass may override this method for a different
        policy.
        """
        from dataclasses import replace as _replace

        if self.estimate_input_tokens(inp) <= self.config.max_input_tokens:
            return inp, False

        # Work on mutable copies
        grids = list(inp.recent_grids)
        actions = list(inp.recent_actions)
        outcomes = list(inp.recent_outcomes)
        beliefs = list(inp.active_belief_summaries)
        goals = list(inp.active_goal_summaries)
        errors = list(inp.prediction_errors)

        def _current() -> "DeliberatorInput":
            return _replace(
                inp,
                recent_grids=tuple(grids),
                recent_actions=tuple(actions),
                recent_outcomes=tuple(outcomes),
                active_belief_summaries=tuple(beliefs),
                active_goal_summaries=tuple(goals),
                prediction_errors=tuple(errors),
                current_grid=inp.current_grid,
            )

        # Drop oldest grids (keep at least 1)
        while len(grids) > 1 and self.estimate_input_tokens(_current()) > self.config.max_input_tokens:
            grids.pop(0)
            if actions:
                actions.pop(0)
            if outcomes:
                outcomes.pop(0)
            if errors:
                errors.pop(0)

        # Drop oldest belief summaries
        while beliefs and self.estimate_input_tokens(_current()) > self.config.max_input_tokens:
            beliefs.pop(0)

        # Drop oldest goal summaries
        while goals and self.estimate_input_tokens(_current()) > self.config.max_input_tokens:
            goals.pop(0)

        # Drop oldest prediction errors
        while errors and self.estimate_input_tokens(_current()) > self.config.max_input_tokens:
            errors.pop(0)

        return _current(), True

    # ------------------------------------------------------------------
    # Call-frequency gate
    # ------------------------------------------------------------------

    def should_call(
        self,
        real_actions_since_last: int,
        *,
        is_initial: bool = False,
        repeated_mismatch: bool = False,
        stalled_progress: bool = False,
    ) -> bool:
        """True when a deliberator call is warranted.

        blueprint §6: call on game initialization, meaningful new level/context,
        repeated model mismatch, or stalled progress.
        """
        if is_initial:
            return True
        if repeated_mismatch or stalled_progress:
            return True
        return real_actions_since_last >= self.config.min_real_actions_between_calls

    # ------------------------------------------------------------------
    # Main call interface
    # ------------------------------------------------------------------

    def call(
        self,
        inp: DeliberatorInput,
        *,
        deadline: float | None = None,
    ) -> DeliberatorOutput:
        """Run one bounded deliberator call.

        If the model returns malformed output, one repair attempt is made.
        If the repair also fails, the last-valid output is returned (blueprint §3).
        If no prior valid output exists, an empty output is returned.

        The deadline argument allows the controller to pass a wall-clock limit
        independent of config.max_wall_seconds.
        """
        if not isinstance(inp, DeliberatorInput):
            raise TypeError("inp must be a DeliberatorInput")

        # Enforce input token budget before building the prompt.
        # Truncation is oldest-first; the flag is forwarded to the output record.
        inp, input_truncated = self._check_input_budget(inp)
        if input_truncated:
            self._total_truncations += 1

        effective_deadline = min(
            self._clock() + self.config.max_wall_seconds,
            deadline if deadline is not None else float("inf"),
        )

        t0 = self._clock()
        raw_output: str | None = None
        repair_applied = False

        try:
            raw_output = self._call_model(inp, deadline=effective_deadline)
        except Exception as exc:
            return self._fallback(
                f"model call raised {type(exc).__name__}: {exc}", t0, repair_applied=False,
                input_truncated=input_truncated,
            )

        if self._clock() >= effective_deadline:
            return self._fallback(
                "deadline exceeded after model call", t0, repair_applied=False,
                input_truncated=input_truncated,
            )

        # Parse
        parsed, parse_error = self._parse_output(raw_output)
        if parsed is None:
            # One repair attempt
            if self.config.max_repair_attempts >= 1:
                repair_applied = True
                self._total_repairs += 1
                repaired_raw = self._repair(inp, raw_output, parse_error, deadline=effective_deadline)
                if repaired_raw is not None:
                    parsed, parse_error = self._parse_output(repaired_raw)
            if parsed is None:
                return self._fallback(
                    f"parse failed after repair: {parse_error}", t0, repair_applied=repair_applied,
                    input_truncated=input_truncated,
                )

        tokens_used = self._estimate_tokens(raw_output or "")
        wall = self._clock() - t0
        output = DeliberatorOutput(
            mechanic_hypotheses=parsed[0],
            goal_signals=parsed[1],
            raw_tokens_used=tokens_used,
            wall_seconds=wall,
            repair_applied=repair_applied,
            input_truncated=input_truncated,
        )
        self._last_valid_output = output
        self._total_calls += 1
        self._total_tokens += tokens_used
        return output

    # ------------------------------------------------------------------
    # Abstract: model backend
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_model(self, inp: DeliberatorInput, *, deadline: float) -> str:
        """Call the underlying model and return raw JSON string output.

        Must respect deadline (wall-clock seconds from monotonic clock).
        Returns a JSON string expected to contain 'mechanic_hypotheses' and
        'goal_signals' arrays.
        """
        ...

    # ------------------------------------------------------------------
    # Optional: repair
    # ------------------------------------------------------------------

    def _repair(
        self,
        inp: DeliberatorInput,
        raw: str,
        error: str,
        *,
        deadline: float,
    ) -> str | None:
        """Attempt one repair pass on malformed output.

        Default implementation: re-call the model with a repair prompt.
        Subclasses may override with a more targeted fix.
        """
        return None  # Base: no repair. Subclasses should implement.

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------

    def _parse_output(
        self, raw: str
    ) -> tuple[tuple[tuple, tuple] | None, str]:
        """Parse raw JSON into (hypotheses, goals). Returns (None, error) on failure."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            return None, f"JSON decode error: {exc}"

        if not isinstance(data, dict):
            return None, f"Expected JSON object, got {type(data).__name__}"

        hypotheses = []
        for i, raw_h in enumerate(data.get("mechanic_hypotheses", [])):
            try:
                hypotheses.append(validate_mechanic_hypothesis_dict(raw_h))
            except ValueError as exc:
                return None, f"mechanic_hypotheses[{i}] invalid: {exc}"

        goals = []
        for i, raw_g in enumerate(data.get("goal_signals", [])):
            try:
                goals.append(validate_goal_signal_dict(raw_g))
            except ValueError as exc:
                return None, f"goal_signals[{i}] invalid: {exc}"

        return (tuple(hypotheses), tuple(goals)), ""

    # ------------------------------------------------------------------
    # Fallback to last-valid output
    # ------------------------------------------------------------------

    def _fallback(
        self, reason: str, t0: float, *, repair_applied: bool, input_truncated: bool = False
    ) -> DeliberatorOutput:
        """Return last-valid output or an empty output if none exists."""
        wall = self._clock() - t0
        if self._last_valid_output is not None:
            # Wrap in a new output record with updated wall time
            return DeliberatorOutput(
                mechanic_hypotheses=self._last_valid_output.mechanic_hypotheses,
                goal_signals=self._last_valid_output.goal_signals,
                raw_tokens_used=0,
                wall_seconds=wall,
                repair_applied=repair_applied,
                input_truncated=input_truncated,
            )
        return DeliberatorOutput(
            mechanic_hypotheses=(),
            goal_signals=(),
            raw_tokens_used=0,
            wall_seconds=wall,
            repair_applied=repair_applied,
            input_truncated=input_truncated,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        return max(0, len(text) // 4)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_repairs": self._total_repairs,
            "total_truncations": self._total_truncations,
            "last_valid_hypotheses": (
                len(self._last_valid_output.mechanic_hypotheses)
                if self._last_valid_output else 0
            ),
            "last_valid_goals": (
                len(self._last_valid_output.goal_signals)
                if self._last_valid_output else 0
            ),
        }


# ---------------------------------------------------------------------------
# Stub deliberator (contract-exercising, no model weights)
# ---------------------------------------------------------------------------

class StubDeliberator(DeliberatorInterface):
    """Deterministic stub for contract and integration tests.

    Returns a fixed set of valid hypotheses and goals derived from a
    supplied responses list. Does not load any model weights.
    """

    def __init__(
        self,
        config: DeliberatorConfig,
        responses: list[str] | None = None,
        clock: Callable[[], float] = monotonic,
    ):
        super().__init__(config, clock)
        self._responses: list[str] = responses or []
        self._call_count: int = 0
        self._should_fail: bool = False

    def set_fail_next(self) -> None:
        """Make the next call return unparseable output (for repair tests)."""
        self._should_fail = True

    def _call_model(self, inp: DeliberatorInput, *, deadline: float) -> str:
        if self._should_fail:
            self._should_fail = False
            return "this is not valid json {"
        if self._responses:
            idx = self._call_count % len(self._responses)
            self._call_count += 1
            return self._responses[idx]
        # Default: empty valid response
        self._call_count += 1
        return json.dumps({"mechanic_hypotheses": [], "goal_signals": []})

    def _repair(
        self,
        inp: DeliberatorInput,
        raw: str,
        error: str,
        *,
        deadline: float,
    ) -> str | None:
        """Return the next scheduled valid response as the repaired output.

        This lets tests verify that the repair path produces a usable result
        rather than always falling back to empty output.
        """
        if self._clock() >= deadline:
            return None
        if self._responses:
            idx = self._call_count % len(self._responses)
            self._call_count += 1
            return self._responses[idx]
        # No responses configured: return a valid empty JSON
        return json.dumps({"mechanic_hypotheses": [], "goal_signals": []})

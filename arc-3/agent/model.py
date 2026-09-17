"""Typed model interface contracts — R07 implementation.

Defines the schema for deliberator hypothesis → simulator conditioning, plus
versioned checkpoint protocol and structured output validation.

No trained weights are loaded here. This module specifies:
  - Typed mechanic/goal hypothesis structures (MechanicHypothesis, GoalSignal)
  - The SimulatorCondition bundle that carries hypotheses into the Dynamics layer
  - The HypothesisConditionedDynamics protocol that extends the base Dynamics protocol
  - Checkpoint version/device contracts
  - Schema validation and malformed-output rejection

Evidence ID references must resolve in the calling WorkingMemory. The deliberator
produces structured hypotheses; the simulator accepts them as conditioning signals.
One bounded repair is allowed on schema errors; then last-valid hypotheses are
retained (blueprint §3).

No gradient operations, weight loading, or inference occur in this module.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, Sequence

from agent.internal_world import Action, Dynamics, Prediction, WorldState


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
_VALID_ID = re.compile(r"^[0-9a-f]{32}$")  # hex UUID without dashes


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _VALID_ID.match(value):
        raise ValueError(f"{name} must be a 32-hex-char ID, got {value!r}")


# ---------------------------------------------------------------------------
# Mechanic hypothesis (deliberator output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MechanicHypothesis:
    """A structured, falsifiable hypothesis about a game mechanic.

    Produced by the deliberator; consumed by the simulator as conditioning.
    All evidence IDs must resolve in the active WorkingMemory before acceptance.

    Fields
    ------
    hypothesis_id : str
        Unique 32-hex ID assigned by the deliberator or WorkingMemory.
    belief_id : str
        ID of the MechanicBelief record in WorkingMemory that backs this hypothesis.
    mechanic_type : str
        One of: 'movement', 'interaction', 'transformation', 'constraint',
                'goal_condition', 'spawn', 'environmental', 'unknown'.
    description : str
        Human-readable description. Must not be empty.
    object_type : str | None
        Optionally identifies the affected object class (e.g. 'avatar', 'block').
    direction_vector : tuple[int, int] | None
        (dx, dy) for movement mechanics. Values in {-1, 0, 1}.
    affected_colors : tuple[int, ...] | None
        ARC color indices (0..15) involved in this mechanic.
    precondition_tags : tuple[str, ...]
        Observable conditions under which this mechanic applies.
    effect_tags : tuple[str, ...]
        Observable effects produced when this mechanic fires.
    supporting_ids : tuple[str, ...]
        EpisodeRecord IDs that support this hypothesis.
    confidence : float
        In [0, 1]. Calibrated by the deliberator; 0 = pure guess.
    schema_version : str
    """
    hypothesis_id: str
    belief_id: str
    mechanic_type: str
    description: str
    object_type: str | None
    direction_vector: tuple[int, int] | None
    affected_colors: tuple[int, ...] | None
    precondition_tags: tuple[str, ...]
    effect_tags: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    confidence: float
    schema_version: str = SCHEMA_VERSION

    _MECHANIC_TYPES = frozenset({
        "movement", "interaction", "transformation", "constraint",
        "goal_condition", "spawn", "environmental", "unknown",
    })

    def __post_init__(self):
        _validate_id(self.hypothesis_id, "hypothesis_id")
        _validate_id(self.belief_id, "belief_id")
        if self.mechanic_type not in self._MECHANIC_TYPES:
            raise ValueError(
                f"mechanic_type {self.mechanic_type!r} not in {sorted(self._MECHANIC_TYPES)}"
            )
        if not self.description.strip():
            raise ValueError("MechanicHypothesis description must not be empty")
        if self.direction_vector is not None:
            dx, dy = self.direction_vector
            if dx not in (-1, 0, 1) or dy not in (-1, 0, 1):
                raise ValueError(
                    f"direction_vector components must be in {{-1,0,1}}, got ({dx},{dy})"
                )
            if dx == 0 and dy == 0:
                raise ValueError("direction_vector (0,0) is not a direction")
        if self.affected_colors is not None:
            for c in self.affected_colors:
                if not isinstance(c, int) or not 0 <= c <= 15:
                    raise ValueError(f"affected_colors must be ARC categories 0..15, got {c!r}")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )


# ---------------------------------------------------------------------------
# Goal signal (deliberator output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalSignal:
    """A goal conditioning signal derived from a GoalHypothesis.

    Carries observable target information into the planner's utility function.
    Does not embed hidden state or fabricated labels.

    Fields
    ------
    goal_id : str
        ID of the GoalHypothesis record in WorkingMemory.
    description : str
        Human-readable goal description.
    target_color : int | None
        ARC color category (0..15) associated with the goal object, if known.
    target_region : tuple[int, int, int, int] | None
        (x_min, y_min, x_max, y_max) in grid coordinates, if known.
    priority : float
        In (0, 1]. Higher priority goals weight the utility function more heavily.
    status : str
        One of: 'provisional', 'active'. Falsified goals must not be forwarded.
    schema_version : str
    """
    goal_id: str
    description: str
    target_color: int | None
    target_region: tuple[int, int, int, int] | None
    priority: float
    status: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        _validate_id(self.goal_id, "goal_id")
        if not self.description.strip():
            raise ValueError("GoalSignal description must not be empty")
        if self.target_color is not None:
            if not isinstance(self.target_color, int) or not 0 <= self.target_color <= 15:
                raise ValueError(f"target_color must be 0..15, got {self.target_color!r}")
        if self.target_region is not None:
            x0, y0, x1, y1 = self.target_region
            if not (0 <= x0 <= x1 <= 63 and 0 <= y0 <= y1 <= 63):
                raise ValueError(
                    f"target_region must satisfy 0<=x0<=x1<=63 and 0<=y0<=y1<=63, "
                    f"got {self.target_region!r}"
                )
        if not isfinite(self.priority) or not 0.0 < self.priority <= 1.0:
            raise ValueError(f"priority must be in (0,1], got {self.priority!r}")
        if self.status not in ("provisional", "active"):
            raise ValueError(
                f"GoalSignal status must be 'provisional' or 'active', got {self.status!r}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )


# ---------------------------------------------------------------------------
# SimulatorCondition bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulatorCondition:
    """Bundles mechanic hypotheses and goal signals for simulator conditioning.

    Passed to HypothesisConditionedDynamics.predict_conditioned() instead of
    or alongside the base Dynamics.predict() path. The simulator uses these
    to bias its latent dynamics toward the believed mechanics/goals.

    Freezing is enforced: conditions must not change during a planning pass.
    """
    game_id: str
    level_id: int
    model_version: str
    mechanic_hypotheses: tuple[MechanicHypothesis, ...]
    goal_signals: tuple[GoalSignal, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not self.game_id:
            raise ValueError("SimulatorCondition game_id must not be empty")
        if not isinstance(self.level_id, int) or self.level_id < 0:
            raise ValueError("SimulatorCondition level_id must be a nonnegative int")
        if not self.model_version:
            raise ValueError("SimulatorCondition model_version must not be empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        # Detect duplicate hypothesis or goal IDs
        hyp_ids = [h.hypothesis_id for h in self.mechanic_hypotheses]
        if len(hyp_ids) != len(set(hyp_ids)):
            raise ValueError("Duplicate hypothesis_id in SimulatorCondition")
        goal_ids = [g.goal_id for g in self.goal_signals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("Duplicate goal_id in SimulatorCondition")


# ---------------------------------------------------------------------------
# Hypothesis-conditioned dynamics protocol
# ---------------------------------------------------------------------------

class HypothesisConditionedDynamics(Dynamics, Protocol):
    """Extends the base Dynamics protocol with hypothesis conditioning.

    Implementations must satisfy both the base predict() contract and the
    predict_conditioned() extension. The base predict() path remains available
    for unconditioned or fallback operation.

    A trained simulator implementing this protocol should:
    - Accept a SimulatorCondition and use it to bias latent dynamics
    - Never modify the condition during a planning pass (weights are frozen)
    - Advance from its own predicted state without reading a new real frame
    - Return predictions with schema_version and model_version that match
      the SimulatorCondition's model_version
    """
    def predict_conditioned(
        self, state: WorldState, action: Action, condition: SimulatorCondition
    ) -> Prediction:
        """Return a prediction biased by the supplied hypothesis condition.

        Contracts (all checked by validate_prediction_against_condition):
        - prediction.state.game == condition.game_id
        - prediction.state.model_version == condition.model_version
        - prediction is finite and in range
        """
        ...


# ---------------------------------------------------------------------------
# Checkpoint version contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointDescriptor:
    """Version + device contract for a loaded model checkpoint.

    Preflight must instantiate both real model roles, verify artifact hashes,
    run nontrivial action-conditioned multi-step predictions, and confirm device.
    This dataclass is the required output of a successful checkpoint load.
    """
    model_name: str          # e.g. 'simulator-v1', 'deliberator-mistral-7b-q4'
    model_version: str       # matches WorldState.model_version throughout a run
    role: str                # 'simulator' or 'deliberator'
    parameter_count: int     # approximate; 0 if unknown
    quantization: str | None # e.g. 'int4', 'fp32', None
    device: str              # e.g. 'cuda:0', 'cpu'
    artifact_hash: str       # SHA-256 of the weights file
    schema_version: str = SCHEMA_VERSION

    _ROLES = frozenset({"simulator", "deliberator"})

    def __post_init__(self):
        if self.role not in self._ROLES:
            raise ValueError(f"CheckpointDescriptor role must be one of {sorted(self._ROLES)}")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if not isinstance(self.parameter_count, int) or self.parameter_count < 0:
            raise ValueError("parameter_count must be a nonneg int")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.artifact_hash):
            raise ValueError("artifact_hash must be a 64-hex-char SHA-256 digest")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------

def validate_mechanic_hypothesis_dict(raw: object) -> MechanicHypothesis:
    """Parse and validate a dict-form hypothesis (e.g. from deliberator JSON output).

    Raises ValueError on any schema or value violation.
    One bounded repair attempt is the caller's responsibility.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict for MechanicHypothesis, got {type(raw).__name__}")
    try:
        dv = raw.get("direction_vector")
        if dv is not None:
            dv = tuple(dv)
        ac = raw.get("affected_colors")
        if ac is not None:
            ac = tuple(ac)
        return MechanicHypothesis(
            hypothesis_id=raw["hypothesis_id"],
            belief_id=raw["belief_id"],
            mechanic_type=raw["mechanic_type"],
            description=raw["description"],
            object_type=raw.get("object_type"),
            direction_vector=dv,
            affected_colors=ac,
            precondition_tags=tuple(raw.get("precondition_tags", [])),
            effect_tags=tuple(raw.get("effect_tags", [])),
            supporting_ids=tuple(raw.get("supporting_ids", [])),
            confidence=float(raw["confidence"]),
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
        )
    except KeyError as exc:
        raise ValueError(f"MechanicHypothesis missing required field: {exc}") from exc


def validate_goal_signal_dict(raw: object) -> GoalSignal:
    """Parse and validate a dict-form GoalSignal."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict for GoalSignal, got {type(raw).__name__}")
    try:
        tr = raw.get("target_region")
        if tr is not None:
            tr = tuple(tr)
        return GoalSignal(
            goal_id=raw["goal_id"],
            description=raw["description"],
            target_color=raw.get("target_color"),
            target_region=tr,
            priority=float(raw["priority"]),
            status=raw["status"],
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
        )
    except KeyError as exc:
        raise ValueError(f"GoalSignal missing required field: {exc}") from exc


def validate_prediction_against_condition(
    prediction: Prediction,
    condition: SimulatorCondition,
) -> None:
    """Assert that a prediction is consistent with the SimulatorCondition that produced it.

    Raises ValueError on any mismatch. Called by the controller after each
    conditioned prediction step.
    """
    if prediction.state.game != condition.game_id:
        raise ValueError(
            f"Prediction game {prediction.state.game!r} != "
            f"condition game_id {condition.game_id!r}"
        )
    if prediction.state.model_version != condition.model_version:
        raise ValueError(
            f"Prediction model_version {prediction.state.model_version!r} != "
            f"condition model_version {condition.model_version!r}"
        )
    if not isfinite(prediction.utility):
        raise ValueError("Prediction utility is not finite")
    if not 0.0 <= prediction.uncertainty <= 1.0:
        raise ValueError(f"Prediction uncertainty {prediction.uncertainty} not in [0,1]")
    if not 0.0 <= prediction.risk <= 1.0:
        raise ValueError(f"Prediction risk {prediction.risk} not in [0,1]")

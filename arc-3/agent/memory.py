"""Bounded game-local working memory — R06 implementation.

Implements the five record types and memory constraints from ARC3_LEAN_BLUEPRINT.md §5:
  - EpisodeRecord      (cap 512, real before/action/predicted/actual)
  - MechanicBelief     (cap 128, versioned hypothesis with supporting/contradicting IDs)
  - GoalHypothesis     (cap 16, observable target with disconfirming test)
  - ProcedureRecord    (cap 64, preconditions/steps/milestones/exceptions)
  - ImaginedBranch     (current plan only, cleared on plan replacement)

Overall ceiling: 32 MiB. Grids stored as bytes, not Python int-per-cell.
Eviction: unreferenced low-value episodes first; evicting an episode that supports
a belief/procedure invalidates the dependent record rather than leaving a dangling
verified reference.

Online learning contract: posterior/belief updates are immediate; adapter updates
are bounded, version-tracked, and rolled back on regression. This module does not
perform gradient updates — it provides the evidence store for the adapter layer.

No graph database, vector database, embeddings service, or geometry library.
Retrieval is exact contextual filtering plus a deterministic relevance score.
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from typing import Sequence


# ---------------------------------------------------------------------------
# Sizing constants from blueprint §5
# ---------------------------------------------------------------------------
EPISODE_CAP = 512
MECHANIC_CAP = 128
GOAL_CAP = 16
PROCEDURE_CAP = 64
MEMORY_CEILING_BYTES = 32 * 1024 * 1024  # 32 MiB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex


def _grid_to_bytes(grid: tuple[tuple[int, ...], ...]) -> bytes:
    """Pack a uint8 ARC grid to bytes: 2-byte height, 2-byte width, then cells."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    buf = bytearray(4 + h * w)
    buf[0] = (h >> 8) & 0xFF
    buf[1] = h & 0xFF
    buf[2] = (w >> 8) & 0xFF
    buf[3] = w & 0xFF
    idx = 4
    for row in grid:
        for cell in row:
            buf[idx] = cell & 0xFF
            idx += 1
    return bytes(buf)


def _bytes_to_grid(data: bytes) -> tuple[tuple[int, ...], ...]:
    h = (data[0] << 8) | data[1]
    w = (data[2] << 8) | data[3]
    idx = 4
    rows = []
    for _ in range(h):
        rows.append(tuple(data[idx + c] for c in range(w)))
        idx += w
    return tuple(rows)


def _grid_bytes(grid: tuple[tuple[int, ...], ...]) -> int:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return 4 + h * w


# ---------------------------------------------------------------------------
# Evidence record types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeRecord:
    """One real before→action→predicted→observed transition.

    Grids are stored as bytes (uint8 + shape header) to respect the 32 MiB cap.
    All IDs are assigned by WorkingMemory; callers supply content only.
    """
    record_id: str
    game_id: str
    level_id: int
    observation_id: str          # ID of the observation that triggered this episode
    model_version: str
    action_id: int
    action_x: int | None
    action_y: int | None
    before_grid_bytes: bytes     # _grid_to_bytes encoding
    predicted_grid_bytes: bytes
    observed_grid_bytes: bytes
    changed_cell_error: int
    actual_success: bool
    discrepancy_score: float     # 0.0 = perfect prediction; -1.0 = no prediction
    timestamp: float
    prediction_available: bool = True  # False for probe steps with no saved prediction

    def before_grid(self) -> tuple[tuple[int, ...], ...]:
        return _bytes_to_grid(self.before_grid_bytes)

    def predicted_grid(self) -> tuple[tuple[int, ...], ...]:
        return _bytes_to_grid(self.predicted_grid_bytes)

    def observed_grid(self) -> tuple[tuple[int, ...], ...]:
        return _bytes_to_grid(self.observed_grid_bytes)

    def byte_size(self) -> int:
        return (
            len(self.before_grid_bytes)
            + len(self.predicted_grid_bytes)
            + len(self.observed_grid_bytes)
            + 256  # overhead estimate for fixed fields
        )


@dataclass(frozen=True)
class MechanicBelief:
    """A versioned, falsifiable hypothesis about a game mechanic.

    Supporting and contradicting IDs reference EpisodeRecord.record_id values.
    A contradiction supersedes rather than erases: the revision_of chain is preserved.
    A belief with no supporting IDs (all evicted) is marked status='unsupported'.
    """
    belief_id: str
    game_id: str
    level_context: int | None    # None = game-wide
    hypothesis: str              # Human-readable description
    mechanic_tags: tuple[str, ...]
    object_relation_tags: tuple[str, ...]
    goal_tags: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    contradicting_ids: tuple[str, ...]
    status: str                  # 'active', 'contradicted', 'unsupported', 'provisional'
    revision: int                # increments on each update; starts at 1
    revision_of: str | None      # belief_id of the belief this supersedes
    schema_version: str
    timestamp: float

    def __post_init__(self):
        if self.status not in ("active", "contradicted", "unsupported", "provisional"):
            raise ValueError(f"Invalid belief status: {self.status!r}")
        if self.revision < 1:
            raise ValueError("Belief revision must be >= 1")
        if not self.hypothesis.strip():
            raise ValueError("Belief hypothesis must not be empty")

    def relevance_score(self, mechanic_tags: Sequence[str],
                        object_tags: Sequence[str],
                        goal_tags: Sequence[str],
                        recency_weight: float = 0.1) -> float:
        """Deterministic relevance for retrieval ranking (no embeddings)."""
        m = sum(t in self.mechanic_tags for t in mechanic_tags)
        o = sum(t in self.object_relation_tags for t in object_tags)
        g = sum(t in self.goal_tags for t in goal_tags)
        support = len(self.supporting_ids)
        # Contradicted beliefs still score (may be informative), but discounted
        status_weight = 0.3 if self.status == "contradicted" else 1.0
        return (m + o + g + 0.5 * support + recency_weight * self.timestamp) * status_weight


@dataclass(frozen=True)
class GoalHypothesis:
    """A falsifiable goal: an observable target state and a disconfirming test.

    One success creates a provisional goal; a second independent applicable
    success permits promotion to 'active'. Failed disconfirming test revokes.
    """
    goal_id: str
    game_id: str
    level_context: int | None
    description: str
    observable_target: str       # Observable condition that constitutes the goal
    disconfirming_test: str      # Observable condition that would falsify this goal
    supporting_ids: tuple[str, ...]
    status: str                  # 'provisional', 'active', 'falsified'
    schema_version: str
    timestamp: float

    def __post_init__(self):
        if self.status not in ("provisional", "active", "falsified"):
            raise ValueError(f"Invalid goal status: {self.status!r}")
        if not self.observable_target.strip():
            raise ValueError("Goal observable_target must not be empty")
        if not self.disconfirming_test.strip():
            raise ValueError("Goal disconfirming_test must not be empty")


@dataclass(frozen=True)
class ProcedureRecord:
    """A contextual procedure: preconditions, steps, milestones, exceptions.

    One independent success -> provisional. Two independent successes -> active.
    A failed precondition or milestone stops execution and triggers revision.
    Chaining requires compatible postconditions/preconditions, not similar labels.
    """
    procedure_id: str
    game_id: str
    level_context: int | None
    name: str
    preconditions: tuple[str, ...]
    parameters: tuple[str, ...]    # Object-relative; no hardcoded coordinates
    steps: tuple[str, ...]
    milestones: tuple[str, ...]
    postconditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    status: str                    # 'provisional', 'active', 'invalidated'
    schema_version: str
    timestamp: float

    def __post_init__(self):
        if self.status not in ("provisional", "active", "invalidated"):
            raise ValueError(f"Invalid procedure status: {self.status!r}")
        if not self.steps:
            raise ValueError("Procedure must have at least one step")

    def chains_with(self, successor: "ProcedureRecord") -> bool:
        """True only when this procedure's postconditions cover successor's preconditions."""
        if not self.postconditions or not successor.preconditions:
            return False
        return all(pre in self.postconditions for pre in successor.preconditions)


@dataclass(frozen=True)
class ImaginedBranch:
    """A predicted state sequence from the current planning pass.

    Explicitly unverified. Cleared and replaced on each new plan. Never
    persisted as real evidence or used for belief promotion.
    """
    branch_id: str
    game_id: str
    parent_branch_id: str | None
    action_sequence: tuple[int, ...]  # action IDs in order
    predicted_grids_bytes: tuple[bytes, ...]
    utility_sequence: tuple[float, ...]
    risk_sequence: tuple[float, ...]
    uncertainty_sequence: tuple[float, ...]
    terminal: bool
    schema_version: str

    def __post_init__(self):
        n = len(self.action_sequence)
        if len(self.predicted_grids_bytes) != n:
            raise ValueError("Branch grids count must match action count")
        if len(self.utility_sequence) != n:
            raise ValueError("Branch utility sequence length must match action count")
        if len(self.risk_sequence) != n:
            raise ValueError("Branch risk sequence length must match action count")
        if len(self.uncertainty_sequence) != n:
            raise ValueError("Branch uncertainty sequence length must match action count")
        for v in (*self.utility_sequence, *self.risk_sequence, *self.uncertainty_sequence):
            if not isfinite(v):
                raise ValueError("Branch scores must be finite")

    def byte_size(self) -> int:
        return sum(len(b) for b in self.predicted_grids_bytes) + 512


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Bounded game-local working memory.

    Enforces record counts and a 32 MiB byte ceiling. All mutations go through
    this class so eviction, byte accounting, and lineage invalidation are atomic.

    Isolation contract: a new WorkingMemory must be constructed for each new game.
    Carry-over across games is explicitly prohibited by the blueprint.
    """

    def __init__(
        self,
        game_id: str,
        schema_version: str = "1.0",
        episode_cap: int = EPISODE_CAP,
        mechanic_cap: int = MECHANIC_CAP,
        goal_cap: int = GOAL_CAP,
        procedure_cap: int = PROCEDURE_CAP,
        ceiling_bytes: int = MEMORY_CEILING_BYTES,
        clock: object = None,
    ):
        if not game_id:
            raise ValueError("game_id must not be empty")
        if any(c < 1 for c in (episode_cap, mechanic_cap, goal_cap, procedure_cap)):
            raise ValueError("All record caps must be positive")
        if ceiling_bytes < 1024:
            raise ValueError("Ceiling must be at least 1 KiB")

        self.game_id = game_id
        self.schema_version = schema_version
        self.episode_cap = episode_cap
        self.mechanic_cap = mechanic_cap
        self.goal_cap = goal_cap
        self.procedure_cap = procedure_cap
        self.ceiling_bytes = ceiling_bytes
        self._clock: object = clock if clock is not None else monotonic

        # Ordered dicts preserve insertion order for eviction (oldest first)
        self._episodes: dict[str, EpisodeRecord] = {}
        self._beliefs: dict[str, MechanicBelief] = {}
        self._goals: dict[str, GoalHypothesis] = {}
        self._procedures: dict[str, ProcedureRecord] = {}
        self._branches: dict[str, ImaginedBranch] = {}

        self._bytes_used: int = 0

    # -----------------------------------------------------------------------
    # Byte accounting
    # -----------------------------------------------------------------------

    def _now(self) -> float:
        t = self._clock()
        return t if isinstance(t, float) else float(t)

    def bytes_used(self) -> int:
        return self._bytes_used

    def _recompute_bytes(self) -> int:
        total = 0
        for ep in self._episodes.values():
            total += ep.byte_size()
        for b in self._branches.values():
            total += b.byte_size()
        # Fixed-size records: rough estimate per record
        total += len(self._beliefs) * 512
        total += len(self._goals) * 256
        total += len(self._procedures) * 512
        return total

    # -----------------------------------------------------------------------
    # Eviction
    # -----------------------------------------------------------------------

    def _evict_to_fit(self, needed: int) -> None:
        """Evict lowest-value episodes until the byte budget has room.

        Evicting an episode that supports a belief, goal, or procedure marks
        that record 'unsupported' rather than leaving a dangling verified reference.
        Only unreferenced or low-value episodes are evicted first.
        """
        if self._bytes_used + needed <= self.ceiling_bytes:
            return

        # Identify which episode IDs are referenced by beliefs/goals/procedures
        referenced: set[str] = set()
        for b in self._beliefs.values():
            referenced.update(b.supporting_ids)
            referenced.update(b.contradicting_ids)
        for g in self._goals.values():
            referenced.update(g.supporting_ids)
        for p in self._procedures.values():
            referenced.update(p.supporting_ids)

        # Score: unreferenced + high error first (low value)
        def eviction_priority(ep: EpisodeRecord) -> tuple:
            is_ref = ep.record_id in referenced
            return (is_ref, ep.discrepancy_score, -ep.timestamp)

        candidates = sorted(self._episodes.values(), key=eviction_priority)

        freed = 0
        evicted_ids: set[str] = set()
        for ep in candidates:
            if self._bytes_used + needed - freed <= self.ceiling_bytes:
                break
            freed += ep.byte_size()
            evicted_ids.add(ep.record_id)

        if not evicted_ids:
            return

        for eid in evicted_ids:
            del self._episodes[eid]
        self._bytes_used = max(0, self._bytes_used - freed)

        # Invalidate beliefs/goals/procedures that now have no supporting evidence
        self._invalidate_unsupported(evicted_ids)

    def _invalidate_unsupported(self, evicted_ids: set[str]) -> None:
        """Mark dependent records unsupported/invalidated when their evidence is evicted."""
        for bid, belief in list(self._beliefs.items()):
            remaining = tuple(s for s in belief.supporting_ids if s not in evicted_ids)
            if remaining != belief.supporting_ids:
                new_status = "unsupported" if not remaining else belief.status
                self._beliefs[bid] = replace(
                    belief,
                    supporting_ids=remaining,
                    status=new_status,
                    timestamp=self._now(),
                )

        for gid, goal in list(self._goals.items()):
            remaining = tuple(s for s in goal.supporting_ids if s not in evicted_ids)
            if remaining != goal.supporting_ids:
                new_status = "provisional" if not remaining else goal.status
                self._goals[gid] = replace(
                    goal,
                    supporting_ids=remaining,
                    status=new_status,
                    timestamp=self._now(),
                )

        for pid, proc in list(self._procedures.items()):
            remaining = tuple(s for s in proc.supporting_ids if s not in evicted_ids)
            if remaining != proc.supporting_ids:
                new_status = "invalidated" if not remaining else proc.status
                self._procedures[pid] = replace(
                    proc,
                    supporting_ids=remaining,
                    status=new_status,
                    timestamp=self._now(),
                )

    # -----------------------------------------------------------------------
    # Episode records
    # -----------------------------------------------------------------------

    def add_episode(
        self,
        *,
        level_id: int,
        observation_id: str,
        model_version: str,
        action_id: int,
        action_x: int | None,
        action_y: int | None,
        before_grid: tuple[tuple[int, ...], ...],
        predicted_grid: tuple[tuple[int, ...], ...],
        observed_grid: tuple[tuple[int, ...], ...],
        actual_success: bool,
        prediction_available: bool = True,
    ) -> EpisodeRecord:
        """Record a real observed transition. Returns the stored record.

        When prediction_available is False (e.g. an uncertainty-probe step that
        made no saved prediction), error/discrepancy are recorded as the sentinel
        -1 rather than fabricated as zero — so honest metrics never count a probe
        as a perfect prediction.
        """
        if type(actual_success) is not bool:
            raise ValueError("actual_success must be a real bool, not a truthy value")
        if not model_version:
            raise ValueError("model_version must not be empty")
        if not observation_id:
            raise ValueError("observation_id must not be empty")

        before_bytes = _grid_to_bytes(before_grid)
        predicted_bytes = _grid_to_bytes(predicted_grid)
        observed_bytes = _grid_to_bytes(observed_grid)

        if not prediction_available:
            # No genuine prediction was made — do not manufacture zero error.
            error = -1
            discrepancy = -1.0
        else:
            # Compute changed-cell error
            if len(predicted_grid) != len(observed_grid) or (
                predicted_grid and len(predicted_grid[0]) != len(observed_grid[0])
            ):
                error = max(
                    sum(len(r) for r in predicted_grid),
                    sum(len(r) for r in observed_grid),
                )
            else:
                error = sum(
                    a != b
                    for rp, ro in zip(predicted_grid, observed_grid)
                    for a, b in zip(rp, ro)
                )
            total_cells = max(1, sum(len(r) for r in observed_grid))
            discrepancy = error / total_cells

        record = EpisodeRecord(
            record_id=_new_id(),
            game_id=self.game_id,
            level_id=level_id,
            observation_id=observation_id,
            model_version=model_version,
            action_id=action_id,
            action_x=action_x,
            action_y=action_y,
            before_grid_bytes=before_bytes,
            predicted_grid_bytes=predicted_bytes,
            observed_grid_bytes=observed_bytes,
            changed_cell_error=error,
            actual_success=actual_success,
            discrepancy_score=discrepancy,
            timestamp=self._now(),
            prediction_available=prediction_available,
        )

        needed = record.byte_size()
        self._evict_to_fit(needed)

        # Evict oldest episode if at record count cap (after byte eviction)
        while len(self._episodes) >= self.episode_cap:
            oldest_id = next(iter(self._episodes))
            oldest = self._episodes.pop(oldest_id)
            self._bytes_used = max(0, self._bytes_used - oldest.byte_size())
            self._invalidate_unsupported({oldest_id})

        self._episodes[record.record_id] = record
        self._bytes_used += needed
        return record

    def get_episode(self, record_id: str) -> EpisodeRecord | None:
        return self._episodes.get(record_id)

    def query_episodes(
        self,
        *,
        level_id: int | None = None,
        model_version: str | None = None,
        min_success: bool | None = None,
        max_discrepancy: float | None = None,
        limit: int = 64,
    ) -> list[EpisodeRecord]:
        """Exact contextual filter, newest first."""
        results = []
        for ep in reversed(list(self._episodes.values())):
            if level_id is not None and ep.level_id != level_id:
                continue
            if model_version is not None and ep.model_version != model_version:
                continue
            if min_success is not None and ep.actual_success != min_success:
                continue
            if max_discrepancy is not None and ep.discrepancy_score > max_discrepancy:
                continue
            results.append(ep)
            if len(results) >= limit:
                break
        return results

    # -----------------------------------------------------------------------
    # Mechanic beliefs
    # -----------------------------------------------------------------------

    def add_belief(
        self,
        *,
        level_context: int | None = None,
        hypothesis: str,
        mechanic_tags: Sequence[str] = (),
        object_relation_tags: Sequence[str] = (),
        goal_tags: Sequence[str] = (),
        supporting_ids: Sequence[str] = (),
        contradicting_ids: Sequence[str] = (),
        revision_of: str | None = None,
    ) -> MechanicBelief:
        """Add a new belief. Validates evidence IDs exist in episode store."""
        for eid in (*supporting_ids, *contradicting_ids):
            if eid not in self._episodes:
                raise ValueError(f"Evidence ID {eid!r} not found in episode store")

        if revision_of is not None and revision_of not in self._beliefs:
            raise ValueError(f"revision_of belief {revision_of!r} not found")

        # Determine status from evidence
        status = "provisional" if supporting_ids else "unsupported"

        # Evict oldest belief if at cap
        while len(self._beliefs) >= self.mechanic_cap:
            oldest_id = next(iter(self._beliefs))
            del self._beliefs[oldest_id]

        belief = MechanicBelief(
            belief_id=_new_id(),
            game_id=self.game_id,
            level_context=level_context,
            hypothesis=hypothesis,
            mechanic_tags=tuple(mechanic_tags),
            object_relation_tags=tuple(object_relation_tags),
            goal_tags=tuple(goal_tags),
            supporting_ids=tuple(supporting_ids),
            contradicting_ids=tuple(contradicting_ids),
            status=status,
            revision=1,
            revision_of=revision_of,
            schema_version=self.schema_version,
            timestamp=self._now(),
        )
        self._beliefs[belief.belief_id] = belief
        self._bytes_used += 512
        return belief

    def contradict_belief(
        self,
        belief_id: str,
        *,
        contradicting_ids: Sequence[str],
        new_hypothesis: str | None = None,
    ) -> MechanicBelief:
        """Mark a belief as contradicted; returns an updated record.

        The contradiction supersedes rather than erases: revision is incremented,
        the original hypothesis is preserved unless new_hypothesis is given.
        """
        existing = self._beliefs.get(belief_id)
        if existing is None:
            raise KeyError(f"Belief {belief_id!r} not found")
        for eid in contradicting_ids:
            if eid not in self._episodes:
                raise ValueError(f"Contradicting evidence ID {eid!r} not in episode store")

        updated = replace(
            existing,
            contradicting_ids=existing.contradicting_ids + tuple(contradicting_ids),
            status="contradicted",
            hypothesis=new_hypothesis or existing.hypothesis,
            revision=existing.revision + 1,
            timestamp=self._now(),
        )
        self._beliefs[belief_id] = updated
        return updated

    def query_beliefs(
        self,
        *,
        mechanic_tags: Sequence[str] = (),
        object_tags: Sequence[str] = (),
        goal_tags: Sequence[str] = (),
        status: str | None = None,
        level_context: int | None = None,
        limit: int = 16,
    ) -> list[MechanicBelief]:
        """Retrieve by relevance score (deterministic, no embeddings)."""
        results = []
        for b in self._beliefs.values():
            if status is not None and b.status != status:
                continue
            if level_context is not None and b.level_context not in (None, level_context):
                continue
            results.append(b)
        results.sort(
            key=lambda b: b.relevance_score(mechanic_tags, object_tags, goal_tags),
            reverse=True,
        )
        return results[:limit]

    # -----------------------------------------------------------------------
    # Goal hypotheses
    # -----------------------------------------------------------------------

    def add_goal(
        self,
        *,
        level_context: int | None = None,
        description: str,
        observable_target: str,
        disconfirming_test: str,
        supporting_ids: Sequence[str] = (),
    ) -> GoalHypothesis:
        """Add a goal hypothesis. First success -> provisional; second -> active."""
        for eid in supporting_ids:
            if eid not in self._episodes:
                raise ValueError(f"Evidence ID {eid!r} not found in episode store")

        # Evict oldest goal if at cap
        while len(self._goals) >= self.goal_cap:
            oldest_id = next(iter(self._goals))
            del self._goals[oldest_id]

        n_support = len(supporting_ids)
        status = "active" if n_support >= 2 else ("provisional" if n_support == 1 else "provisional")

        goal = GoalHypothesis(
            goal_id=_new_id(),
            game_id=self.game_id,
            level_context=level_context,
            description=description,
            observable_target=observable_target,
            disconfirming_test=disconfirming_test,
            supporting_ids=tuple(supporting_ids),
            status=status,
            schema_version=self.schema_version,
            timestamp=self._now(),
        )
        self._goals[goal.goal_id] = goal
        return goal

    def promote_goal(self, goal_id: str, *, new_supporting_id: str) -> GoalHypothesis:
        """Add supporting evidence. Promotes provisional -> active on second independent success."""
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal {goal_id!r} not found")
        if new_supporting_id not in self._episodes:
            raise ValueError(f"Evidence ID {new_supporting_id!r} not in episode store")
        if new_supporting_id in goal.supporting_ids:
            raise ValueError("Evidence ID already registered for this goal")

        updated_ids = goal.supporting_ids + (new_supporting_id,)
        new_status = "active" if len(updated_ids) >= 2 else goal.status
        updated = replace(
            goal,
            supporting_ids=updated_ids,
            status=new_status,
            timestamp=self._now(),
        )
        self._goals[goal_id] = updated
        return updated

    def falsify_goal(self, goal_id: str) -> GoalHypothesis:
        """Mark a goal falsified. Disconfirming test has been observed."""
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal {goal_id!r} not found")
        updated = replace(goal, status="falsified", timestamp=self._now())
        self._goals[goal_id] = updated
        return updated

    def query_goals(
        self,
        *,
        status: str | None = None,
        level_context: int | None = None,
        limit: int = 16,
    ) -> list[GoalHypothesis]:
        results = [
            g for g in self._goals.values()
            if (status is None or g.status == status)
            and (level_context is None or g.level_context in (None, level_context))
        ]
        results.sort(key=lambda g: g.timestamp, reverse=True)
        return results[:limit]

    def promote_goals_from_attribution(
        self,
        goal_attribution: "Sequence[Sequence[str]]",
        *,
        episode_id: str,
    ) -> list[GoalHypothesis]:
        """Promote goals that received credit in a GoalDirectedPlan.

        Reads the goal_attribution field of a GoalDirectedPlan (a sequence of
        per-step goal_id tuples) and calls promote_goal() for each goal_id that
        appears at least once. Only goals in the working-memory store and not
        yet falsified are eligible; unknown IDs are silently skipped so the
        caller does not need to filter.

        Parameters
        ----------
        goal_attribution : sequence of sequences of str
            GoalDirectedPlan.goal_attribution — same length as plan.actions.
        episode_id : str
            The EpisodeRecord.record_id for the step that produced the plan.
            Must exist in the episode store (validated by promote_goal).

        Returns
        -------
        list[GoalHypothesis]
            Updated records for every goal that was promoted (status may have
            changed from provisional → active on second independent evidence).
        """
        if episode_id not in self._episodes:
            raise ValueError(f"episode_id {episode_id!r} not found in episode store")

        # Collect unique goal IDs that received any credit across all plan steps
        credited: set[str] = set()
        for step_ids in goal_attribution:
            for gid in step_ids:
                credited.add(gid)

        updated = []
        for gid in credited:
            goal = self._goals.get(gid)
            if goal is None:
                continue  # not in this memory; skip
            if goal.status == "falsified":
                continue  # never promote a falsified goal
            if episode_id in goal.supporting_ids:
                continue  # duplicate evidence; promote_goal would reject it
            updated.append(self.promote_goal(gid, new_supporting_id=episode_id))
        return updated

    def falsify_goals_from_observation(
        self,
        observed_grid: "tuple[tuple[int,...],...]",
        *,
        level_context: int | None = None,
        satisfaction_threshold: float = 0.8,
    ) -> list[GoalHypothesis]:
        """Falsify goals whose disconfirming condition is observed in the grid.

        For each active or provisional goal the method checks whether the
        disconfirming_test observable is satisfied in the observed_grid.
        The disconfirming_test string encodes the test in one of two forms:

          "color:<N>"            — color N present anywhere in the grid
          "color:<N>@<x0>,<y0>,<x1>,<y1>"  — color N present in a region

        Any goal whose disconfirming test is satisfied at or above
        satisfaction_threshold has falsify_goal() called on it.

        Goals with disconfirming_test strings that do not match either pattern
        are left unchanged — the check is conservative (no false falsification).

        Parameters
        ----------
        observed_grid : grid tuple
            The real post-action observation.
        level_context : int | None
            If supplied, only goals whose level_context matches are checked.
        satisfaction_threshold : float
            Fraction of target region that must match to count as satisfied.
            Default 0.8 (not full coverage, to handle partial observations).

        Returns
        -------
        list[GoalHypothesis]
            Newly falsified records.
        """
        import re as _re

        _COLOR_ONLY = _re.compile(r"^color:(\d+)$")
        _COLOR_REGION = _re.compile(
            r"^color:(\d+)@(\d+),(\d+),(\d+),(\d+)$"
        )

        def _check_disconf(test: str) -> bool:
            """Return True if the disconfirming condition holds in observed_grid."""
            m = _COLOR_ONLY.match(test.strip())
            if m:
                color = int(m.group(1))
                if not 0 <= color <= 15:
                    return False
                return any(cell == color for row in observed_grid for cell in row)

            m = _COLOR_REGION.match(test.strip())
            if m:
                color = int(m.group(1))
                x0, y0, x1, y1 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
                if not 0 <= color <= 15:
                    return False
                h = len(observed_grid)
                w = len(observed_grid[0]) if h else 0
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(w - 1, x1), min(h - 1, y1)
                total = (x1c - x0c + 1) * (y1c - y0c + 1)
                if total <= 0:
                    return False
                matching = sum(
                    1
                    for r in range(y0c, y1c + 1)
                    for c in range(x0c, x1c + 1)
                    if observed_grid[r][c] == color
                )
                return (matching / total) >= satisfaction_threshold

            # Unknown pattern — leave unchanged
            return False

        falsified = []
        for gid, goal in list(self._goals.items()):
            if goal.status == "falsified":
                continue
            if level_context is not None and goal.level_context not in (None, level_context):
                continue
            if _check_disconf(goal.disconfirming_test):
                falsified.append(self.falsify_goal(gid))
        return falsified

    # -----------------------------------------------------------------------
    # Procedure records
    # -----------------------------------------------------------------------

    def add_procedure(
        self,
        *,
        level_context: int | None = None,
        name: str,
        preconditions: Sequence[str],
        parameters: Sequence[str] = (),
        steps: Sequence[str],
        milestones: Sequence[str] = (),
        postconditions: Sequence[str] = (),
        exceptions: Sequence[str] = (),
        supporting_ids: Sequence[str] = (),
    ) -> ProcedureRecord:
        """Add a procedure. Status follows the two-success promotion rule."""
        for eid in supporting_ids:
            if eid not in self._episodes:
                raise ValueError(f"Evidence ID {eid!r} not found in episode store")

        while len(self._procedures) >= self.procedure_cap:
            oldest_id = next(iter(self._procedures))
            del self._procedures[oldest_id]

        status = "active" if len(supporting_ids) >= 2 else "provisional"
        proc = ProcedureRecord(
            procedure_id=_new_id(),
            game_id=self.game_id,
            level_context=level_context,
            name=name,
            preconditions=tuple(preconditions),
            parameters=tuple(parameters),
            steps=tuple(steps),
            milestones=tuple(milestones),
            postconditions=tuple(postconditions),
            exceptions=tuple(exceptions),
            supporting_ids=tuple(supporting_ids),
            status=status,
            schema_version=self.schema_version,
            timestamp=self._now(),
        )
        self._procedures[proc.procedure_id] = proc
        self._bytes_used += 512
        return proc

    def invalidate_procedure(self, procedure_id: str) -> ProcedureRecord:
        """Mark a procedure invalidated (failed precondition or milestone)."""
        proc = self._procedures.get(procedure_id)
        if proc is None:
            raise KeyError(f"Procedure {procedure_id!r} not found")
        updated = replace(proc, status="invalidated", timestamp=self._now())
        self._procedures[procedure_id] = updated
        return updated

    def promote_procedure(
        self, procedure_id: str, *, new_supporting_id: str
    ) -> ProcedureRecord:
        """Add supporting evidence. Promotes provisional -> active on second success."""
        proc = self._procedures.get(procedure_id)
        if proc is None:
            raise KeyError(f"Procedure {procedure_id!r} not found")
        if new_supporting_id not in self._episodes:
            raise ValueError(f"Evidence ID {new_supporting_id!r} not in episode store")
        if new_supporting_id in proc.supporting_ids:
            raise ValueError("Evidence ID already registered for this procedure")

        updated_ids = proc.supporting_ids + (new_supporting_id,)
        new_status = "active" if len(updated_ids) >= 2 else proc.status
        updated = replace(proc, supporting_ids=updated_ids, status=new_status,
                          timestamp=self._now())
        self._procedures[procedure_id] = updated
        return updated

    def promote_procedure_diverse(
        self, procedure_id: str, *, new_supporting_id: str
    ) -> ProcedureRecord:
        """Add evidence; promote to 'active' only on DIVERSE-context second success.

        R12 anti-overfitting guard: two successes from the SAME level context do
        not justify promotion (they may reflect one memorised situation). The
        procedure only becomes 'active' when its supporting episodes span at
        least two distinct level contexts. Same-context evidence is still
        recorded (it strengthens the record) but the status stays 'provisional'.

        The supporting episode must be a real, successful transition.
        """
        proc = self._procedures.get(procedure_id)
        if proc is None:
            raise KeyError(f"Procedure {procedure_id!r} not found")
        ep = self._episodes.get(new_supporting_id)
        if ep is None:
            raise ValueError(f"Evidence ID {new_supporting_id!r} not in episode store")
        if new_supporting_id in proc.supporting_ids:
            raise ValueError("Evidence ID already registered for this procedure")
        if not ep.actual_success:
            raise ValueError("Promotion evidence must be a successful episode")

        updated_ids = proc.supporting_ids + (new_supporting_id,)
        # Gather the level contexts of all supporting episodes still in memory.
        contexts = {
            self._episodes[sid].level_id
            for sid in updated_ids
            if sid in self._episodes
        }
        diverse = len(contexts) >= 2
        new_status = "active" if (len(updated_ids) >= 2 and diverse) else proc.status
        if new_status == "active":
            new_status = "active"
        elif len(updated_ids) >= 2:
            # Enough successes but not yet diverse: keep provisional explicitly.
            new_status = "provisional" if proc.status != "invalidated" else proc.status
        updated = replace(
            proc, supporting_ids=updated_ids, status=new_status, timestamp=self._now()
        )
        self._procedures[procedure_id] = updated
        return updated

    def revoke_procedure_on_contradiction(
        self, procedure_id: str, *, failing_episode_id: str
    ) -> ProcedureRecord:
        """Invalidate a procedure when new real evidence contradicts it.

        R12 contradiction revocation: an 'active' or 'provisional' procedure that
        later produces a FAILED outcome (a real, unsuccessful episode) is demoted
        to 'invalidated' rather than silently retained. The failing episode is
        recorded in the procedure's supporting_ids as documented evidence of the
        contradiction (never erased). Reuse of an invalidated procedure is blocked
        by find_chainable (which only returns 'active').
        """
        proc = self._procedures.get(procedure_id)
        if proc is None:
            raise KeyError(f"Procedure {procedure_id!r} not found")
        ep = self._episodes.get(failing_episode_id)
        if ep is None:
            raise ValueError(f"Evidence ID {failing_episode_id!r} not in episode store")
        if ep.actual_success:
            raise ValueError("Contradiction evidence must be an UNsuccessful episode")

        new_ids = proc.supporting_ids
        if failing_episode_id not in new_ids:
            new_ids = new_ids + (failing_episode_id,)
        updated = replace(
            proc, supporting_ids=new_ids, status="invalidated", timestamp=self._now()
        )
        self._procedures[procedure_id] = updated
        return updated

    def query_procedures(
        self,
        *,
        status: str | None = None,
        level_context: int | None = None,
        limit: int = 16,
    ) -> list[ProcedureRecord]:
        results = [
            p for p in self._procedures.values()
            if (status is None or p.status == status)
            and (level_context is None or p.level_context in (None, level_context))
        ]
        results.sort(key=lambda p: p.timestamp, reverse=True)
        return results[:limit]

    def find_chainable(
        self, predecessor: ProcedureRecord, limit: int = 8
    ) -> list[ProcedureRecord]:
        """Return active procedures whose preconditions are covered by predecessor's postconditions."""
        return [
            p for p in self._procedures.values()
            if p.status == "active"
            and p.procedure_id != predecessor.procedure_id
            and predecessor.chains_with(p)
        ][:limit]

    # -----------------------------------------------------------------------
    # Imagined branches (current plan only)
    # -----------------------------------------------------------------------

    def set_plan_branches(self, branches: list[ImaginedBranch]) -> None:
        """Replace the current plan's imagined branches entirely.

        Called at the start of each new planning pass. Old branches are discarded.
        Branches are never persisted as real evidence.
        """
        for b in branches:
            if b.game_id != self.game_id:
                raise ValueError("Branch game_id does not match memory game_id")
        # Remove old branch byte accounting
        for b in self._branches.values():
            self._bytes_used = max(0, self._bytes_used - b.byte_size())
        self._branches = {b.branch_id: b for b in branches}
        for b in branches:
            self._bytes_used += b.byte_size()

    def get_branches(self) -> list[ImaginedBranch]:
        return list(self._branches.values())

    def clear_branches(self) -> None:
        for b in self._branches.values():
            self._bytes_used = max(0, self._bytes_used - b.byte_size())
        self._branches.clear()

    # -----------------------------------------------------------------------
    # Summary / diagnostics
    # -----------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "game_id": self.game_id,
            "schema_version": self.schema_version,
            "episodes": len(self._episodes),
            "beliefs": len(self._beliefs),
            "goals": len(self._goals),
            "procedures": len(self._procedures),
            "branches": len(self._branches),
            "bytes_used": self._bytes_used,
            "ceiling_bytes": self.ceiling_bytes,
            "caps": {
                "episode": self.episode_cap,
                "mechanic": self.mechanic_cap,
                "goal": self.goal_cap,
                "procedure": self.procedure_cap,
            },
        }

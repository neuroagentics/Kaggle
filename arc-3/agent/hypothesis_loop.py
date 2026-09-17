"""Testable hypothesis loop (Step 3 of the trustworthy-internal-game milestone).

The intended reasoning:
    propose competing explanations
      -> simulate their consequences
      -> choose a distinguishing action
      -> observe
      -> reject or strengthen explanations
      -> replan.

Success is NOT "changing the hypothesis text changes an output". Success is:
a supported hypothesis improves PREDICTIONS and ACTION SELECTION. Observed
evidence — not text — decides which hypotheses survive.

This module is proposer-agnostic: a hypothesis exposes a `predict(grid, action)`
that returns the next grid it EXPECTS. A rule-based proposer is provided for the
controlled worlds; the same interface is what a Qwen-driven proposer would fill
(each MechanicHypothesis maps to a concrete predicted consequence). The loop's
logic — distinguish, observe, score, prune — is identical regardless of source.

Nothing here is submitted or scored on the competition; it is validated on the
controlled worlds where ground truth is known.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Grid = tuple[tuple[int, ...], ...]


@dataclass
class Hypothesis:
    """A competing explanation that makes a CONCRETE, falsifiable prediction.

    name        : short label (e.g. 'move_left').
    predict     : (grid, action_id, x, y) -> expected next grid.
    confidence  : running belief in [0, 1]; updated by evidence, not by text.
    alive       : False once contradicted by observation.
    hits/misses : evidence counters (observed agreements / disagreements).
    """
    name: str
    predict: Callable[[Grid, int, "int|None", "int|None"], Grid]
    confidence: float = 0.5
    alive: bool = True
    hits: int = 0
    misses: int = 0
    meta: dict = field(default_factory=dict)


def _grid_eq(a: Grid, b: Grid) -> bool:
    return a == b


def _changed_cells(before: Grid, after: Grid) -> int:
    n = 0
    for r in range(len(before)):
        for c in range(len(before[0])):
            if before[r][c] != after[r][c]:
                n += 1
    return n


class HypothesisLoop:
    """Runs the propose -> simulate -> distinguish -> observe -> update cycle.

    The loop does not know or care whether hypotheses came from a rule, a
    template, or a language model. It only uses each hypothesis's concrete
    prediction and the real observed outcome.
    """

    def __init__(self, hypotheses: list[Hypothesis], *, reject_threshold: float = 0.05):
        if not hypotheses:
            raise ValueError("HypothesisLoop requires at least one hypothesis")
        self.hypotheses = hypotheses
        # A hypothesis is rejected once its confidence falls below this.
        self.reject_threshold = reject_threshold

    def alive(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.alive]

    # -- distinguish -------------------------------------------------------
    def distinguishing_action(self, grid: Grid, candidate_actions) -> tuple:
        """Pick the action under which the ALIVE hypotheses disagree the most.

        Disagreement = number of distinct predicted next-grids across alive
        hypotheses (ties broken by total pairwise changed-cell distance). An
        action that all survivors agree on teaches us nothing; the best probe is
        where they diverge.
        """
        alive = self.alive()
        best = None
        best_score = -1
        for act in candidate_actions:
            aid, x, y = act
            preds = [h.predict(grid, aid, x, y) for h in alive]
            distinct = len({p for p in preds})
            # pairwise divergence as a tie-breaker
            div = 0
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    div += _changed_cells(preds[i], preds[j])
            score = (distinct, div)
            if score > (best_score if isinstance(best_score, tuple) else (-1, -1)):
                best_score = score
                best = act
        return best if best is not None else candidate_actions[0]

    # -- observe + update --------------------------------------------------
    def update(self, grid: Grid, action: tuple, observed_next: Grid) -> dict:
        """Score alive hypotheses against the real outcome; strengthen/reject.

        Returns a small report: which hypotheses matched, updated confidences,
        and which were rejected this step.
        """
        aid, x, y = action
        alive = self.alive()
        matched = []
        for h in alive:
            pred = h.predict(grid, aid, x, y)
            if _grid_eq(pred, observed_next):
                h.hits += 1
                matched.append(h.name)
            else:
                h.misses += 1
        # Confidence = smoothed hit rate (Laplace). Evidence, not text, drives it.
        rejected = []
        for h in alive:
            total = h.hits + h.misses
            h.confidence = (h.hits + 1) / (total + 2)
            if h.confidence < self.reject_threshold or (h.misses >= 3 and h.hits == 0):
                h.alive = False
                rejected.append(h.name)
        return {
            "matched": matched,
            "confidences": {h.name: round(h.confidence, 3) for h in self.hypotheses},
            "alive": [h.name for h in self.alive()],
            "rejected": rejected,
        }

    def best(self) -> Hypothesis | None:
        alive = self.alive()
        if not alive:
            return None
        return max(alive, key=lambda h: h.confidence)


# ---------------------------------------------------------------------------
# Adapter: MechanicHypothesis (deliberator/Qwen output) -> testable Hypothesis
# ---------------------------------------------------------------------------

def _cells_of_color(grid: Grid, colors) -> list[tuple[int, int]]:
    out = []
    cs = set(colors or ())
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] in cs:
                out.append((r, c))
    return out


def hypothesis_from_mechanic(mech) -> "Hypothesis":
    """Turn a MechanicHypothesis (agent.model) into a testable Hypothesis.

    This closes the boundary between the language-model proposer and the evidence
    loop: whatever Qwen proposes becomes a CONCRETE next-frame prediction that the
    loop can distinguish and score. Currently supports the two mechanic shapes the
    controlled worlds exercise:

      - movement: move the cells of `affected_colors` by `direction_vector`.
      - interaction/transformation: recolor `affected_colors` cells to the color
        named in an effect_tag like 'to_color:9' (falls back to no-op if absent).

    Unsupported types produce a no-op predictor (predicts "no change"), which the
    loop will reject if the world actually changes — an honest default, not a guess.
    """
    mtype = getattr(mech, "mechanic_type", "unknown")
    colors = getattr(mech, "affected_colors", None)
    dvec = getattr(mech, "direction_vector", None)
    effect_tags = getattr(mech, "effect_tags", ()) or ()
    name = f"{mtype}:{getattr(mech, 'hypothesis_id', '?')[:6]}"
    conf = float(getattr(mech, "confidence", 0.5) or 0.5)

    def _to_color():
        for t in effect_tags:
            if isinstance(t, str) and t.startswith("to_color:"):
                try:
                    return int(t.split(":", 1)[1])
                except ValueError:
                    return None
        return None

    if mtype == "movement" and dvec is not None:
        # direction_vector is (dx, dy) in grid coords; we treat as (dcol, drow).
        dx, dy = dvec

        def predict(grid, aid, x, y):
            H, W = len(grid), len(grid[0])
            targets = _cells_of_color(grid, colors) if colors else []
            g = [list(row) for row in grid]
            for (r, c) in targets:
                nr = max(0, min(H - 1, r + dy))
                nc = max(0, min(W - 1, c + dx))
                val = grid[r][c]
                g[r][c] = 0
                g[nr][nc] = val
            return tuple(tuple(row) for row in g)

        return Hypothesis(name=name, predict=predict, confidence=conf,
                          meta={"mechanic_type": mtype, "direction_vector": dvec})

    if mtype in ("interaction", "transformation"):
        new_color = _to_color()

        def predict(grid, aid, x, y):
            if new_color is None:
                return grid
            g = [list(row) for row in grid]
            for (r, c) in _cells_of_color(grid, colors):
                g[r][c] = new_color
            return tuple(tuple(row) for row in g)

        return Hypothesis(name=name, predict=predict, confidence=conf,
                          meta={"mechanic_type": mtype})

    # Unsupported: honest no-op predictor (predicts no change).
    def predict(grid, aid, x, y):
        return grid

    return Hypothesis(name=name, predict=predict, confidence=conf,
                      meta={"mechanic_type": mtype, "unsupported": True})

"""Step 3 harness — testable hypotheses that improve predictions AND actions.

Scenario (controlled, ground-truth known): a movement world where the mapping
from action-id to direction is UNKNOWN to the agent (it must be discovered). Four
competing hypotheses are proposed, each asserting a different action->direction
mapping. The loop:
  1. proposes the competing hypotheses,
  2. simulates each hypothesis's predicted next frame,
  3. picks the action where they DISAGREE most (a distinguishing probe),
  4. observes the real outcome,
  5. rejects contradicted hypotheses / strengthens the consistent one,
  6. once one survives, uses it to PLAN toward a goal cell.

Two success criteria, both measured honestly against ground truth:
  A. PREDICTION: after the discovery phase, the surviving hypothesis is the TRUE
     mapping (rejects the wrong ones from observed evidence, not text).
  B. ACTION SELECTION: an agent that uses the discovered mapping to move toward a
     goal reaches it in FEWER steps than a no-hypothesis baseline (which cannot
     aim because it does not know which action moves which way).

Comparison is against a matched control (same world, same start/goal, random legal
moves = "no usable hypothesis"). Averaged over many seeds.

Not submitted; controlled-world validation only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.hypothesis_loop import Hypothesis, HypothesisLoop
from training.controlled_worlds import make_movement_world, AGENT, BG, A_UP, A_DOWN, A_LEFT, A_RIGHT

ROOT = Path(__file__).resolve().parent.parent

# The TRUE mapping (what the world actually does). The agent does NOT know this;
# it must discover it. This mirrors training.controlled_worlds._MOVE_DELTA.
TRUE_DELTA = {A_UP: (-1, 0), A_DOWN: (1, 0), A_LEFT: (0, -1), A_RIGHT: (0, 1)}


def _grid(world) -> tuple:
    return tuple(tuple(row) for row in world.grid)


def _find_agent(grid) -> tuple[int, int]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == AGENT:
                return (r, c)
    return (0, 0)


def _predict_with_delta(delta_map):
    """Build a Hypothesis.predict that assumes a given action->(dr,dc) mapping."""
    def predict(grid, aid, x, y):
        H, W = len(grid), len(grid[0])
        r, c = _find_agent(grid)
        dr, dc = delta_map.get(aid, (0, 0))
        nr = max(0, min(H - 1, r + dr))
        nc = max(0, min(W - 1, c + dc))
        g = [list(row) for row in grid]
        g[r][c] = BG
        g[nr][nc] = AGENT
        return tuple(tuple(row) for row in g)
    return predict


def _candidate_hypotheses() -> list[Hypothesis]:
    """Four competing action->direction mappings. Exactly one matches the world.

    We build distinct WRONG mappings by rotating the direction assignments, so
    only one hypothesis is the true mapping and the others are falsifiable.
    """
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    acts = [A_UP, A_DOWN, A_LEFT, A_RIGHT]
    hyps = []
    for rot in range(4):
        mapping = {acts[i]: dirs[(i + rot) % 4] for i in range(4)}
        name = "true" if rot == 0 else f"rot{rot}"
        hyps.append(Hypothesis(name=name, predict=_predict_with_delta(mapping),
                               meta={"mapping": mapping}))
    return hyps


def discover_mapping(world, rng, max_probes: int) -> tuple[HypothesisLoop, int]:
    """Run the distinguishing-probe loop until one hypothesis survives.

    Returns (loop, probes_used). Steps the REAL world with the chosen probe action.
    """
    loop = HypothesisLoop(_candidate_hypotheses())
    candidate_actions = [(A_UP, None, None), (A_DOWN, None, None),
                         (A_LEFT, None, None), (A_RIGHT, None, None)]
    probes = 0
    for _ in range(max_probes):
        if len(loop.alive()) <= 1:
            break
        grid = _grid(world)
        act = loop.distinguishing_action(grid, candidate_actions)
        world.step(act[0], act[1], act[2])
        loop.update(grid, act, _grid(world))
        probes += 1
    return loop, probes


def _steps_to_goal_with_mapping(world, mapping, goal, max_steps: int) -> int | None:
    """Greedily move toward goal using a KNOWN action->direction mapping.

    Returns steps taken to reach goal, or None if not reached within max_steps.
    """
    # invert mapping: direction -> action
    dir_to_act = {v: k for k, v in mapping.items()}
    for step in range(1, max_steps + 1):
        grid = _grid(world)
        r, c = _find_agent(grid)
        gr, gc = goal
        if (r, c) == goal:
            return step - 1
        # choose the axis with the larger remaining distance
        want = None
        if abs(gr - r) >= abs(gc - c) and gr != r:
            want = (1 if gr > r else -1, 0)
        elif gc != c:
            want = (0, 1 if gc > c else -1)
        elif gr != r:
            want = (1 if gr > r else -1, 0)
        if want is None or want not in dir_to_act:
            return None
        aid = dir_to_act[want]
        world.step(aid, None, None)
        if _find_agent(_grid(world)) == goal:
            return step
    return None


def _steps_to_goal_random(world, rng, goal, max_steps: int) -> int | None:
    """Baseline with NO usable hypothesis: random legal moves (cannot aim)."""
    acts = [A_UP, A_DOWN, A_LEFT, A_RIGHT]
    for step in range(1, max_steps + 1):
        if _find_agent(_grid(world)) == goal:
            return step - 1
        world.step(rng.choice(acts), None, None)
        if _find_agent(_grid(world)) == goal:
            return step
    return None


def run(n_trials: int, grid: int, max_probes: int, max_steps: int, seed: int) -> dict:
    correct_survivor = 0
    rejected_all_wrong = 0
    hyp_reached = 0
    base_reached = 0
    hyp_steps = []
    base_steps = []

    for i in range(n_trials):
        rng = random.Random(seed * 7919 + i)

        # --- discovery phase (uses probes on a fresh world) ---
        w1 = make_movement_world(rng, h=grid, w=grid)
        loop, probes = discover_mapping(w1, rng, max_probes)
        best = loop.best()
        if best is not None and best.name == "true":
            correct_survivor += 1
        # did the loop reject every wrong hypothesis?
        wrong_alive = [h for h in loop.alive() if h.name != "true"]
        if not wrong_alive:
            rejected_all_wrong += 1

        discovered = best.meta["mapping"] if best is not None else None

        # --- action-selection phase: fresh world, fixed start & goal ---
        rng2 = random.Random(seed * 104729 + i)
        w_h = make_movement_world(rng2, h=grid, w=grid)
        start = _find_agent(_grid(w_h))
        # goal = a distinct random cell
        goal = start
        while goal == start:
            goal = (rng2.randrange(grid), rng2.randrange(grid))

        # rebuild identical world for the baseline
        rng2b = random.Random(seed * 104729 + i)
        w_b = make_movement_world(rng2b, h=grid, w=grid)

        if discovered is not None:
            hs = _steps_to_goal_with_mapping(w_h, discovered, goal, max_steps)
        else:
            hs = None
        bs = _steps_to_goal_random(w_b, random.Random(seed * 31 + i), goal, max_steps)

        if hs is not None:
            hyp_reached += 1
            hyp_steps.append(hs)
        if bs is not None:
            base_reached += 1
            base_steps.append(bs)

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "n_trials": n_trials,
        "grid": grid,
        "max_probes": max_probes,
        "max_steps": max_steps,
        "prediction": {
            "correct_survivor_rate": round(correct_survivor / n_trials, 3),
            "rejected_all_wrong_rate": round(rejected_all_wrong / n_trials, 3),
        },
        "action_selection": {
            "hypothesis_reached_goal_rate": round(hyp_reached / n_trials, 3),
            "baseline_reached_goal_rate": round(base_reached / n_trials, 3),
            "hypothesis_avg_steps": _avg(hyp_steps),
            "baseline_avg_steps": _avg(base_steps),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/step3_hypothesis_report.json")
    args = ap.parse_args()

    report = run(args.n_trials, args.grid, args.max_probes, args.max_steps, args.seed)
    print(json.dumps(report, indent=2))
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[step3] wrote {out}")


if __name__ == "__main__":
    main()

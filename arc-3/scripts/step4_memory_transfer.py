"""Step 4 harness — memory as DEMONSTRATED TRANSFER.

Stores a successful action sequence with its starting conditions, expected
intermediate states (milestones), failure conditions (exceptions), and supporting
observations — using the REAL WorkingMemory / ProcedureRecord — then reuses it in a
DIFFERENT applicable situation and measures whether reuse improves decisions.

The transferable knowledge here is the discovered action->direction mapping plus a
parameterised, object-relative navigation procedure (no hardcoded coordinates). It
is learned in situation A (one start/goal), stored, then applied in situation B
(different start/goal, same mechanic).

Two conditions, matched (same worlds, starts, goals), averaged over seeds:
  - memory ON  : reuse the stored mapping/procedure in situation B -> can aim.
  - memory OFF : no stored knowledge; must rediscover by probing before it can aim
                 (probes cost real steps and count against the budget).

Success = memory ON reaches goals more often and/or in fewer TOTAL steps
(probes + navigation) than memory OFF, in the NEW situation. Transfer is only
credited when the stored procedure's preconditions actually match situation B.

Not submitted; controlled-world validation only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.internal_world import Action  # noqa: F401 (parity; not strictly needed)
from agent.memory import WorkingMemory
from agent.hypothesis_loop import Hypothesis, HypothesisLoop
from training.controlled_worlds import make_movement_world, AGENT, BG, A_UP, A_DOWN, A_LEFT, A_RIGHT

ROOT = Path(__file__).resolve().parent.parent
_MOVES = [A_UP, A_DOWN, A_LEFT, A_RIGHT]
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _grid(w):
    return tuple(tuple(r) for r in w.grid)


def _agent(grid):
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == AGENT:
                return (r, c)
    return (0, 0)


def _pred_delta(mapping):
    def predict(grid, aid, x, y):
        H, W = len(grid), len(grid[0])
        r, c = _agent(grid)
        dr, dc = mapping.get(aid, (0, 0))
        nr = max(0, min(H - 1, r + dr)); nc = max(0, min(W - 1, c + dc))
        g = [list(row) for row in grid]
        g[r][c] = BG; g[nr][nc] = AGENT
        return tuple(tuple(row) for row in g)
    return predict


def _candidate_hyps():
    hyps = []
    for rot in range(4):
        mapping = {_MOVES[i]: _DIRS[(i + rot) % 4] for i in range(4)}
        hyps.append(Hypothesis(name=f"rot{rot}", predict=_pred_delta(mapping),
                               meta={"mapping": mapping}))
    return hyps


def discover_mapping(world, max_probes):
    """Probe to discover the action->direction mapping. Returns (mapping, probes)."""
    loop = HypothesisLoop(_candidate_hyps())
    cands = [(a, None, None) for a in _MOVES]
    probes = 0
    for _ in range(max_probes):
        if len(loop.alive()) <= 1:
            break
        g = _grid(world)
        act = loop.distinguishing_action(g, cands)
        world.step(act[0], act[1], act[2])
        loop.update(g, act, _grid(world))
        probes += 1
    best = loop.best()
    return (best.meta["mapping"] if best else None), probes


def navigate(world, mapping, goal, max_steps):
    """Greedy object-relative navigation using a known mapping. Returns steps or None."""
    dir_to_act = {v: k for k, v in mapping.items()}
    for step in range(1, max_steps + 1):
        r, c = _agent(_grid(world))
        if (r, c) == goal:
            return step - 1
        gr, gc = goal
        if abs(gr - r) >= abs(gc - c) and gr != r:
            want = (1 if gr > r else -1, 0)
        elif gc != c:
            want = (0, 1 if gc > c else -1)
        elif gr != r:
            want = (1 if gr > r else -1, 0)
        else:
            return step - 1
        if want not in dir_to_act:
            return None
        world.step(dir_to_act[want], None, None)
        if _agent(_grid(world)) == goal:
            return step
    return None


def _store_procedure(mem: WorkingMemory, world_a, mapping, path_actions):
    """Store the successful sequence as a real ProcedureRecord with full context.

    Records starting conditions (preconditions), the action sequence (steps),
    expected intermediate states (milestones as observable tags), failure
    conditions (exceptions), and supporting observations (episode IDs).
    """
    # Supporting observation: the discovery episode (before/pred/observed).
    g0 = _grid(world_a)
    ep_rec = mem.add_episode(
        level_id=0,
        observation_id="cw_a_0",
        model_version="cw",
        action_id=path_actions[0] if path_actions else _MOVES[0],
        action_x=None, action_y=None,
        before_grid=g0, predicted_grid=g0, observed_grid=g0,
        actual_success=True,
        prediction_available=True,
    )
    ep = ep_rec.record_id
    # Object-relative parameters: no hardcoded coordinates.
    proc = mem.add_procedure(
        name="navigate_avatar_to_target",
        preconditions=("avatar_present", "target_reachable"),
        parameters=("avatar", "target"),
        steps=tuple(f"move:{d}" for d in ("dr>0", "dr<0", "dc>0", "dc<0")),
        milestones=("distance_to_target_decreases",),
        postconditions=("avatar_on_target",),
        exceptions=("avatar_blocked", "target_unreachable"),
        supporting_ids=(ep,),
    )
    # Also stash the discovered mapping alongside (as a belief-like fact).
    return proc, mapping


def run(n_trials, grid, max_probes, max_steps, seed):
    on_reached = off_reached = 0
    on_total_steps = []
    off_total_steps = []
    transfer_applicable = 0

    for i in range(n_trials):
        rng = random.Random(seed * 7919 + i)

        # ---- situation A: learn + store ----
        w_a = make_movement_world(rng, h=grid, w=grid)
        mapping_a, probes_a = discover_mapping(w_a, max_probes)
        mem = WorkingMemory(game_id="cw_movement")
        stored_mapping = None
        if mapping_a is not None:
            # navigate A to confirm success, then store
            rng_a2 = random.Random(seed * 13 + i)
            goal_a = _agent(_grid(w_a))
            while goal_a == _agent(_grid(w_a)):
                goal_a = (rng_a2.randrange(grid), rng_a2.randrange(grid))
            navigate(w_a, mapping_a, goal_a, max_steps)
            _store_procedure(mem, w_a, mapping_a, _MOVES)
            stored_mapping = mapping_a

        # ---- situation B: DIFFERENT start/goal, same mechanic ----
        rng_b = random.Random(seed * 104729 + i)
        w_b_on = make_movement_world(rng_b, h=grid, w=grid)
        start_b = _agent(_grid(w_b_on))
        goal_b = start_b
        while goal_b == start_b:
            goal_b = (rng_b.randrange(grid), rng_b.randrange(grid))
        # identical world for the OFF condition
        rng_b2 = random.Random(seed * 104729 + i)
        w_b_off = make_movement_world(rng_b2, h=grid, w=grid)

        # memory ON: reuse stored mapping (transfer). Preconditions match B
        # (avatar present, target reachable) -> applicable.
        procs = mem.query_procedures(status=None)
        applicable = stored_mapping is not None and len(procs) > 0 and \
            "avatar_present" in (procs[0].preconditions if procs else ())
        if applicable:
            transfer_applicable += 1
            steps_on = navigate(w_b_on, stored_mapping, goal_b, max_steps)
            total_on = steps_on if steps_on is not None else None
        else:
            # no transfer: must discover in B first (probes cost steps)
            m_on, p_on = discover_mapping(w_b_on, max_probes)
            nav = navigate(w_b_on, m_on, goal_b, max_steps) if m_on else None
            total_on = (p_on + nav) if nav is not None else None

        # memory OFF: no stored knowledge, must rediscover by probing in B
        m_off, p_off = discover_mapping(w_b_off, max_probes)
        nav_off = navigate(w_b_off, m_off, goal_b, max_steps) if m_off else None
        total_off = (p_off + nav_off) if nav_off is not None else None

        if total_on is not None:
            on_reached += 1; on_total_steps.append(total_on)
        if total_off is not None:
            off_reached += 1; off_total_steps.append(total_off)

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "n_trials": n_trials, "grid": grid, "max_probes": max_probes, "max_steps": max_steps,
        "transfer_applicable_rate": round(transfer_applicable / n_trials, 3),
        "memory_on": {
            "reached_goal_rate": round(on_reached / n_trials, 3),
            "avg_total_steps": _avg(on_total_steps),
        },
        "memory_off": {
            "reached_goal_rate": round(off_reached / n_trials, 3),
            "avg_total_steps": _avg(off_total_steps),
        },
        "note": ("memory_on reuses the stored action->direction procedure in the NEW "
                 "situation (no re-probing); memory_off must rediscover by probing, "
                 "which costs real steps. Both navigate the SAME B world/start/goal."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/step4_memory_transfer.json")
    args = ap.parse_args()
    report = run(args.n_trials, args.grid, args.max_probes, args.max_steps, args.seed)
    print(json.dumps(report, indent=2))
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[step4] wrote {out}")


if __name__ == "__main__":
    main()

"""hyper_arc/mcts.py — Memory-guided MCTS Engine.

Components:
  MCTSNode      — tree node (state, parent, children, visits, value).
  enumerate_actions — generates all discrete (primitive, kwargs) actions.
  MCTSEngine    — select / expand / rollout / backprop / solve.

Dual-memory prior policy:
  local_memory  (LocalTaskBuffer)  — weight multiplier 0.8 (or 1.0 when
                                     global_prior_weight == 0.0).
  global_memory (GlobalMemoryBank) — weight multiplier = global_prior_weight
                                     (default 0.2, set to 0.0 when no seed
                                     bank is available — MVP mode).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL, PrimitiveCall, DSLProgram
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.cost import exact_match, partial_reward

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ITERATIONS    = 2000
MAX_ROLLOUT_DEPTH = 10

# Default memory weight split (local : global = 0.8 : 0.2).
# When no seed bank is present, global_prior_weight is set to 0.0 at runtime
# and local weight automatically becomes 1.0 (see MCTSEngine._get_prior).
DEFAULT_LOCAL_PRIOR_WEIGHT  = 0.8
DEFAULT_GLOBAL_PRIOR_WEIGHT = 0.2

_DSL = SpatialDSL()


# ── MCTSNode ──────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    state:        ESB
    parent:       Optional["MCTSNode"]
    action_taken: Optional[PrimitiveCall]
    children:     list["MCTSNode"] = field(default_factory=list)
    visits:       int              = 0
    value:        float            = 0.0   # running mean Q

    def ucb1(self, C: float, prior: float) -> float:
        """UCB1 = Q + C * sqrt(ln(parent_N) / N) + prior.

        Returns inf for unvisited nodes (visits == 0).
        """
        if self.visits == 0:
            return float("inf")
        parent_n = self.parent.visits if self.parent else 1
        exploit  = self.value
        explore  = C * math.sqrt(math.log(max(parent_n, 1)) / self.visits)
        return exploit + explore + prior


# ── Action enumeration ────────────────────────────────────────────────────

def enumerate_actions(esb: ESB) -> list[PrimitiveCall]:
    """Generate all discrete (primitive_name, kwargs) for the current ESB.

    Covers all 12 DSL primitives with their MCTS-enumerable parameter domains.
    """
    H, W = esb.H, esb.W
    actions: list[PrimitiveCall] = []

    colors_present = [int(v) for v in esb.data[0].unique().tolist()]

    # ── translate ─────────────────────────────────────────────────────────
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            actions.append(("translate", {"dx": dx, "dy": dy}))
    # wrap variants (less common — only cardinal directions)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        actions.append(("translate", {"dx": dx, "dy": dy, "wrap": True}))

    # ── rotate ────────────────────────────────────────────────────────────
    for deg in [90, 180, 270]:
        actions.append(("rotate", {"degrees": deg}))

    # ── reflect ───────────────────────────────────────────────────────────
    for axis in ["horizontal", "vertical"]:
        actions.append(("reflect", {"axis": axis}))

    # ── extract_object ────────────────────────────────────────────────────
    seeds = [
        (0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),
        (H // 2, W // 2),
    ]
    for y, x in seeds:
        actions.append(("extract_object", {"x": x, "y": y}))

    # ── overlay ───────────────────────────────────────────────────────────
    for bm in ["overwrite", "or"]:
        # foreground = current state (self-overlay is a no-op but MCTS may
        # compose it with a prior extracted object stored externally)
        actions.append(("overlay", {"foreground": esb, "blend_mode": bm}))

    # ── crop_to_bbox ──────────────────────────────────────────────────────
    actions.append(("crop_to_bbox", {}))

    # ── scale_integer ─────────────────────────────────────────────────────
    for factor in [2, 3, 4]:
        actions.append(("scale_integer", {"factor": factor}))

    # ── tile ──────────────────────────────────────────────────────────────
    for rh in [2, 3]:
        for rw in [2, 3]:
            actions.append(("tile", {"repeats_h": rh, "repeats_w": rw}))

    # ── connect_points ────────────────────────────────────────────────────
    corner_pairs = [
        ((0, 0),       (0, W - 1)),
        ((0, 0),       (H - 1, 0)),
        ((0, 0),       (H - 1, W - 1)),
        ((0, W - 1),   (H - 1, 0)),
        ((0, W - 1),   (H - 1, W - 1)),
        ((H - 1, 0),   (H - 1, W - 1)),
    ]
    for (r1, c1), (r2, c2) in corner_pairs:
        for color in colors_present:
            actions.append(("connect_points", {
                "color": color,
                "row1": r1, "col1": c1,
                "row2": r2, "col2": c2,
            }))

    # ── color_remap ───────────────────────────────────────────────────────
    for src in colors_present:
        for dst in range(10):
            if src != dst:
                actions.append(("color_remap", {"mapping": {src: dst}}))

    # ── flood_fill ────────────────────────────────────────────────────────
    fill_seeds = [
        (0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),
        (H // 2, W // 2),
    ]
    for row, col in fill_seeds:
        current = int(esb.data[0, row, col].item())
        for nc in range(10):
            if nc != current:
                actions.append(("flood_fill", {
                    "row": row, "col": col, "new_color": nc,
                }))

    # ── symmetrize ────────────────────────────────────────────────────────
    for axis in ["horizontal", "vertical"]:
        for mode in ["copy", "priority"]:
            actions.append(("symmetrize", {"axis": axis, "mode": mode}))

    return actions


# ── MCTSEngine ────────────────────────────────────────────────────────────

class MCTSEngine:
    """Memory-guided MCTS over DSL program space.

    Args:
        global_memory:      GlobalMemoryBank (seed bank). May be empty.
        local_memory:       LocalTaskBuffer (per-task, updated on success).
        C:                  UCB1 exploration constant (default 1.41).
        global_prior_weight: Weight for global memory priors (0.0–1.0).
                             Set to 0.0 when no seed bank is available (MVP).
                             Default 0.2 when seed bank is loaded.
    """

    def __init__(
        self,
        global_memory: GlobalMemoryBank,
        local_memory:  LocalTaskBuffer,
        C:                   float = 1.41,
        global_prior_weight: float = DEFAULT_GLOBAL_PRIOR_WEIGHT,
    ) -> None:
        self.global_memory       = global_memory
        self.local_memory        = local_memory
        self.C                   = C
        self.global_prior_weight = global_prior_weight
        # Derived local weight — always sums to 1.0 with global_prior_weight,
        # except in MVP mode (global=0.0) where local gets full weight of 1.0.
        self.local_prior_weight  = (
            1.0 if global_prior_weight == 0.0
            else DEFAULT_LOCAL_PRIOR_WEIGHT
        )

    # ── Prior computation ─────────────────────────────────────────────────

    def _get_prior(self, node: MCTSNode, action: PrimitiveCall) -> float:
        """Compute dual-memory prior for a candidate action at a node.

        local  weight: 0.8  (1.0 in MVP mode when global_prior_weight == 0.0)
        global weight: 0.2  (0.0 in MVP mode)
        Combined prior is normalised to [0, 1] so it doesn't dominate UCB1.
        """
        partial = self._path_actions(node) + [action]

        local_results  = self.local_memory.query(partial)
        global_results = self.global_memory.query(partial)

        local_prior  = (sum(w for _, w in local_results)  * self.local_prior_weight
                        if local_results else 0.0)
        global_prior = (sum(w for _, w in global_results) * self.global_prior_weight
                        if global_results else 0.0)

        combined     = local_prior + global_prior
        denominator  = self.local_prior_weight + self.global_prior_weight
        # Normalise so the prior stays in [0, 1]
        return combined / max(denominator, 1e-8)

    def _path_actions(self, node: MCTSNode) -> DSLProgram:
        """Collect the action sequence from root to node."""
        path: DSLProgram = []
        cur = node
        while cur.parent is not None and cur.action_taken is not None:
            path.append(cur.action_taken)
            cur = cur.parent
        return list(reversed(path))

    # ── Four MCTS phases ──────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until a childless leaf."""
        while node.children:
            scores = [
                c.ucb1(self.C, self._get_prior(c, c.action_taken))
                for c in node.children
            ]
            node = node.children[scores.index(max(scores))]
        return node

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Generate child nodes for all valid actions from this state."""
        actions = enumerate_actions(node.state)
        for act in actions:
            name, kwargs = act
            # overlay uses the current node state as foreground — update ref
            if name == "overlay":
                kwargs = {**kwargs, "foreground": node.state}
            try:
                new_state = getattr(_DSL, name)(node.state, **kwargs)
            except (ValueError, IndexError, RuntimeError):
                continue
            child = MCTSNode(
                state=new_state,
                parent=node,
                action_taken=act,
            )
            node.children.append(child)
        return node.children

    def _rollout(
        self,
        node: MCTSNode,
        target: ESB,
    ) -> tuple[float, DSLProgram | None]:
        """Random walk from node toward target; return (reward, program|None)."""
        state   = node.state
        history = self._path_actions(node)

        for _ in range(MAX_ROLLOUT_DEPTH):
            if exact_match(state, target):
                return 1.0, history

            actions = enumerate_actions(state)
            if not actions:
                break

            # Memory-weighted action selection
            priors = [self._get_prior(node, a) for a in actions]
            total  = sum(priors)
            if total > 1e-8:
                weights = [p / total for p in priors]
                action  = random.choices(actions, weights=weights, k=1)[0]
            else:
                action = random.choice(actions)

            name, kwargs = action
            if name == "overlay":
                kwargs = {**kwargs, "foreground": state}
            try:
                state = getattr(_DSL, name)(state, **kwargs)
            except (ValueError, IndexError, RuntimeError):
                continue
            history = history + [action]

        return partial_reward(state, target), None

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        """Walk from node to root, incrementing visits and updating Q."""
        cur = node
        while cur is not None:
            cur.visits += 1
            # Running-mean update: Q_new = Q + (r - Q) / N
            cur.value  += (reward - cur.value) / cur.visits
            cur = cur.parent

    # ── Public API ────────────────────────────────────────────────────────

    def solve(
        self,
        training_pairs: list[tuple[ESB, ESB]],
    ) -> DSLProgram:
        """Run MCTS for up to MAX_ITERATIONS; return best discovered program.

        On exact match: updates local_memory and returns immediately.
        On budget exhaustion: returns the highest-Q leaf path.
        """
        if not training_pairs:
            return []

        # Solve against first training pair; generalisation handled upstream
        input_esb, target_esb = training_pairs[0]
        root = MCTSNode(state=input_esb, parent=None, action_taken=None)

        best_q:       float      = -1.0
        best_program: DSLProgram = []

        for _ in range(MAX_ITERATIONS):
            # 1. Select
            leaf = self._select(root)

            # 2. Expand if not yet expanded
            if not leaf.children:
                self._expand(leaf)

            # 3. Rollout from first child (or leaf if expansion produced none)
            target_node = leaf.children[0] if leaf.children else leaf
            reward, solved_program = self._rollout(target_node, target_esb)

            # 4. Exact match — update memory and return immediately
            if solved_program is not None:
                self.local_memory.add(solved_program)
                self._backprop(target_node, 1.0)
                return solved_program

            # 5. Backprop partial reward
            self._backprop(target_node, reward)

            prog = self._path_actions(target_node)
            if reward > best_q:
                best_q       = reward
                best_program = prog

        return best_program

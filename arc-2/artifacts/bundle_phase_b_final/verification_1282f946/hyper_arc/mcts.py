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
import time
from dataclasses import dataclass, field
from typing import Optional

from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL, PrimitiveCall, DSLProgram
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.cost import exact_match, partial_reward

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ITERATIONS = 2000
MAX_ROLLOUT_DEPTH = 10
MAX_GRID_DIM = 30

# Default memory weight split (local : global = 0.8 : 0.2).
# When no seed bank is present, global_prior_weight is set to 0.0 at runtime
# and local weight automatically becomes 1.0 (see MCTSEngine._get_prior).
DEFAULT_LOCAL_PRIOR_WEIGHT = 0.8
DEFAULT_GLOBAL_PRIOR_WEIGHT = 0.2

_DSL = SpatialDSL()


# ── MCTSNode ──────────────────────────────────────────────────────────────


@dataclass
class MCTSNode:
    state: ESB
    parent: Optional["MCTSNode"]
    action_taken: Optional[PrimitiveCall]
    children: list["MCTSNode"] = field(default_factory=list)
    untried_actions: Optional[list[PrimitiveCall]] = None
    visits: int = 0
    value: float = 0.0  # running mean Q

    def ucb1(self, C: float, prior: float) -> float:
        """UCB1 = Q + C * sqrt(ln(parent_N) / N) + prior.

        Returns inf for unvisited nodes (visits == 0).
        """
        if self.visits == 0:
            return float("inf")
        parent_n = self.parent.visits if self.parent else 1
        exploit = self.value
        explore = C * math.sqrt(math.log(max(parent_n, 1)) / self.visits)
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
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        actions.append(("translate", {"dx": dx, "dy": dy, "wrap": True}))

    # ── rotate ────────────────────────────────────────────────────────────
    for deg in [90, 180, 270]:
        actions.append(("rotate", {"degrees": deg}))

    # ── reflect ───────────────────────────────────────────────────────────
    for axis in ["horizontal", "vertical"]:
        actions.append(("reflect", {"axis": axis}))

    # ── extract_object ────────────────────────────────────────────────────
    seeds = [
        (0, 0),
        (0, W - 1),
        (H - 1, 0),
        (H - 1, W - 1),
        (H // 2, W // 2),
    ]
    for y, x in seeds:
        actions.append(("extract_object", {"x": x, "y": y}))

    # ── overlay ───────────────────────────────────────────────────────────
    for bm in ["overwrite", "or"]:
        # The linear DSL has no second-state register. Keep a serialisable
        # self marker for compatibility; progressive expansion drops no-ops.
        actions.append(("overlay", {"foreground": "__self__", "blend_mode": bm}))

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
        ((0, 0), (0, W - 1)),
        ((0, 0), (H - 1, 0)),
        ((0, 0), (H - 1, W - 1)),
        ((0, W - 1), (H - 1, 0)),
        ((0, W - 1), (H - 1, W - 1)),
        ((H - 1, 0), (H - 1, W - 1)),
    ]
    for (r1, c1), (r2, c2) in corner_pairs:
        for color in colors_present:
            actions.append(
                (
                    "connect_points",
                    {
                        "color": color,
                        "row1": r1,
                        "col1": c1,
                        "row2": r2,
                        "col2": c2,
                    },
                )
            )

    # ── color_remap ───────────────────────────────────────────────────────
    for src in colors_present:
        for dst in range(10):
            if src != dst:
                actions.append(("color_remap", {"mapping": {src: dst}}))

    # ── flood_fill ────────────────────────────────────────────────────────
    fill_seeds = [
        (0, 0),
        (0, W - 1),
        (H - 1, 0),
        (H - 1, W - 1),
        (H // 2, W // 2),
    ]
    for row, col in fill_seeds:
        current = int(esb.data[0, row, col].item())
        for nc in range(10):
            if nc != current:
                actions.append(
                    (
                        "flood_fill",
                        {
                            "row": row,
                            "col": col,
                            "new_color": nc,
                        },
                    )
                )

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
        local_memory: LocalTaskBuffer,
        C: float = 1.41,
        global_prior_weight: float = DEFAULT_GLOBAL_PRIOR_WEIGHT,
        max_iterations: int = MAX_ITERATIONS,
        max_rollout_depth: int = MAX_ROLLOUT_DEPTH,
        random_seed: int = 0,
    ) -> None:
        self.global_memory = global_memory
        self.local_memory = local_memory
        self.C = C
        self.global_prior_weight = global_prior_weight
        self.max_iterations = max_iterations
        self.max_rollout_depth = max_rollout_depth
        self.rng = random.Random(random_seed)
        # Derived local weight — always sums to 1.0 with global_prior_weight,
        # except in MVP mode (global=0.0) where local gets full weight of 1.0.
        self.local_prior_weight = (
            1.0 if global_prior_weight == 0.0 else DEFAULT_LOCAL_PRIOR_WEIGHT
        )

    # ── Prior computation ─────────────────────────────────────────────────

    def _get_prior(self, node: MCTSNode, action: PrimitiveCall) -> float:
        """Compute dual-memory prior for a candidate action at a node.

        local  weight: 0.8  (1.0 in MVP mode when global_prior_weight == 0.0)
        global weight: 0.2  (0.0 in MVP mode)
        Combined prior is normalised to [0, 1] so it doesn't dominate UCB1.
        """
        return self._get_prior_for_prefix(self._path_actions(node), action)

    def _get_prior_for_prefix(
        self,
        partial: DSLProgram,
        action: PrimitiveCall,
    ) -> float:
        """Compute prior mass for the exact next action after a prefix."""

        local_results = self.local_memory.query(partial)
        global_results = self.global_memory.query(partial)

        def next_action_mass(results: list[tuple[DSLProgram, float]]) -> float:
            prefix_len = len(partial)
            return sum(
                weight
                for program, weight in results
                if len(program) > prefix_len and program[prefix_len] == action
            )

        local_prior = next_action_mass(local_results) * self.local_prior_weight
        global_prior = next_action_mass(global_results) * self.global_prior_weight

        combined = local_prior + global_prior
        denominator = self.local_prior_weight + self.global_prior_weight
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
        """Traverse using UCB1 until a node can be progressively expanded."""
        while node.children and not node.untried_actions:
            scores = [
                c.ucb1(self.C, self._get_prior(node, c.action_taken))
                for c in node.children
            ]
            node = node.children[scores.index(max(scores))]
        return node

    @staticmethod
    def _apply_action(state: ESB, action: PrimitiveCall) -> ESB:
        """Apply one serialisable DSL action to a state."""
        name, raw_kwargs = action
        kwargs = dict(raw_kwargs)
        if name == "overlay" and kwargs.get("foreground") == "__self__":
            kwargs["foreground"] = state
        result = getattr(_DSL, name)(state, **kwargs)
        if result.H > MAX_GRID_DIM or result.W > MAX_GRID_DIM:
            raise ValueError(
                f"Action {name} produced invalid ARC grid size {result.H}x{result.W}"
            )
        return result

    @classmethod
    def apply_program(cls, program: DSLProgram, initial_state: ESB) -> ESB:
        """Apply a program using the same semantics used during search."""
        state = initial_state
        for action in program:
            state = cls._apply_action(state, action)
        return state

    @classmethod
    def score_program(
        cls,
        program: DSLProgram,
        training_pairs: list[tuple[ESB, ESB]],
    ) -> tuple[float, bool]:
        """Return mean partial reward and whether every pair is exact."""
        if not training_pairs:
            return 0.0, False
        rewards: list[float] = []
        all_exact = True
        for input_esb, target_esb in training_pairs:
            try:
                candidate = cls.apply_program(program, input_esb)
            except (ValueError, IndexError, RuntimeError):
                return 0.0, False
            pair_exact = exact_match(candidate, target_esb)
            all_exact = all_exact and pair_exact
            rewards.append(partial_reward(candidate, target_esb))
        return sum(rewards) / len(rewards), all_exact

    def _initialize_actions(self, node: MCTSNode) -> None:
        if node.untried_actions is None:
            node.untried_actions = enumerate_actions(node.state)
            self.rng.shuffle(node.untried_actions)

    def _expand_one(self, node: MCTSNode) -> MCTSNode:
        """Progressively add one valid, non-no-op child."""
        self._initialize_actions(node)
        while node.untried_actions:
            act = node.untried_actions.pop()
            try:
                new_state = self._apply_action(node.state, act)
            except (ValueError, IndexError, RuntimeError):
                continue
            if exact_match(new_state, node.state):
                continue
            child = MCTSNode(
                state=new_state,
                parent=node,
                action_taken=act,
            )
            node.children.append(child)
            return child
        return node

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Compatibility helper that exhausts progressive expansion."""
        self._initialize_actions(node)
        while node.untried_actions:
            before = len(node.children)
            self._expand_one(node)
            if len(node.children) == before and not node.untried_actions:
                break
        return node.children

    def _rollout(
        self,
        node: MCTSNode,
        training_pairs: list[tuple[ESB, ESB]],
        deadline: float | None = None,
    ) -> tuple[float, DSLProgram, bool]:
        """Random walk and return the best scored complete history visited."""
        state = node.state
        history = self._path_actions(node)
        best_reward, solved = self.score_program(history, training_pairs)
        best_history = list(history)
        if solved:
            return best_reward, best_history, True

        for _ in range(self.max_rollout_depth):
            if deadline is not None and time.monotonic() >= deadline:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            # Memory-weighted action selection
            priors = [self._get_prior_for_prefix(history, a) for a in actions]
            total = sum(priors)
            if total > 1e-8:
                weights = [p / total for p in priors]
                action = self.rng.choices(actions, weights=weights, k=1)[0]
            else:
                action = self.rng.choice(actions)

            try:
                state = self._apply_action(state, action)
            except (ValueError, IndexError, RuntimeError):
                continue
            history = history + [action]
            reward, solved = self.score_program(history, training_pairs)
            if reward > best_reward:
                best_reward = reward
                best_history = list(history)
            if solved:
                return reward, list(history), True

        return best_reward, best_history, False

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        """Walk from node to root, incrementing visits and updating Q."""
        cur = node
        while cur is not None:
            cur.visits += 1
            # Running-mean update: Q_new = Q + (r - Q) / N
            cur.value += (reward - cur.value) / cur.visits
            cur = cur.parent

    # ── Public API ────────────────────────────────────────────────────────

    def solve(
        self,
        training_pairs: list[tuple[ESB, ESB]],
        timeout_sec: float | None = None,
    ) -> DSLProgram:
        """Run MCTS for up to MAX_ITERATIONS; return best discovered program.

        On exact match: updates local_memory and returns immediately.
        On budget exhaustion: returns the highest-Q leaf path.
        """
        if not training_pairs:
            return []

        input_esb, _ = training_pairs[0]
        root = MCTSNode(state=input_esb, parent=None, action_taken=None)
        best_q, root_solved = self.score_program([], training_pairs)
        best_program: DSLProgram = []
        if root_solved:
            self.local_memory.add([])
            return []

        deadline = (
            time.monotonic() + timeout_sec
            if timeout_sec is not None and timeout_sec > 0
            else None
        )

        # Deterministically cover the complete one-step DSL before stochastic
        # search. This makes simple transformations reliable and establishes a
        # meaningful lower bound for deeper MCTS programs.
        root_actions = enumerate_actions(input_esb)
        for action in root_actions:
            if deadline is not None and time.monotonic() >= deadline:
                return best_program
            reward, solved = self.score_program([action], training_pairs)
            if reward > best_q:
                best_q = reward
                best_program = [action]
            if solved:
                self.local_memory.add([action])
                return [action]

        root.untried_actions = list(root_actions)
        self.rng.shuffle(root.untried_actions)

        for _ in range(self.max_iterations):
            if deadline is not None and time.monotonic() >= deadline:
                break
            # 1. Select
            leaf = self._select(root)

            # 2. Progressively expand one action
            target_node = self._expand_one(leaf)

            # 3. Roll out and retain the exact history that was scored
            reward, candidate_program, solved = self._rollout(
                target_node,
                training_pairs,
                deadline,
            )

            # 4. Exact match — update memory and return immediately
            if solved:
                self.local_memory.add(candidate_program)
                self._backprop(target_node, 1.0)
                return candidate_program

            # 5. Backprop partial reward
            self._backprop(target_node, reward)

            if reward > best_q:
                best_q = reward
                best_program = candidate_program

        return best_program

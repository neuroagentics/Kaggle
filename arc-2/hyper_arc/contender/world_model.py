"""Recursive, task-local world-model search with hyperbolic experience memory."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from hyper_arc.contender.executor import (
    ExecutionError,
    TypedExecutor,
    input_node,
    replay_program,
)
from hyper_arc.contender.experience_bank import task_fingerprint_v2
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory, MemoryNeighbor
from hyper_arc.contender.object_programs import generate_object_programs
from hyper_arc.contender.perception import perceive_task
from hyper_arc.contender.relational_plans import generate_relational_programs
from hyper_arc.contender.repair import grid_error, propose_repairs, wrap_program
from hyper_arc.contender.schemas import (
    Grid,
    Hypothesis,
    ProgramAST,
    Residual,
    TaskState,
    ValueType,
    stable_digest,
)
from hyper_arc.contender.structural_rules import generate_structural_programs


@dataclass(frozen=True)
class WorldModelConfig:
    max_depth: int = 2
    max_candidates: int = 512
    max_object_seeds: int = 192
    max_memory_seeds: int = 48
    max_relational_seeds: int = 16
    attempts: int = 2
    retrieve_memory: bool = True
    learn_memory: bool = True


@dataclass(frozen=True)
class ReasoningState:
    """One simulated world-state transition for an executable hypothesis."""

    state_id: str
    name: str
    source: str
    program: ProgramAST
    depth: int
    predictions: tuple[Grid, ...]
    residuals: tuple[Residual, ...]
    total_error: int
    parent_id: str | None = None
    memory_distance: float | None = None


@dataclass(frozen=True)
class WorldModelResult:
    task_state: TaskState
    hypotheses: tuple[Hypothesis, ...]
    reasoning_states: tuple[ReasoningState, ...]
    # Outer tuple follows test-input order; inner tuple contains up to pass@2.
    test_predictions: tuple[tuple[Grid, ...], ...]
    memory_neighbors: tuple[MemoryNeighbor, ...]
    candidates_executed: int


def _base_programs() -> tuple[tuple[str, ProgramAST], ...]:
    source = input_node()
    specifications: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
        ("identity", "identity", {}),
        ("rotate-90", "rotate", {"degrees": 90}),
        ("rotate-180", "rotate", {"degrees": 180}),
        ("rotate-270", "rotate", {"degrees": 270}),
        ("reflect-horizontal", "reflect", {"axis": "horizontal"}),
        ("reflect-vertical", "reflect", {"axis": "vertical"}),
        ("transpose", "transpose", {}),
        ("anti-transpose", "anti_transpose", {}),
        ("crop-mode", "crop_mode", {}),
        ("scale-2", "scale_integer", {"factor": 2}),
        ("scale-3", "scale_integer", {"factor": 3}),
    )
    return tuple(
        (
            name,
            ProgramAST(
                op=op,
                output_type=ValueType.GRID,
                arguments=arguments,
                children=(source,),
            ),
        )
        for name, op, arguments in specifications
    )


def _attempt_selection_order(
    hypotheses: list[Hypothesis],
) -> tuple[Hypothesis, ...]:
    """Protect promoted families from unvalidated-family slot displacement."""
    experimental_sources = {"structural-world"}
    promoted = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.provenance.get("source") not in experimental_sources
    ]
    experimental = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.provenance.get("source") in experimental_sources
    ]
    return tuple(promoted + experimental)


class RecursiveReasoningWorldModel:
    """Simulate, diagnose, repair, retrieve, and remember exact programs."""

    def __init__(
        self,
        memory: HyperbolicWorldMemory | None = None,
        *,
        config: WorldModelConfig | None = None,
    ) -> None:
        self.memory = memory or HyperbolicWorldMemory()
        self.config = config or WorldModelConfig()
        self.executor = TypedExecutor()

    def solve(self, task_id: str, task_data: Mapping[str, Any], *, deadline: float | None = None) -> WorldModelResult:
        task_state = perceive_task(
            task_id,
            task_data,
            provenance={"solver": "recursive-world-model-v1"},
        )
        fingerprint = task_fingerprint_v2(task_data)
        neighbors = (
            self.memory.retrieve(
                fingerprint,
                limit=self.config.max_memory_seeds,
                require_program=True,
            )
            if self.config.retrieve_memory
            else ()
        )
        seeds: list[tuple[str, str, ProgramAST, float | None]] = []
        for neighbor in neighbors:
            if neighbor.item.program is not None:
                seeds.append(
                    (
                        f"memory:{neighbor.item.record_id[:10]}",
                        "hyperbolic-memory",
                        neighbor.item.program,
                        neighbor.distance,
                    )
                )
        seeds.extend(
            (name, "world-prior", program, None) for name, program in _base_programs()
        )
        seeds.extend(
            (name, "object-world", program, None)
            for name, program, _penalty in generate_object_programs(task_data)[
                : self.config.max_object_seeds
            ]
        )
        seeds.extend(
            (name, "relational-world", program, None)
            for name, program in generate_relational_programs(task_data)[
                : self.config.max_relational_seeds
            ]
        )
        seeds.extend(
            (candidate.name, "structural-world", candidate.program, None)
            for candidate in generate_structural_programs(task_state)
        )

        seen_programs: set[str] = set()
        seen_worlds: set[tuple[Grid, ...]] = set()
        states: list[ReasoningState] = []
        exact: list[Hypothesis] = []
        executed = [0]

        def reason(
            name: str,
            source: str,
            program: ProgramAST,
            depth: int,
            parent_id: str | None,
            memory_distance: float | None,
        ) -> None:
            if (
                executed[0] >= self.config.max_candidates
                or program.digest in seen_programs
                or (deadline is not None and time.monotonic() >= deadline)
            ):
                return
            seen_programs.add(program.digest)
            executed[0] += 1
            try:
                predictions = tuple(
                    self.executor.execute(
                        program, pair.input, capture_trace=False
                    ).value
                    for pair in task_state.train
                )
            except (ExecutionError, ValueError):
                return
            targets = tuple(pair.output for pair in task_state.train)
            total_error = sum(
                grid_error(value, target) for value, target in zip(predictions, targets)
            )
            # Equivalent failed worlds lead to the same residual wrappers, so
            # prune them. Exact demonstration worlds are intentionally kept:
            # different programs may diverge on the unseen test input and are
            # precisely where pass@2 is useful.
            if total_error and predictions in seen_worlds:
                return
            if total_error:
                seen_worlds.add(predictions)
            residuals: list[Residual] = []
            if total_error:
                for index, (value, target) in enumerate(zip(predictions, targets)):
                    if value == target:
                        continue
                    mismatches: tuple[tuple[int, int], ...] = ()
                    if len(value) == len(target) and len(value[0]) == len(target[0]):
                        mismatches = tuple(
                            (row, col)
                            for row in range(len(target))
                            for col in range(len(target[0]))
                            if value[row][col] != target[row][col]
                        )
                    residuals.append(
                        Residual(
                            example_index=index,
                            mismatched_cells=mismatches,
                            expected_shape=(len(target), len(target[0])),
                            actual_shape=(len(value), len(value[0])),
                        )
                    )
            state_id = stable_digest((task_id, program.digest, depth, source))
            states.append(
                ReasoningState(
                    state_id=state_id,
                    name=name,
                    source=source,
                    program=program,
                    depth=depth,
                    predictions=predictions,
                    residuals=tuple(residuals),
                    total_error=total_error,
                    parent_id=parent_id,
                    memory_distance=memory_distance,
                )
            )
            if total_error == 0:
                traces, replay_residuals, replay_exact = replay_program(
                    self.executor, program, task_state.train, task_id=task_id
                )
                if not replay_exact:
                    return
                memory_support = (
                    math.exp(-memory_distance) if memory_distance is not None else 0.0
                )
                exact.append(
                    Hypothesis(
                        hypothesis_id=state_id,
                        program=program,
                        traces=traces,
                        residuals=replay_residuals,
                        exact_replay=True,
                        complexity=program.complexity,
                        confidence=min(
                            0.999,
                            0.8
                            + 0.15 * memory_support
                            + 0.04 / (1 + program.complexity),
                        ),
                        channel="recursive-world-model",
                        parent_id=parent_id,
                        provenance={
                            "seed": name,
                            "source": source,
                            "repair_depth": depth,
                            "memory_distance": memory_distance,
                        },
                    )
                )
                return
            if depth >= self.config.max_depth:
                return
            for repair in propose_repairs(predictions, targets, total_error):
                reason(
                    f"{name}+{repair.name}",
                    source,
                    wrap_program(program, repair),
                    depth + 1,
                    state_id,
                    memory_distance,
                )

        for name, source, program, memory_distance in seeds:
            if executed[0] >= self.config.max_candidates or (deadline is not None and time.monotonic() >= deadline):
                break
            reason(
                name,
                source,
                program,
                depth=0,
                parent_id=None,
                memory_distance=memory_distance,
            )

        exact.sort(
            key=lambda hypothesis: (
                -hypothesis.confidence,
                hypothesis.complexity,
                hypothesis.hypothesis_id,
            )
        )
        # Newly introduced hypothesis families are quarantined behind the
        # established solver families until they prove themselves on an
        # independent split.  This makes capability additions monotonic: an
        # experimental family can fill an unused pass@2 slot, but cannot crowd
        # two previously available exact-replay programs out of both slots.
        # Once a family clears the promotion gate it should be removed from
        # this set rather than accumulating ad-hoc confidence bonuses.
        selection_order = _attempt_selection_order(exact)

        behaviors: list[tuple[Grid, ...]] = []
        for hypothesis in selection_order:
            try:
                behavior = tuple(
                    self.executor.execute(
                        hypothesis.program, grid, capture_trace=False
                    ).value
                    for grid in task_state.test_inputs
                )
            except ExecutionError:
                continue
            if behavior not in behaviors:
                behaviors.append(behavior)
            if len(behaviors) >= self.config.attempts:
                break
        test_predictions = tuple(
            tuple(behavior[index] for behavior in behaviors)
            for index in range(len(task_state.test_inputs))
        )

        # Session learning is exact-demo only. It affects later tasks in this
        # solver instance, but is not persisted unless the caller explicitly saves it.
        if exact and self.config.learn_memory:
            self.memory.remember(
                source_task_id=task_id,
                family="recursive-world-model",
                fingerprint=fingerprint,
                program=exact[0].program,
                demonstration_exact=True,
                provenance={"hypothesis_id": exact[0].hypothesis_id},
            )
        return WorldModelResult(
            task_state=task_state,
            hypotheses=tuple(exact),
            reasoning_states=tuple(states),
            test_predictions=test_predictions,
            memory_neighbors=neighbors,
            candidates_executed=executed[0],
        )

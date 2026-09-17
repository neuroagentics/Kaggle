"""Typed, serializable records for the recursive ARC world-model loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping


Grid = tuple[tuple[int, ...], ...]
Coordinate = tuple[int, int]


class ValueType(str, Enum):
    GRID = "grid"
    OBJECT = "object"
    OBJECT_SET = "object_set"
    COLOR = "color"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    COORDINATE = "coordinate"


@dataclass(frozen=True)
class GridPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    pixels: tuple[Coordinate, ...]
    colors: tuple[int, ...]
    bounding_box: tuple[int, int, int, int]
    connectivity: int
    area: int
    holes: int = 0
    grid_ref: str = ""
    relative_pixels: tuple[Coordinate, ...] = ()
    shape_signature: str = ""
    color_role: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    kind: str
    source_id: str
    target_id: str
    grid_ref: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridState:
    grid_ref: str
    grid: Grid
    height: int
    width: int
    palette: tuple[int, ...]
    background_candidates: tuple[int, ...]
    objects: tuple[ObjectState, ...]
    relations: tuple[Relation, ...]
    symmetries: tuple[str, ...]
    periodicity: tuple[tuple[str, int], ...]
    separator_rows: tuple[int, ...]
    separator_columns: tuple[int, ...]
    canonical_signature: str


@dataclass(frozen=True)
class ObjectCorrespondence:
    source_id: str
    target_id: str
    evidence: str
    translation: Coordinate
    color_changed: bool


@dataclass(frozen=True)
class ExampleDelta:
    example_index: int
    input_ref: str
    output_ref: str
    correspondences: tuple[ObjectCorrespondence, ...]
    added_object_ids: tuple[str, ...]
    removed_object_ids: tuple[str, ...]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]


@dataclass(frozen=True)
class TaskState:
    task_id: str
    train: tuple[GridPair, ...]
    test_inputs: tuple[Grid, ...]
    objects: tuple[ObjectState, ...] = ()
    relations: tuple[Relation, ...] = ()
    palette: tuple[int, ...] = ()
    grids: tuple[GridState, ...] = ()
    deltas: tuple[ExampleDelta, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramAST:
    op: str
    output_type: ValueType
    arguments: Mapping[str, Any] = field(default_factory=dict)
    children: tuple["ProgramAST", ...] = ()
    preconditions: tuple["ProgramAST", ...] = ()
    postconditions: tuple["ProgramAST", ...] = ()

    @property
    def digest(self) -> str:
        return stable_digest(self)

    @property
    def complexity(self) -> int:
        leaf_cost = 0 if self.op in {"input", "load", "literal"} else 1
        control_cost = 1 if self.op in {"if", "map_objects", "render_objects"} else 0
        return (
            leaf_cost
            + control_cost
            + sum(child.complexity for child in self.children)
            + sum(condition.complexity for condition in self.preconditions)
            + sum(condition.complexity for condition in self.postconditions)
        )


@dataclass(frozen=True)
class TraceStep:
    node_digest: str
    op: str
    input_digests: tuple[str, ...]
    output_digest: str
    elapsed_ms: float
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    task_id: str
    example_index: int
    steps: tuple[TraceStep, ...]
    prediction: Grid
    exact: bool


@dataclass(frozen=True)
class Residual:
    example_index: int
    mismatched_cells: tuple[Coordinate, ...]
    missing_objects: tuple[str, ...] = ()
    extra_objects: tuple[str, ...] = ()
    violated_relations: tuple[str, ...] = ()
    expected_shape: tuple[int, int] | None = None
    actual_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    program: ProgramAST
    traces: tuple[Trace, ...]
    residuals: tuple[Residual, ...]
    exact_replay: bool
    complexity: int
    confidence: float
    channel: str
    parent_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    source_task_ids: tuple[str, ...]
    task_signature: str
    program: ProgramAST
    bindings: Mapping[str, Any]
    invariants: tuple[str, ...]
    failure_modes: tuple[str, ...]
    exact_replay: bool
    validation_scope: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return stable_digest(self)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def program_to_dict(program: ProgramAST) -> dict[str, Any]:
    return to_jsonable(program)


def program_from_dict(payload: Mapping[str, Any]) -> ProgramAST:
    allowed = {
        "op",
        "output_type",
        "arguments",
        "children",
        "preconditions",
        "postconditions",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown ProgramAST fields: {sorted(unknown)}")
    try:
        op = payload["op"]
        output_type = ValueType(payload["output_type"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ProgramAST requires a valid op and output_type") from exc
    if not isinstance(op, str) or not op:
        raise ValueError("ProgramAST op must be a non-empty string")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("ProgramAST arguments must be a mapping")

    def children(name: str) -> tuple[ProgramAST, ...]:
        values = payload.get(name, [])
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, Mapping) for value in values
        ):
            raise ValueError(f"ProgramAST {name} must contain program mappings")
        return tuple(program_from_dict(value) for value in values)

    return ProgramAST(
        op=op,
        output_type=output_type,
        arguments=dict(arguments),
        children=children("children"),
        preconditions=children("preconditions"),
        postconditions=children("postconditions"),
    )

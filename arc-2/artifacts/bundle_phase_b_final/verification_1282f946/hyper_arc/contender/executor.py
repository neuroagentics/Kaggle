"""Typed multi-register execution over the existing deterministic DSL leaves."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping

from hyper_arc.contender.perception import canonical_grid, perceive_grid
from hyper_arc.contender.schemas import (
    Grid,
    GridPair,
    ObjectState,
    ProgramAST,
    Residual,
    Trace,
    TraceStep,
    ValueType,
    stable_digest,
)
from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL


class ExecutionError(ValueError):
    """Raised when a typed program is invalid or cannot execute safely."""


@dataclass(frozen=True)
class ExecutionResult:
    value: Any
    value_type: ValueType
    trace: Trace


GRID_PRIMITIVES = {
    "identity",
    "translate",
    "rotate",
    "reflect",
    "extract_object",
    "crop_to_bbox",
    "crop_mode",
    "scale_integer",
    "tile",
    "connect_points",
    "color_remap",
    "flood_fill",
    "symmetrize",
    "transpose",
    "anti_transpose",
    "edge_marker_motif_propagation",
    "keyed_object_frames",
    "frame_signal_repeat",
    "component_recolor",
}


def _grid_digest(grid: Grid) -> str:
    return stable_digest(grid)


@lru_cache(maxsize=2_048)
def _runtime_perception(grid: Grid):
    """Reuse immutable perception for repeated candidate replay on one grid."""
    reference = f"runtime:{_grid_digest(grid)[:12]}"
    return perceive_grid(grid, reference)


def _to_grid(value: Any) -> Grid:
    if isinstance(value, ESB):
        value = value.to_grid()
    return canonical_grid(value)


def _apply_grid_primitive(op: str, grid: Grid, arguments: Mapping[str, Any]) -> Grid:
    if op == "identity":
        return tuple(tuple(row) for row in grid)
    if op == "transpose":
        return tuple(tuple(row) for row in zip(*grid))
    if op == "anti_transpose":
        return tuple(tuple(row)[::-1] for row in zip(*grid[::-1]))
    if op == "component_recolor":
        from hyper_arc.contender.structural_rules import component_features

        state = _runtime_perception(grid)
        rules = arguments.get("rules", [])
        if not isinstance(rules, (list, tuple)):
            raise ExecutionError("component_recolor rules must be a sequence")
        output = [list(row) for row in grid]
        for item in state.objects:
            attributes = component_features(
                item, height=state.height, width=state.width
            )
            for rule in rules:
                if not isinstance(rule, Mapping) or not isinstance(
                    rule.get("when"), Mapping
                ):
                    raise ExecutionError("Invalid component_recolor rule")
                if all(attributes.get(key) == value for key, value in rule["when"].items()):
                    color = int(rule["color"])
                    if not 0 <= color <= 9:
                        raise ExecutionError("component_recolor color must be 0-9")
                    for row, col in item.pixels:
                        output[row][col] = color
                    break
        return tuple(tuple(row) for row in output)
    if op in {
        "edge_marker_motif_propagation",
        "keyed_object_frames",
        "frame_signal_repeat",
    }:
        from hyper_arc.contender.relational_primitives import (
            edge_marker_motif_propagation,
            frame_signal_repeat,
            keyed_object_frames,
        )

        try:
            if op == "edge_marker_motif_propagation":
                return edge_marker_motif_propagation(grid)
            if op == "keyed_object_frames":
                return keyed_object_frames(grid)
            return frame_signal_repeat(grid, arguments)
        except (TypeError, ValueError, IndexError) as exc:
            raise ExecutionError(f"{op} failed: {exc}") from exc
    esb = ESB.from_grid([list(row) for row in grid])
    if op == "crop_mode":
        counts: dict[int, int] = {}
        for row in grid:
            for color in row:
                counts[color] = counts.get(color, 0) + 1
        background = min(counts, key=lambda color: (-counts[color], color))
        return _to_grid(SpatialDSL.crop_to_bbox(esb, background=background))
    function = getattr(SpatialDSL, op, None)
    if function is None or op == "overlay":
        raise ExecutionError(f"Unsupported grid primitive: {op}")
    try:
        return _to_grid(function(esb, **dict(arguments)))
    except (TypeError, ValueError, IndexError, RuntimeError) as exc:
        raise ExecutionError(f"{op} failed: {exc}") from exc


class TypedExecutor:
    """Execute a closed typed AST with task-local named registers."""

    def __init__(
        self,
        *,
        max_nodes: int = 512,
        max_depth: int = 64,
        max_trace_steps: int = 10_000,
        max_map_items: int = 900,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_trace_steps = max_trace_steps
        self.max_map_items = max_map_items

    def execute(
        self,
        program: ProgramAST,
        input_grid: Any,
        *,
        task_id: str = "",
        example_index: int = 0,
        capture_trace: bool = True,
    ) -> ExecutionResult:
        self._validate_program_size(program)
        canonical_input = _to_grid(input_grid)
        registers: dict[str, tuple[ValueType, Any]] = {
            "input": (ValueType.GRID, canonical_input)
        }
        steps: list[TraceStep] = []
        step_count = [0]
        value_type, value = self._evaluate(
            program,
            canonical_input,
            registers,
            steps,
            step_count,
            capture_trace,
        )
        if value_type != program.output_type:
            raise ExecutionError(
                f"Program declared {program.output_type}, produced {value_type}"
            )
        prediction = value if value_type == ValueType.GRID else canonical_input
        return ExecutionResult(
            value=value,
            value_type=value_type,
            trace=Trace(
                task_id=task_id,
                example_index=example_index,
                steps=tuple(steps),
                prediction=prediction,
                exact=False,
            ),
        )

    def _evaluate(
        self,
        node: ProgramAST,
        input_grid: Grid,
        registers: dict[str, tuple[ValueType, Any]],
        steps: list[TraceStep],
        step_count: list[int],
        capture_trace: bool,
    ) -> tuple[ValueType, Any]:
        if step_count[0] >= self.max_trace_steps:
            raise ExecutionError("Execution trace budget exhausted")
        step_count[0] += 1
        started = time.perf_counter() if capture_trace else 0.0
        self._check_conditions(
            node.preconditions,
            "precondition",
            input_grid,
            registers,
            steps,
            step_count,
            capture_trace,
        )
        child_values: list[tuple[ValueType, Any]] = []
        if node.op not in {"sequence", "store", "if", "map_objects"}:
            child_values = [
                self._evaluate(
                    child,
                    input_grid,
                    registers,
                    steps,
                    step_count,
                    capture_trace,
                )
                for child in node.children
            ]

        if node.op == "input":
            self._expect_children(node, 0)
            result = ValueType.GRID, input_grid
        elif node.op == "literal":
            self._expect_children(node, 0)
            result = node.output_type, self._literal_value(node)
        elif node.op == "load":
            self._expect_children(node, 0)
            name = self._load_name(node)
            if name not in registers:
                raise ExecutionError(f"Register has not been assigned: {name}")
            result = registers[name]
        elif node.op == "store":
            self._expect_children(node, 1)
            result = self._evaluate(
                node.children[0],
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
            child_values.append(result)
            name = self._store_name(node)
            registers[name] = result
        elif node.op == "sequence":
            if not node.children:
                raise ExecutionError("sequence requires at least one child")
            result = ValueType.GRID, input_grid
            for child in node.children:
                result = self._evaluate(
                    child,
                    input_grid,
                    registers,
                    steps,
                    step_count,
                    capture_trace,
                )
                child_values.append(result)
        elif node.op == "if":
            self._expect_children(node, 3)
            condition = self._evaluate(
                node.children[0],
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
            if condition[0] != ValueType.BOOLEAN:
                raise ExecutionError("if condition must be boolean")
            branch = node.children[1] if condition[1] else node.children[2]
            result = self._evaluate(
                branch,
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
            child_values.extend((condition, result))
        elif node.op == "map_objects":
            self._expect_children(node, 2)
            source = self._evaluate(
                node.children[0],
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
            if source[0] != ValueType.OBJECT_SET:
                raise ExecutionError("map_objects requires an object set")
            if len(source[1]) > self.max_map_items:
                raise ExecutionError(
                    f"map_objects item budget exceeded: {len(source[1])}"
                )
            name = self._iteration_name(node)
            previous = registers.get(name)
            mapped: list[ObjectState] = []
            try:
                for item in source[1]:
                    registers[name] = (ValueType.OBJECT, item)
                    item_result = self._evaluate(
                        node.children[1],
                        input_grid,
                        registers,
                        steps,
                        step_count,
                        capture_trace,
                    )
                    if item_result[0] != ValueType.OBJECT:
                        raise ExecutionError("map_objects body must produce an object")
                    mapped.append(item_result[1])
            finally:
                if previous is None:
                    registers.pop(name, None)
                else:
                    registers[name] = previous
            result = ValueType.OBJECT_SET, tuple(mapped)
            child_values.append(source)
        elif node.op == "overlay":
            self._expect_children(node, 2)
            self._expect_grid_children(node, child_values)
            background = ESB.from_grid([list(row) for row in child_values[0][1]])
            foreground = ESB.from_grid([list(row) for row in child_values[1][1]])
            try:
                value = SpatialDSL.overlay(
                    background,
                    foreground,
                    blend_mode=str(node.arguments.get("blend_mode", "overwrite")),
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ExecutionError(f"overlay failed: {exc}") from exc
            result = ValueType.GRID, _to_grid(value)
        elif node.op == "objects":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.GRID,))
            grid = child_values[0][1]
            result = ValueType.OBJECT_SET, _runtime_perception(grid).objects
        elif node.op == "select_objects":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.OBJECT_SET,))
            selected = tuple(
                item
                for item in child_values[0][1]
                if self._matches_object(item, node.arguments.get("where", {}))
            )
            extremum = node.arguments.get("extremum")
            if extremum in {"largest", "smallest"} and selected:
                target = (max if extremum == "largest" else min)(
                    item.area for item in selected
                )
                selected = tuple(item for item in selected if item.area == target)
            elif extremum is not None:
                raise ExecutionError("extremum must be largest or smallest")
            result = ValueType.OBJECT_SET, selected
        elif node.op == "first_object":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.OBJECT_SET,))
            if not child_values[0][1]:
                raise ExecutionError("first_object cannot select from an empty set")
            result = ValueType.OBJECT, child_values[0][1][0]
        elif node.op == "count":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.OBJECT_SET,))
            result = ValueType.INTEGER, len(child_values[0][1])
        elif node.op == "object_attr":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.OBJECT,))
            result = (
                node.output_type,
                self._object_attribute(
                    child_values[0][1], node.arguments.get("name"), node.output_type
                ),
            )
        elif node.op == "translate_object":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.OBJECT,))
            item = child_values[0][1]
            dy = self._integer_argument(node, "dy", 0)
            dx = self._integer_argument(node, "dx", 0)
            top, left, bottom, right = item.bounding_box
            result = (
                ValueType.OBJECT,
                replace(
                    item,
                    pixels=tuple((row + dy, col + dx) for row, col in item.pixels),
                    bounding_box=(top + dy, left + dx, bottom + dy, right + dx),
                ),
            )
        elif node.op == "recolor_object":
            if len(child_values) not in (1, 2):
                raise ExecutionError("recolor_object expects one or two children")
            expected = (
                (ValueType.OBJECT,)
                if len(child_values) == 1
                else (ValueType.OBJECT, ValueType.COLOR)
            )
            self._expect_types(node, child_values, expected)
            color = (
                child_values[1][1]
                if len(child_values) == 2
                else self._color_argument(node, "color")
            )
            result = ValueType.OBJECT, replace(child_values[0][1], colors=(color,))
        elif node.op == "blank_canvas":
            self._expect_children(node, 0)
            height = self._dimension_argument(node, "height")
            width = self._dimension_argument(node, "width")
            color = self._color_argument(node, "color", default=0)
            result = (
                ValueType.GRID,
                tuple(tuple(color for _ in range(width)) for _ in range(height)),
            )
        elif node.op == "canvas_like":
            if len(child_values) not in (1, 2):
                raise ExecutionError("canvas_like expects one or two children")
            expected = (
                (ValueType.GRID,)
                if len(child_values) == 1
                else (ValueType.GRID, ValueType.COLOR)
            )
            self._expect_types(node, child_values, expected)
            color = (
                child_values[1][1]
                if len(child_values) == 2
                else self._color_argument(node, "color", default=0)
            )
            source = child_values[0][1]
            result = ValueType.GRID, tuple(tuple(color for _ in row) for row in source)
        elif node.op == "background_color":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.GRID,))
            background = _runtime_perception(child_values[0][1]).background_candidates[
                0
            ]
            result = ValueType.COLOR, background
        elif node.op == "erase_objects":
            if len(child_values) not in (2, 3):
                raise ExecutionError("erase_objects expects two or three children")
            expected = (
                (ValueType.GRID, ValueType.OBJECT_SET)
                if len(child_values) == 2
                else (ValueType.GRID, ValueType.OBJECT_SET, ValueType.COLOR)
            )
            self._expect_types(node, child_values, expected)
            color = (
                child_values[2][1]
                if len(child_values) == 3
                else self._color_argument(node, "color", default=0)
            )
            result = (
                ValueType.GRID,
                self._erase_objects(child_values[0][1], child_values[1][1], color),
            )
        elif node.op == "render_objects":
            self._expect_children(node, 2)
            self._expect_types(
                node, child_values, (ValueType.GRID, ValueType.OBJECT_SET)
            )
            result = (
                ValueType.GRID,
                self._render_objects(child_values[0][1], child_values[1][1]),
            )
        elif node.op == "crop_object":
            if len(child_values) not in (1, 2):
                raise ExecutionError("crop_object expects one or two children")
            expected = (
                (ValueType.OBJECT,)
                if len(child_values) == 1
                else (ValueType.OBJECT, ValueType.COLOR)
            )
            self._expect_types(node, child_values, expected)
            background = (
                child_values[1][1]
                if len(child_values) == 2
                else self._color_argument(node, "background", default=0)
            )
            result = ValueType.GRID, self._crop_object(child_values[0][1], background)
        elif node.op in {"grid_height", "grid_width"}:
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.GRID,))
            grid = child_values[0][1]
            result = (
                ValueType.INTEGER,
                len(grid) if node.op == "grid_height" else len(grid[0]),
            )
        elif node.op == "equals":
            self._expect_children(node, 2)
            result = ValueType.BOOLEAN, child_values[0] == child_values[1]
        elif node.op == "compare":
            self._expect_children(node, 2)
            self._expect_types(
                node, child_values, (ValueType.INTEGER, ValueType.INTEGER)
            )
            result = (
                ValueType.BOOLEAN,
                self._compare(
                    child_values[0][1],
                    child_values[1][1],
                    node.arguments.get("operator"),
                ),
            )
        elif node.op in {"and", "or"}:
            if not child_values:
                raise ExecutionError(f"{node.op} requires at least one child")
            self._expect_types(
                node, child_values, tuple(ValueType.BOOLEAN for _ in child_values)
            )
            result = (
                ValueType.BOOLEAN,
                all(item[1] for item in child_values)
                if node.op == "and"
                else any(item[1] for item in child_values),
            )
        elif node.op == "not":
            self._expect_children(node, 1)
            self._expect_types(node, child_values, (ValueType.BOOLEAN,))
            result = ValueType.BOOLEAN, not child_values[0][1]
        elif node.op in GRID_PRIMITIVES:
            self._expect_children(node, 1)
            self._expect_grid_children(node, child_values)
            result = (
                ValueType.GRID,
                _apply_grid_primitive(node.op, child_values[0][1], node.arguments),
            )
        else:
            raise ExecutionError(f"Unknown or disallowed AST operation: {node.op}")

        if result[0] != node.output_type:
            raise ExecutionError(
                f"Node {node.op} declared {node.output_type}, produced {result[0]}"
            )
        previous_result = registers.get("result")
        registers["result"] = result
        try:
            self._check_conditions(
                node.postconditions,
                "postcondition",
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
        finally:
            if previous_result is None:
                registers.pop("result", None)
            else:
                registers["result"] = previous_result
        if capture_trace:
            elapsed_ms = (time.perf_counter() - started) * 1000
            input_digests = tuple(stable_digest(value) for _, value in child_values)
            steps.append(
                TraceStep(
                    node_digest=node.digest,
                    op=node.op,
                    input_digests=input_digests,
                    output_digest=stable_digest(result[1]),
                    elapsed_ms=elapsed_ms,
                    attributes={"register": node.arguments.get("name")}
                    if node.op in {"load", "store"}
                    else {},
                )
            )
        return result

    @staticmethod
    def _expect_children(node: ProgramAST, count: int) -> None:
        if len(node.children) != count:
            raise ExecutionError(
                f"{node.op} expects {count} children, found {len(node.children)}"
            )

    def _validate_program_size(self, program: ProgramAST) -> None:
        stack = [(program, 1)]
        count = 0
        while stack:
            node, depth = stack.pop()
            count += 1
            if count > self.max_nodes:
                raise ExecutionError("Program node budget exceeded")
            if depth > self.max_depth:
                raise ExecutionError("Program depth budget exceeded")
            stack.extend(
                (child, depth + 1)
                for child in (node.children + node.preconditions + node.postconditions)
            )

    @staticmethod
    def _expect_grid_children(
        node: ProgramAST, children: list[tuple[ValueType, Any]]
    ) -> None:
        if any(value_type != ValueType.GRID for value_type, _ in children):
            raise ExecutionError(f"{node.op} accepts only grid inputs")

    @staticmethod
    def _expect_types(
        node: ProgramAST,
        children: list[tuple[ValueType, Any]],
        expected: tuple[ValueType, ...],
    ) -> None:
        actual = tuple(item[0] for item in children)
        if actual != expected:
            raise ExecutionError(f"{node.op} expects {expected}, found {actual}")

    @staticmethod
    def _load_name(node: ProgramAST) -> str:
        name = node.arguments.get("name")
        if not isinstance(name, str) or not name:
            raise ExecutionError("Register name must be a non-empty string")
        return name

    @classmethod
    def _store_name(cls, node: ProgramAST) -> str:
        name = cls._load_name(node)
        if name in {"input", "result"}:
            raise ExecutionError(f"Cannot overwrite reserved register: {name}")
        return name

    @classmethod
    def _iteration_name(cls, node: ProgramAST) -> str:
        name = str(node.arguments.get("item_register", "item"))
        if not name or name in {"input", "result"}:
            raise ExecutionError("Invalid map_objects item register")
        return name

    def _check_conditions(
        self,
        conditions: tuple[ProgramAST, ...],
        label: str,
        input_grid: Grid,
        registers: dict[str, tuple[ValueType, Any]],
        steps: list[TraceStep],
        step_count: list[int],
        capture_trace: bool,
    ) -> None:
        for condition in conditions:
            result = self._evaluate(
                condition,
                input_grid,
                registers,
                steps,
                step_count,
                capture_trace,
            )
            if result[0] != ValueType.BOOLEAN or result[1] is not True:
                raise ExecutionError(f"{label} failed for {condition.op}")

    @staticmethod
    def _literal_value(node: ProgramAST) -> Any:
        value = node.arguments.get("value")
        if node.output_type == ValueType.GRID:
            return _to_grid(value)
        if node.output_type == ValueType.BOOLEAN and isinstance(value, bool):
            return value
        if (
            node.output_type == ValueType.INTEGER
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return value
        if (
            node.output_type == ValueType.COLOR
            and isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 9
        ):
            return value
        if (
            node.output_type == ValueType.COORDINATE
            and isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            )
        ):
            return tuple(value)
        raise ExecutionError(f"Invalid literal for {node.output_type}: {value!r}")

    @staticmethod
    def _matches_object(item: ObjectState, where: Any) -> bool:
        if not isinstance(where, Mapping):
            raise ExecutionError("select_objects where must be a mapping")
        available = {
            "area": item.area,
            "holes": item.holes,
            "color": item.colors[0] if item.colors else None,
            "color_role": item.color_role,
            "shape_signature": item.shape_signature,
            **dict(item.attributes),
        }
        for key, expected in where.items():
            if key == "min_area" and item.area < expected:
                return False
            if key == "max_area" and item.area > expected:
                return False
            if key not in {"min_area", "max_area"} and available.get(key) != expected:
                return False
        return True

    @staticmethod
    def _object_attribute(item: ObjectState, name: Any, output_type: ValueType) -> Any:
        top, left, bottom, right = item.bounding_box
        values = {
            "area": item.area,
            "holes": item.holes,
            "color": item.colors[0] if item.colors else None,
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
            **dict(item.attributes),
        }
        if name not in values:
            raise ExecutionError(f"Unknown object attribute: {name}")
        value = values[name]
        if (
            output_type == ValueType.INTEGER
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return value
        if (
            output_type == ValueType.COLOR
            and isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 9
        ):
            return value
        if output_type == ValueType.BOOLEAN and isinstance(value, bool):
            return value
        raise ExecutionError(f"Object attribute {name} is not {output_type}")

    @staticmethod
    def _integer_argument(
        node: ProgramAST, name: str, default: int | None = None
    ) -> int:
        value = node.arguments.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ExecutionError(f"{node.op} argument {name} must be an integer")
        return value

    @classmethod
    def _dimension_argument(cls, node: ProgramAST, name: str) -> int:
        value = cls._integer_argument(node, name)
        if not 1 <= value <= 30:
            raise ExecutionError(f"{name} must be between 1 and 30")
        return value

    @classmethod
    def _color_argument(
        cls, node: ProgramAST, name: str, default: int | None = None
    ) -> int:
        value = cls._integer_argument(node, name, default)
        if not 0 <= value <= 9:
            raise ExecutionError(f"{name} must be an ARC color 0-9")
        return value

    @staticmethod
    def _render_objects(grid: Grid, objects: tuple[ObjectState, ...]) -> Grid:
        canvas = [list(row) for row in grid]
        height, width = len(canvas), len(canvas[0])
        for item in objects:
            color = item.colors[0] if item.colors else 0
            for row, col in item.pixels:
                if 0 <= row < height and 0 <= col < width:
                    canvas[row][col] = color
        return tuple(tuple(row) for row in canvas)

    @staticmethod
    def _erase_objects(
        grid: Grid, objects: tuple[ObjectState, ...], color: int
    ) -> Grid:
        canvas = [list(row) for row in grid]
        height, width = len(canvas), len(canvas[0])
        for item in objects:
            for row, col in item.pixels:
                if 0 <= row < height and 0 <= col < width:
                    canvas[row][col] = color
        return tuple(tuple(row) for row in canvas)

    @staticmethod
    def _crop_object(item: ObjectState, background: int) -> Grid:
        top, left, bottom, right = item.bounding_box
        height, width = bottom - top + 1, right - left + 1
        if height <= 0 or width <= 0 or height > 30 or width > 30:
            raise ExecutionError("Object bounding box cannot form an ARC canvas")
        canvas = [[background for _ in range(width)] for _ in range(height)]
        color = item.colors[0] if item.colors else background
        for row, col in item.pixels:
            canvas[row - top][col - left] = color
        return tuple(tuple(row) for row in canvas)

    @staticmethod
    def _compare(first: int, second: int, operator: Any) -> bool:
        operations = {
            "lt": first < second,
            "le": first <= second,
            "eq": first == second,
            "ge": first >= second,
            "gt": first > second,
        }
        if operator not in operations:
            raise ExecutionError(f"Unsupported comparison operator: {operator}")
        return operations[operator]


def replay_program(
    executor: TypedExecutor,
    program: ProgramAST,
    train_pairs: tuple[GridPair, ...],
    *,
    task_id: str,
) -> tuple[tuple[Trace, ...], tuple[Residual, ...], bool]:
    """Execute all demonstrations; exact replay is the only acceptance signal."""
    if not train_pairs:
        raise ExecutionError("Exact replay requires at least one training pair")
    if program.output_type != ValueType.GRID:
        raise ExecutionError("Exact replay requires a grid-producing program")
    traces: list[Trace] = []
    residuals: list[Residual] = []
    for index, pair in enumerate(train_pairs):
        result = executor.execute(
            program, pair.input, task_id=task_id, example_index=index
        )
        prediction = _to_grid(result.value)
        exact = prediction == pair.output
        traces.append(
            Trace(
                task_id=task_id,
                example_index=index,
                steps=result.trace.steps,
                prediction=prediction,
                exact=exact,
            )
        )
        if not exact:
            mismatches: list[tuple[int, int]] = []
            if len(prediction) == len(pair.output) and len(prediction[0]) == len(
                pair.output[0]
            ):
                mismatches = [
                    (row, col)
                    for row in range(len(pair.output))
                    for col in range(len(pair.output[0]))
                    if prediction[row][col] != pair.output[row][col]
                ]
            residuals.append(
                Residual(
                    example_index=index,
                    mismatched_cells=tuple(mismatches),
                    expected_shape=(len(pair.output), len(pair.output[0])),
                    actual_shape=(len(prediction), len(prediction[0])),
                )
            )
    return tuple(traces), tuple(residuals), not residuals


def input_node() -> ProgramAST:
    return ProgramAST(op="input", output_type=ValueType.GRID)


def legacy_name_to_ast(
    name: str, *, color_map: Mapping[int, int] | None = None
) -> ProgramAST:
    """Translate one deterministic-channel name into an executable typed AST."""
    remap = name.endswith("+remap")
    if remap:
        name = name[: -len("+remap")]
        if color_map is None:
            raise ExecutionError("A legacy remap requires the inferred color_map")
    suffix_match = re.search(r"\+(upscale(2|3|4)|tile([1-4])x([1-4]))$", name)
    suffix = suffix_match.group(1) if suffix_match else None
    if suffix_match:
        name = name[: suffix_match.start()]
    node = input_node()
    if name.startswith("crop0+"):
        node = ProgramAST(
            op="crop_to_bbox",
            output_type=ValueType.GRID,
            arguments={"background": 0},
            children=(node,),
        )
        name = name[len("crop0+") :]
    elif name.startswith("crop_mode+"):
        node = ProgramAST(op="crop_mode", output_type=ValueType.GRID, children=(node,))
        name = name[len("crop_mode+") :]
    geometry: dict[str, tuple[str, Mapping[str, Any]]] = {
        "identity": ("identity", {}),
        "rotate90": ("rotate", {"degrees": 90}),
        "rotate180": ("rotate", {"degrees": 180}),
        "rotate270": ("rotate", {"degrees": 270}),
        "reflect_horizontal": ("reflect", {"axis": "horizontal"}),
        "reflect_vertical": ("reflect", {"axis": "vertical"}),
        "transpose": ("transpose", {}),
        "anti_transpose": ("anti_transpose", {}),
    }
    if name not in geometry:
        raise ExecutionError(f"Unknown deterministic transform name: {name}")
    op, arguments = geometry[name]
    node = ProgramAST(
        op=op, output_type=ValueType.GRID, arguments=arguments, children=(node,)
    )
    if suffix:
        if suffix.startswith("upscale"):
            node = ProgramAST(
                op="scale_integer",
                output_type=ValueType.GRID,
                arguments={"factor": int(suffix[-1])},
                children=(node,),
            )
        else:
            tile_match = re.fullmatch(r"tile([1-4])x([1-4])", suffix)
            if tile_match is None:
                raise ExecutionError(f"Invalid tile suffix: {suffix}")
            node = ProgramAST(
                op="tile",
                output_type=ValueType.GRID,
                arguments={
                    "repeats_h": int(tile_match.group(1)),
                    "repeats_w": int(tile_match.group(2)),
                },
                children=(node,),
            )
    if remap:
        node = ProgramAST(
            op="color_remap",
            output_type=ValueType.GRID,
            arguments={"mapping": dict(color_map or {})},
            children=(node,),
        )
    return node

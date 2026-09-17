"""Task-state-derived structural rules for conservative typed expansion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

from hyper_arc.contender.perception import perceive_grid
from hyper_arc.contender.schemas import Grid, ObjectState, ProgramAST, TaskState, ValueType


@dataclass(frozen=True)
class StructuralProgramCandidate:
    name: str
    program: ProgramAST
    evidence_count: int


def component_features(
    item: ObjectState, *, height: int, width: int
) -> dict[str, Any]:
    top, left, bottom, right = item.bounding_box
    return {
        "color": item.colors[0] if item.colors else -1,
        "area": item.area,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "holes": item.holes,
        "shape": item.shape_signature,
        "is_line": bool(item.attributes.get("is_line")),
        "is_square": bool(item.attributes.get("is_square")),
        "touches_border": top == 0 or left == 0 or bottom == height - 1 or right == width - 1,
    }


def _component_observations(task_state: TaskState):
    observations = []
    for example_index, pair in enumerate(task_state.train):
        source = perceive_grid(pair.input, f"structural:{task_state.task_id}:{example_index}")
        target = pair.output
        if (source.height, source.width) != (len(target), len(target[0])):
            return ()
        background = source.background_candidates[0]
        covered = {pixel for item in source.objects for pixel in item.pixels}
        if any(
            target[row][col] != pair.input[row][col]
            for row in range(source.height)
            for col in range(source.width)
            if (row, col) not in covered
        ):
            return ()
        for item in source.objects:
            colors = {target[row][col] for row, col in item.pixels}
            if len(colors) != 1:
                return ()
            source_color = item.colors[0] if item.colors else background
            target_color = next(iter(colors))
            observations.append(
                (
                    component_features(item, height=source.height, width=source.width),
                    "self" if target_color == source_color else target_color,
                    background,
                )
            )
    return tuple(observations)


def _feature_sets() -> tuple[tuple[str, ...], ...]:
    primary = ("color", "area", "shape", "holes", "height", "width", "touches_border")
    values: list[tuple[str, ...]] = [(name,) for name in primary]
    values.extend(combinations(primary, 2))
    values.extend(
        (
            ("height", "width", "holes"),
            ("color", "height", "width"),
            ("color", "shape", "touches_border"),
        )
    )
    return tuple(values)


def generate_component_recolor_programs(
    task_state: TaskState,
) -> tuple[StructuralProgramCandidate, ...]:
    """Infer minimal component-attribute rules that exactly replay demonstrations."""
    observations = _component_observations(task_state)
    if not observations:
        return ()
    source = ProgramAST(op="input", output_type=ValueType.GRID)
    candidates: list[StructuralProgramCandidate] = []
    seen_rules: set[tuple] = set()
    for features in _feature_sets():
        mapping: dict[tuple[Any, ...], int | str] = {}
        valid = True
        for attributes, target_color, _background in observations:
            key = tuple(attributes[name] for name in features)
            if key in mapping and mapping[key] != target_color:
                valid = False
                break
            mapping[key] = target_color
        if not valid:
            continue
        changed = []
        for key, target_color in sorted(mapping.items(), key=lambda item: repr(item[0])):
            conditions = dict(zip(features, key))
            # Self is the default behavior and need not consume a rule.
            if target_color == "self":
                continue
            changed.append({"when": conditions, "color": target_color})
        if not changed:
            continue
        canonical = tuple(
            (tuple(sorted(rule["when"].items())), rule["color"]) for rule in changed
        )
        if canonical in seen_rules:
            continue
        seen_rules.add(canonical)
        program = ProgramAST(
            op="component_recolor",
            output_type=ValueType.GRID,
            arguments={"features": list(features), "rules": changed},
            children=(source,),
        )
        candidates.append(
            StructuralProgramCandidate(
                name="component-recolor:" + "+".join(features),
                program=program,
                evidence_count=len(observations),
            )
        )
    return tuple(candidates)


def generate_structural_programs(
    task_state: TaskState,
) -> tuple[StructuralProgramCandidate, ...]:
    return generate_component_recolor_programs(task_state)

"""Exact object-level program generation over the typed contender grammar."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping

from hyper_arc.contender.executor import ExecutionError, TypedExecutor, input_node
from hyper_arc.contender.perception import perceive_grid, perceive_task
from hyper_arc.contender.schemas import Grid, GridPair, ProgramAST, ValueType


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    where: Mapping[str, Any]
    extremum: str | None = None
    penalty: int = 0


@dataclass(frozen=True)
class ObjectProgramCandidate:
    name: str
    program: ProgramAST
    selector_penalty: int

    @cached_property
    def complexity(self) -> int:
        return self.program.complexity + self.selector_penalty

    def predict(self, grid: list[list[int]] | Grid) -> list[list[int]]:
        value = TypedExecutor().execute(self.program, grid).value
        return [list(row) for row in value]


def _objects(source: ProgramAST) -> ProgramAST:
    return ProgramAST(op="objects", output_type=ValueType.OBJECT_SET, children=(source,))


def _selected(source: ProgramAST, selector: SelectorSpec) -> ProgramAST:
    objects = _objects(source)
    if selector.name == "all":
        return objects
    arguments: dict[str, Any] = {"where": dict(selector.where)}
    if selector.extremum:
        arguments["extremum"] = selector.extremum
    return ProgramAST(
        op="select_objects",
        output_type=ValueType.OBJECT_SET,
        arguments=arguments,
        children=(objects,),
    )


def _background(source: ProgramAST) -> ProgramAST:
    return ProgramAST(
        op="background_color", output_type=ValueType.COLOR, children=(source,)
    )


def _map_object(
    selected: ProgramAST,
    *,
    dy: int = 0,
    dx: int = 0,
    color: int | None = None,
) -> ProgramAST:
    item = ProgramAST(
        op="load", output_type=ValueType.OBJECT, arguments={"name": "item"}
    )
    body = item
    if color is not None:
        body = ProgramAST(
            op="recolor_object",
            output_type=ValueType.OBJECT,
            arguments={"color": color},
            children=(body,),
        )
    if dy or dx:
        body = ProgramAST(
            op="translate_object",
            output_type=ValueType.OBJECT,
            arguments={"dy": dy, "dx": dx},
            children=(body,),
        )
    return ProgramAST(
        op="map_objects",
        output_type=ValueType.OBJECT_SET,
        arguments={"item_register": "item"},
        children=(selected, body),
    )


def _selectors(task_data: Mapping[str, Any]) -> tuple[SelectorSpec, ...]:
    states = [
        perceive_grid(pair["input"], f"selector:{index}")
        for index, pair in enumerate(task_data["train"])
    ]
    selectors = [
        SelectorSpec("all", {}),
        SelectorSpec("largest", {}, extremum="largest", penalty=1),
        SelectorSpec("smallest", {}, extremum="smallest", penalty=1),
    ]
    if all(any(item.attributes.get("is_line") for item in state.objects) for state in states):
        selectors.append(SelectorSpec("lines", {"is_line": True}, penalty=2))
    if all(any(item.attributes.get("is_square") for item in state.objects) for state in states):
        selectors.append(SelectorSpec("squares", {"is_square": True}, penalty=2))

    def common_values(attribute: str) -> set[Any]:
        values = []
        for state in states:
            if attribute == "color":
                background = state.background_candidates[0]
                values.append(
                    {item.colors[0] for item in state.objects if item.colors[0] != background}
                )
            else:
                values.append({getattr(item, attribute) for item in state.objects})
        return set.intersection(*values) if values else set()

    for color in sorted(common_values("color")):
        selectors.append(SelectorSpec(f"color={color}", {"color": color}, penalty=3))
    for area in sorted(common_values("area")):
        selectors.append(SelectorSpec(f"area={area}", {"area": area}, penalty=3))
    for holes in sorted(value for value in common_values("holes") if value > 0):
        selectors.append(SelectorSpec(f"holes={holes}", {"holes": holes}, penalty=2))
    unique = {(
        selector.name,
        tuple(sorted(selector.where.items())),
        selector.extremum,
    ): selector for selector in selectors}
    return tuple(unique.values())


def _observed_deltas(
    task_data: Mapping[str, Any],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    state = perceive_task("candidate-generation", task_data)
    objects = {item.object_id: item for item in state.objects}
    translations_by_example: list[set[tuple[int, int]]] = []
    changes_by_example: list[set[tuple[int, int, int]]] = []
    for delta in state.deltas:
        translations: set[tuple[int, int]] = set()
        changes: set[tuple[int, int, int]] = set()
        for correspondence in delta.correspondences:
            dy, dx = correspondence.translation
            target = objects[correspondence.target_id]
            if dy or dx:
                translations.add((dy, dx))
            if target.colors:
                changes.add((dy, dx, target.colors[0]))
        translations_by_example.append(translations)
        changes_by_example.append(changes)
    common_translations = (
        set.intersection(*translations_by_example) if translations_by_example else set()
    )
    common_changes = set.intersection(*changes_by_example) if changes_by_example else set()
    return tuple(sorted(common_translations)), tuple(sorted(common_changes))


def _target_colors(task_data: Mapping[str, Any]) -> tuple[int, ...]:
    colors: set[int] = set()
    for index, pair in enumerate(task_data["train"]):
        state = perceive_grid(pair["output"], f"target-color:{index}")
        background = state.background_candidates[0]
        colors.update(color for color in state.palette if color != background)
    return tuple(sorted(colors))


def generate_object_programs(task_data: Mapping[str, Any]) -> tuple[tuple[str, ProgramAST, int], ...]:
    source = input_node()
    background = _background(source)
    translations, observed_changes = _observed_deltas(task_data)
    target_colors = _target_colors(task_data)
    programs: list[tuple[str, ProgramAST, int]] = []
    for selector in _selectors(task_data):
        selected = _selected(source, selector)
        first = ProgramAST(
            op="first_object", output_type=ValueType.OBJECT, children=(selected,)
        )
        programs.append(
            (
                f"crop:{selector.name}",
                ProgramAST(
                    op="crop_object",
                    output_type=ValueType.GRID,
                    children=(first, background),
                ),
                selector.penalty,
            )
        )
        blank = ProgramAST(
            op="canvas_like",
            output_type=ValueType.GRID,
            children=(source, background),
        )
        programs.append(
            (
                f"extract:{selector.name}",
                ProgramAST(
                    op="render_objects",
                    output_type=ValueType.GRID,
                    children=(blank, selected),
                ),
                selector.penalty,
            )
        )
        programs.append(
            (
                f"erase:{selector.name}",
                ProgramAST(
                    op="erase_objects",
                    output_type=ValueType.GRID,
                    children=(source, selected, background),
                ),
                selector.penalty,
            )
        )
        for color in target_colors:
            recolored = _map_object(selected, color=color)
            programs.append(
                (
                    f"recolor:{selector.name}:{color}",
                    ProgramAST(
                        op="render_objects",
                        output_type=ValueType.GRID,
                        children=(source, recolored),
                    ),
                    selector.penalty + 1,
                )
            )
        for dy, dx in translations:
            moved = _map_object(selected, dy=dy, dx=dx)
            erased = ProgramAST(
                op="erase_objects",
                output_type=ValueType.GRID,
                children=(source, selected, background),
            )
            programs.append(
                (
                    f"move:{selector.name}:{dy},{dx}",
                    ProgramAST(
                        op="render_objects",
                        output_type=ValueType.GRID,
                        children=(erased, moved),
                    ),
                    selector.penalty + 2,
                )
            )
            programs.append(
                (
                    f"copy:{selector.name}:{dy},{dx}",
                    ProgramAST(
                        op="render_objects",
                        output_type=ValueType.GRID,
                        children=(source, moved),
                    ),
                    selector.penalty + 2,
                )
            )
        for dy, dx, color in observed_changes:
            changed = _map_object(selected, dy=dy, dx=dx, color=color)
            erased = ProgramAST(
                op="erase_objects",
                output_type=ValueType.GRID,
                children=(source, selected, background),
            )
            programs.append(
                (
                    f"move-recolor:{selector.name}:{dy},{dx}:{color}",
                    ProgramAST(
                        op="render_objects",
                        output_type=ValueType.GRID,
                        children=(erased, changed),
                    ),
                    selector.penalty + 3,
                )
            )
    unique = {program.digest: (name, program, penalty) for name, program, penalty in programs}
    return tuple(unique.values())


def exact_object_candidates(
    task_data: Mapping[str, Any], *, max_programs: int = 5_000
) -> list[ObjectProgramCandidate]:
    train_pairs = tuple(
        GridPair(
            input=tuple(tuple(row) for row in pair["input"]),
            output=tuple(tuple(row) for row in pair["output"]),
        )
        for pair in task_data["train"]
    )
    test_inputs = [pair["input"] for pair in task_data.get("test", [])]
    generated = generate_object_programs(task_data)
    if len(generated) > max_programs:
        raise ExecutionError(
            f"Object program budget exceeded: {len(generated)} > {max_programs}"
        )
    executor = TypedExecutor()
    exact: list[ObjectProgramCandidate] = []
    seen_behavior: set[tuple[Grid, ...]] = set()
    for name, program, penalty in generated:
        try:
            if any(
                executor.execute(
                    program, pair.input, capture_trace=False
                ).value
                != pair.output
                for pair in train_pairs
            ):
                continue
            behavior = tuple(
                executor.execute(program, grid, capture_trace=False).value
                for grid in test_inputs
            )
        except ExecutionError:
            continue
        if behavior in seen_behavior:
            continue
        seen_behavior.add(behavior)
        exact.append(ObjectProgramCandidate(name, program, penalty))
    exact.sort(key=lambda item: (item.complexity, len(item.name), item.name))
    return exact

"""Typed execution, legacy migration, and exact replay tests."""

from __future__ import annotations

import pytest

from hyper_arc.contender.executor import (
    ExecutionError,
    TypedExecutor,
    input_node,
    legacy_name_to_ast,
    replay_program,
)
from hyper_arc.contender.schemas import (
    GridPair,
    ProgramAST,
    ValueType,
    program_from_dict,
    program_to_dict,
)
from hyper_arc.deterministic import _base_transforms


def test_every_legacy_base_transform_replays_through_typed_ast():
    grid = [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
    executor = TypedExecutor()

    for name, transform, _ in _base_transforms():
        program = legacy_name_to_ast(name)
        expected = transform(grid)
        if not expected:
            with pytest.raises(ExecutionError):
                executor.execute(program, grid)
        else:
            result = executor.execute(program, grid)
            assert [list(row) for row in result.value] == expected, name


def test_legacy_remap_requires_and_executes_explicit_binding():
    program = legacy_name_to_ast("rotate90+remap", color_map={1: 7})
    result = TypedExecutor().execute(program, [[1, 0], [1, 1]])

    assert result.value == ((7, 7), (7, 0))


def test_multi_register_program_preserves_original_input():
    source = input_node()
    store_rotated = ProgramAST(
        op="store",
        output_type=ValueType.GRID,
        arguments={"name": "rotated"},
        children=(
            ProgramAST(
                op="rotate",
                output_type=ValueType.GRID,
                arguments={"degrees": 180},
                children=(source,),
            ),
        ),
    )
    overlay = ProgramAST(
        op="overlay",
        output_type=ValueType.GRID,
        children=(
            source,
            ProgramAST(
                op="load",
                output_type=ValueType.GRID,
                arguments={"name": "rotated"},
            ),
        ),
    )
    program = ProgramAST(
        op="sequence",
        output_type=ValueType.GRID,
        children=(store_rotated, overlay),
    )

    result = TypedExecutor().execute(program, [[1, 0], [0, 2]])

    assert result.value == ((2, 0), (0, 1))
    assert any(step.op == "store" for step in result.trace.steps)
    assert any(step.op == "load" for step in result.trace.steps)
    assert next(step for step in result.trace.steps if step.op == "store").input_digests


def test_exact_replay_returns_traces_and_cell_residuals():
    executor = TypedExecutor()
    rotate = legacy_name_to_ast("rotate90")
    pairs = (
        GridPair(input=((1, 0), (1, 1)), output=((1, 1), (1, 0))),
        GridPair(input=((2, 0), (0, 0)), output=((9, 2), (0, 0))),
    )

    traces, residuals, exact = replay_program(
        executor, rotate, pairs, task_id="replay"
    )

    assert exact is False
    assert [trace.exact for trace in traces] == [True, False]
    assert residuals[0].mismatched_cells == ((0, 0),)
    assert residuals[0].expected_shape == (2, 2)


def test_executor_rejects_unknown_operation_and_unassigned_register():
    with pytest.raises(ExecutionError, match="Unknown or disallowed"):
        TypedExecutor().execute(
            ProgramAST(op="python_eval", output_type=ValueType.GRID), [[1]]
        )
    with pytest.raises(ExecutionError, match="not been assigned"):
        TypedExecutor().execute(
            ProgramAST(
                op="load",
                output_type=ValueType.GRID,
                arguments={"name": "missing"},
            ),
            [[1]],
        )


def test_object_selection_mapping_and_blank_canvas_rendering():
    source = input_node()
    objects = ProgramAST(
        op="objects", output_type=ValueType.OBJECT_SET, children=(source,)
    )
    selected = ProgramAST(
        op="select_objects",
        output_type=ValueType.OBJECT_SET,
        arguments={"where": {"color": 1}},
        children=(objects,),
    )
    item = ProgramAST(
        op="load", output_type=ValueType.OBJECT, arguments={"name": "item"}
    )
    recolored = ProgramAST(
        op="recolor_object",
        output_type=ValueType.OBJECT,
        arguments={"color": 3},
        children=(item,),
    )
    moved = ProgramAST(
        op="translate_object",
        output_type=ValueType.OBJECT,
        arguments={"dy": 1, "dx": 0},
        children=(recolored,),
    )
    mapped = ProgramAST(
        op="map_objects",
        output_type=ValueType.OBJECT_SET,
        arguments={"item_register": "item"},
        children=(selected, moved),
    )
    blank = ProgramAST(
        op="canvas_like",
        output_type=ValueType.GRID,
        arguments={"color": 0},
        children=(source,),
    )
    program = ProgramAST(
        op="render_objects",
        output_type=ValueType.GRID,
        children=(blank, mapped),
    )

    result = TypedExecutor().execute(
        program,
        [[0, 1, 1, 0], [0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 0]],
    )

    assert result.value == (
        (0, 0, 0, 0),
        (0, 3, 3, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )


def test_typed_condition_selects_only_one_branch():
    source = input_node()
    count = ProgramAST(
        op="count",
        output_type=ValueType.INTEGER,
        children=(
            ProgramAST(
                op="objects", output_type=ValueType.OBJECT_SET, children=(source,)
            ),
        ),
    )
    more_than_one = ProgramAST(
        op="compare",
        output_type=ValueType.BOOLEAN,
        arguments={"operator": "gt"},
        children=(
            count,
            ProgramAST(
                op="literal",
                output_type=ValueType.INTEGER,
                arguments={"value": 1},
            ),
        ),
    )
    program = ProgramAST(
        op="if",
        output_type=ValueType.GRID,
        children=(
            more_than_one,
            ProgramAST(
                op="rotate",
                output_type=ValueType.GRID,
                arguments={"degrees": 180},
                children=(source,),
            ),
            ProgramAST(op="python_eval", output_type=ValueType.GRID),
        ),
    )

    result = TypedExecutor().execute(program, [[1, 0], [0, 2]])

    assert result.value == ((2, 0), (0, 1))


def test_crop_largest_object_constructs_minimal_canvas():
    largest = ProgramAST(
        op="select_objects",
        output_type=ValueType.OBJECT_SET,
        arguments={"where": {}, "extremum": "largest"},
        children=(
            ProgramAST(
                op="objects",
                output_type=ValueType.OBJECT_SET,
                children=(input_node(),),
            ),
        ),
    )
    program = ProgramAST(
        op="crop_object",
        output_type=ValueType.GRID,
        children=(
            ProgramAST(
                op="first_object",
                output_type=ValueType.OBJECT,
                children=(largest,),
            ),
        ),
    )

    result = TypedExecutor().execute(program, [[4, 0, 2], [0, 0, 2], [0, 0, 2]])

    assert result.value == ((2,), (2,), (2,))


def test_explicit_precondition_and_postcondition_are_enforced():
    source = input_node()
    height = ProgramAST(
        op="grid_height", output_type=ValueType.INTEGER, children=(source,)
    )
    height_is_two = ProgramAST(
        op="equals",
        output_type=ValueType.BOOLEAN,
        children=(
            height,
            ProgramAST(
                op="literal",
                output_type=ValueType.INTEGER,
                arguments={"value": 2},
            ),
        ),
    )
    result_height_is_two = ProgramAST(
        op="equals",
        output_type=ValueType.BOOLEAN,
        children=(
            ProgramAST(
                op="grid_height",
                output_type=ValueType.INTEGER,
                children=(
                    ProgramAST(
                        op="load",
                        output_type=ValueType.GRID,
                        arguments={"name": "result"},
                    ),
                ),
            ),
            ProgramAST(
                op="literal",
                output_type=ValueType.INTEGER,
                arguments={"value": 2},
            ),
        ),
    )
    guarded = ProgramAST(
        op="identity",
        output_type=ValueType.GRID,
        children=(source,),
        preconditions=(height_is_two,),
        postconditions=(result_height_is_two,),
    )

    assert TypedExecutor().execute(guarded, [[1], [0]]).value == ((1,), (0,))
    with pytest.raises(ExecutionError, match="precondition failed"):
        TypedExecutor().execute(guarded, [[1]])

    impossible_postcondition = ProgramAST(
        op="identity",
        output_type=ValueType.GRID,
        children=(source,),
        postconditions=(
            ProgramAST(
                op="literal",
                output_type=ValueType.BOOLEAN,
                arguments={"value": False},
            ),
        ),
    )
    with pytest.raises(ExecutionError, match="postcondition failed"):
        TypedExecutor().execute(impossible_postcondition, [[1]])


def test_ast_complexity_counts_control_and_constraints():
    simple = legacy_name_to_ast("identity")
    mapped = ProgramAST(
        op="map_objects",
        output_type=ValueType.OBJECT_SET,
        children=(
            ProgramAST(
                op="objects",
                output_type=ValueType.OBJECT_SET,
                children=(input_node(),),
            ),
            ProgramAST(
                op="load",
                output_type=ValueType.OBJECT,
                arguments={"name": "item"},
            ),
        ),
    )

    assert simple.complexity == 1
    assert mapped.complexity > simple.complexity


def test_program_serialization_round_trip_remains_executable():
    source = input_node()
    height_is_two = ProgramAST(
        op="equals",
        output_type=ValueType.BOOLEAN,
        children=(
            ProgramAST(
                op="grid_height",
                output_type=ValueType.INTEGER,
                children=(source,),
            ),
            ProgramAST(
                op="literal",
                output_type=ValueType.INTEGER,
                arguments={"value": 2},
            ),
        ),
    )
    original = ProgramAST(
        op="rotate",
        output_type=ValueType.GRID,
        arguments={"degrees": 180},
        children=(source,),
        preconditions=(height_is_two,),
    )

    restored = program_from_dict(program_to_dict(original))

    assert restored.digest == original.digest
    assert restored.complexity == original.complexity
    assert TypedExecutor().execute(restored, [[1, 0], [0, 2]]).value == (
        (2, 0),
        (0, 1),
    )


def test_executor_enforces_program_and_iteration_budgets():
    nested = input_node()
    for _ in range(5):
        nested = ProgramAST(
            op="identity", output_type=ValueType.GRID, children=(nested,)
        )
    with pytest.raises(ExecutionError, match="depth budget"):
        TypedExecutor(max_depth=3).execute(nested, [[1]])

    objects = ProgramAST(
        op="objects", output_type=ValueType.OBJECT_SET, children=(input_node(),)
    )
    mapped = ProgramAST(
        op="map_objects",
        output_type=ValueType.OBJECT_SET,
        children=(
            objects,
            ProgramAST(
                op="load",
                output_type=ValueType.OBJECT,
                arguments={"name": "item"},
            ),
        ),
    )
    with pytest.raises(ExecutionError, match="item budget"):
        TypedExecutor(max_map_items=1).execute(mapped, [[1, 0, 2]])


def test_dynamic_background_can_erase_selected_objects():
    source = input_node()
    selected = ProgramAST(
        op="select_objects",
        output_type=ValueType.OBJECT_SET,
        arguments={"where": {"color": 1}},
        children=(
            ProgramAST(
                op="objects", output_type=ValueType.OBJECT_SET, children=(source,)
            ),
        ),
    )
    program = ProgramAST(
        op="erase_objects",
        output_type=ValueType.GRID,
        children=(
            source,
            selected,
            ProgramAST(
                op="background_color",
                output_type=ValueType.COLOR,
                children=(source,),
            ),
        ),
    )

    result = TypedExecutor().execute(program, [[9, 9, 9], [9, 1, 9]])

    assert result.value == ((9, 9, 9), (9, 9, 9))


def test_search_execution_skips_trace_materialization_but_keeps_budget():
    program = ProgramAST(
        op="identity", output_type=ValueType.GRID, children=(input_node(),)
    )

    result = TypedExecutor().execute(program, [[1]], capture_trace=False)

    assert result.value == ((1,),)
    assert result.trace.steps == ()
    with pytest.raises(ExecutionError, match="trace budget"):
        TypedExecutor(max_trace_steps=1).execute(
            program, [[1]], capture_trace=False
        )

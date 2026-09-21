"""Execution-guided neural code agent for ARC task-level program synthesis."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agentic_sandbox import SAFE_CALLS, SAFE_METHODS, validate_code
from hyper_arc.contender.perception import perceive_task


Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class AgenticReasoningError(RuntimeError):
    """Raised when the neural reasoning boundary cannot produce a usable result."""


@dataclass(frozen=True)
class CodeHypothesis:
    digest: str
    summary: str
    code: str
    complexity: int
    round_index: int
    test_predictions: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class AgenticResult:
    hypotheses: tuple[CodeHypothesis, ...]
    test_predictions: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
    rounds_executed: int
    candidates_generated: int
    candidates_executed: int
    failures: tuple[str, ...]
    active_trace: tuple["ReasoningTrace", ...] = ()


@dataclass(frozen=True)
class ReasoningTrace:
    """Dehydrated, task-local experience; never a claim of test correctness."""

    round_index: int
    rule: str
    evidence: str
    algorithm: str
    code_digest: str
    demo_residual: int
    causes: tuple[str, ...]


def _valid_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid or len(grid) > 30:
        return False
    if not all(isinstance(row, list) and row for row in grid):
        return False
    width = len(grid[0])
    return width <= 30 and all(
        len(row) == width
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= 9
            for value in row
        )
        for row in grid
    )


def execute_code(
    code: str,
    grids: Sequence[list[list[int]]],
    *,
    timeout_seconds: float = 2.0,
) -> list[list[list[int]]]:
    """Run validated code in a short-lived worker and validate every result."""
    validate_code(code)
    worker = Path(__file__).with_name("agentic_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps({"code": code, "grids": grids}, separators=(",", ":")),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\n", " ")[-300:]
        raise ValueError(f"worker rejected program: {detail or completed.returncode}")
    try:
        outputs = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("worker returned malformed JSON") from exc
    if not isinstance(outputs, list) or len(outputs) != len(grids):
        raise ValueError("worker returned the wrong output count")
    if not all(_valid_grid(grid) for grid in outputs):
        raise ValueError("worker returned an invalid ARC grid")
    return outputs


def _extract_json_object(text: str) -> Mapping[str, Any]:
    """Return the response object, preferring one that carries `hypotheses`.

    Models sometimes emit a small JSON fragment (e.g. a stray ``{...}`` inside
    prose, or a single hypothesis object) before the real payload. Returning the
    first object found then fails downstream as "missing hypotheses" and burns a
    whole round. Scan every decodable object and prefer the first one that has a
    `hypotheses` list, falling back to the first valid object otherwise.
    """
    decoder = json.JSONDecoder()
    first_object: Mapping[str, Any] | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        if isinstance(value.get("hypotheses"), list):
            return value
        if first_object is None:
            first_object = value
    if first_object is not None:
        return first_object
    raise AgenticReasoningError("model response contains no valid JSON object")


def _normalize_code(code: str) -> str:
    """Recover runnable source from common model formatting quirks.

    Models wrap code in markdown fences (sometimes preceded by prose) and
    occasionally emit literal ``\\n``/``\\t`` escapes instead of real
    whitespace. Prior logic only stripped a fence at position 0 and only
    unescaped when the string had no real newlines, so mixed responses stayed
    unparseable and were discarded as syntax errors. Extract the fenced block
    when present, otherwise strip a leading fence marker, then unescape literal
    line/tab escapes whenever they appear.
    """
    text = code.strip()
    # Prefer the contents of the first fenced block anywhere in the response.
    fence = text.find("```")
    if fence != -1:
        rest = text[fence + 3 :]
        if rest[:6].lower().startswith("python"):
            rest = rest[6:]
        elif rest[:2] == "py":
            rest = rest[2:]
        close = rest.find("```")
        text = (rest if close == -1 else rest[:close]).strip()
    # Some models JSON-escape newlines/tabs into the code string itself.
    if "\\n" in text or "\\t" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text.strip()


def _code_response_schema(max_candidates: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "maxItems": max_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["summary", "code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["hypotheses"],
        "additionalProperties": False,
    }


def _task_payload(task_data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train": [
            {"input": pair["input"], "output": pair["output"]}
            for pair in task_data.get("train", [])
        ],
        "test": [{"input": pair["input"]} for pair in task_data.get("test", [])],
    }


# Payload budget. On busy grids the full per-object pixel lists dominated the
# prompt (12k-18k chars) and, against a fixed output-token budget, starved the
# model so it emitted no runnable code. Bound object count and per-object pixel
# detail; the raw grids are already in the task payload, and bbox/area/shape/
# holes carry the structural summary the planner actually needs.
_MAX_OBJECTS_PER_GRID = 24
_MAX_RELATIVE_PIXELS = 24
_MAX_RELATIONS_PER_GRID = 40


# Object attribute keys that the model cannot cheaply recompute from bbox/area
# and are worth the tokens. height/width/is_line/is_square are derivable from
# bbox and were dropped to shrink the payload.
_KEPT_OBJECT_ATTRIBUTES = ("filled_bbox", "shape_symmetries")


def _object_payload(item: Any, *, include_pixels: bool) -> dict[str, Any]:
    attributes = {
        key: item.attributes[key]
        for key in _KEPT_OBJECT_ATTRIBUTES
        if key in item.attributes
    }
    payload = {
        "id": item.object_id.rsplit(":", 1)[-1],
        "color": item.colors[0],
        "bbox": list(item.bounding_box),
        "area": item.area,
        "holes": item.holes,
        "shape": item.shape_signature[:12],
        "attributes": attributes,
    }
    # Only inline the pixel mask for small objects on uncluttered grids; the
    # bbox + shape signature already localize larger ones.
    if include_pixels and len(item.relative_pixels) <= _MAX_RELATIVE_PIXELS:
        payload["relative_pixels"] = [list(pixel) for pixel in item.relative_pixels]
    return payload


def _scene_payload(task_id: str, task_data: Mapping[str, Any]) -> dict[str, Any]:
    """Compact deterministic scene graph supplied to the neural planner."""
    state = perceive_task(task_id, task_data)
    grids = []
    for grid in state.grids:
        objects = grid.objects[:_MAX_OBJECTS_PER_GRID]
        include_pixels = len(grid.objects) <= _MAX_OBJECTS_PER_GRID
        grids.append(
            {
                "ref": grid.grid_ref,
                "shape": [grid.height, grid.width],
                "palette": list(grid.palette),
                "background_candidates": list(grid.background_candidates),
                "symmetries": list(grid.symmetries),
                "periodicity": [list(item) for item in grid.periodicity],
                "separator_rows": list(grid.separator_rows),
                "separator_columns": list(grid.separator_columns),
                "object_count": len(grid.objects),
                "objects": [
                    _object_payload(item, include_pixels=include_pixels)
                    for item in objects
                ],
                "relations": [
                    {
                        "kind": item.kind,
                        "source": item.source_id.rsplit(":", 1)[-1],
                        "target": item.target_id.rsplit(":", 1)[-1],
                    }
                    for item in grid.relations[:_MAX_RELATIONS_PER_GRID]
                ],
            }
        )
    return {
        "palette": list(state.palette),
        "grids": grids,
        "demonstration_deltas": [
            {
                "example": delta.example_index,
                "input_shape": list(delta.input_shape),
                "output_shape": list(delta.output_shape),
                "added": list(delta.added_object_ids),
                "removed": list(delta.removed_object_ids),
                "correspondences": [
                    {
                        "source": item.source_id.rsplit(":", 1)[-1],
                        "target": item.target_id.rsplit(":", 1)[-1],
                        "translation": list(item.translation),
                        "color_changed": item.color_changed,
                    }
                    for item in delta.correspondences
                ],
            }
            for delta in state.deltas
        ],
    }


def _planning_schema(max_candidates: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "maxItems": max_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "rule": {"type": "string"},
                        "evidence": {"type": "string"},
                        "algorithm": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["id", "rule", "evidence", "algorithm", "code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["hypotheses"],
        "additionalProperties": False,
    }


# --- Explore / exploit policy -------------------------------------------------
#
# Grounded in the bakeoff trajectories, a candidate's demonstration residual
# (total wrong cells across training pairs; shape errors encoded as >=900/pair
# by _grid_error) falls into four states that drive the planner:
#
#   DIRECT HIT     residual == 0        -> exact; accept, stop (handled in solve)
#   REFINABLE      right geometry,      -> REFINE-FIRST: keep the rule family,
#                  0 < residual <= ~all    fix the failing detail (phase/offset/
#                  cells                   binding/ordering). This is what "got
#                                          the ball rolling" on the only solve:
#                                          a 12/24-cell miss jumped to exact in
#                                          one repair. Refinable resolves in a
#                                          single jump, not a smooth descent.
#   PLATEAU        refinable, but best  -> ABANDON: the model cannot repair it;
#                  residual flat for       force a different family (diversity).
#                  _REFINE_STALL_LIMIT     Trajectories show residuals hold flat
#                  rounds                  (12,12,12) far more than they descend.
#   BIG MISS       wrong geometry, or   -> DIVERSITY: wrong rule; explore anew.
#                  ~100% wrong cells,
#                  or no attempt yet
#
# The bias is toward exploitation of a fresh refinable near-miss (blocking one
# costs a solve; allowing it is cheap since spare candidate slots still explore),
# but strictly time-boxed so a plateau cannot burn the whole round budget.
_SHAPE_ERROR_FLOOR = 900
_NEAR_MISS_MAX_FRACTION = 0.95  # right shape but ~100% wrong == wrong rule
_REFINE_STALL_LIMIT = 2  # non-improving rounds tolerated before abandoning


def _is_near_miss(best_residual: int | None, total_output_cells: int) -> bool:
    """Right output geometry with at least a few correct cells: refinable.

    Excludes exact hits (0 is a direct hit, not a near-miss), shape/geometry
    errors, and essentially-100%-wrong outputs (a wrong rule that only overlaps
    pixels, e.g. an unsolved case that plateaued at residual 2 without ever being
    refinable).
    """
    if not best_residual or total_output_cells <= 0:
        return False  # None or 0: no refinable near-miss (0 is a direct hit)
    if best_residual >= _SHAPE_ERROR_FLOOR:
        return False  # wrong output geometry — not a phase/parameter fix
    return best_residual <= int(_NEAR_MISS_MAX_FRACTION * total_output_cells)


def _diversity_directive(
    max_candidates: int,
    prior_summaries: Sequence[str],
    failed_families: Sequence[str] = (),
    *,
    best_residual: int | None = None,
    near_miss: bool = False,
) -> str:
    """Balance exploration (distinct hypotheses) against exploitation (repair).

    When there is a near-miss (a prior candidate already close to exact), forcing
    every hypothesis into a *different* family pushes the model off the productive
    trajectory — this regressed a solve where residual grew 12 -> 18 under blanket
    diversity pressure. So: on a near-miss, prioritize refining that specific
    hypothesis and only spend spare slots on alternatives. Only when the best
    attempt is still a big miss do we apply full anti-duplication pressure to
    broaden the search.
    """
    if near_miss:
        residual_note = (
            f" (residual {best_residual} cells)" if best_residual is not None else ""
        )
        directive = (
            f"REFINE-FIRST: a previous hypothesis is close{residual_note}. Devote the "
            "first hypothesis to REPAIRING that near-miss — keep its rule family and "
            "fix the specific failing detail (phase, offset, color binding, ordering, "
            "or boundary), guided by the verifier feedback. Use any remaining slots "
            "for genuinely different alternatives, but do not abandon a nearly-correct "
            "rule just to be different."
        )
    else:
        directive = (
            f"DIVERSITY REQUIREMENT: the {max_candidates} hypotheses MUST be mutually "
            "distinct. Each must use a different rule FAMILY or a materially "
            "different mechanism from every other hypothesis in this response — not a "
            "reworded restatement and not the same code with cosmetic changes. Draw "
            "from different families such as geometric transform, color remap, object "
            "selection/move, counting/classification, panel selection, symmetry "
            "completion, scaling/tiling, or cropping. If you can only justify one rule "
            "confidently, return that one plus genuinely different alternative "
            "theories for the remaining slots rather than duplicates. Two hypotheses "
            "whose transform(grid) would return identical outputs on the "
            "demonstrations are duplicates and are forbidden."
        )
    if prior_summaries:
        directive += (
            " You have ALREADY proposed these approaches for this task; do not "
            "repeat any of them verbatim unless you are repairing a named near-miss: "
            + json.dumps(list(prior_summaries[-16:]))
        )
    if failed_families:
        directive += (
            " MEMORY — these approach families were tried on structurally similar "
            "tasks and did NOT reproduce the demonstrations exactly; do not repeat "
            "them, choose a genuinely different direction: "
            + json.dumps(list(failed_families[:12]))
        )
    return directive


def _planning_prompt(
    task: Mapping[str, Any],
    scene: Mapping[str, Any],
    memory_cues: Sequence[str],
    feedback: str,
    rejected_summaries: Sequence[str],
    max_candidates: int,
    prior_summaries: Sequence[str] = (),
    failed_families: Sequence[str] = (),
    best_residual: int | None = None,
    near_miss: bool = False,
) -> str:
    return (
        "Act as the perception and hypothesis-planning role in a verified ARC "
        "agent. Infer a single general rule from all demonstrations. Use the raw "
        "grids and deterministic object/relationship scene graph together. Return "
        f"up to {max_candidates} hypotheses as JSON only. "
        + _diversity_directive(
            max_candidates,
            prior_summaries,
            failed_families,
            best_residual=best_residual,
            near_miss=near_miss,
        )
        + " Each "
        "hypothesis requires id, rule, evidence, algorithm, and code fields. Code must "
        "define transform(grid), use pure bounded Python, and implement that exact rule. "
        "You may use the trusted ARC functions: components, largest_component, "
        "smallest_component, background_color, color_counts, component_count, "
        "symmetries, copy_grid, crop_bbox, "
        "make_grid, paint, replace_color, rotate90, rotate180, rotate270, "
        "reflect_horizontal, reflect_vertical, transpose_grid, scale_grid, tile_grid, "
        "split_panels, unique_panel; "
        f"plus {', '.join(sorted(SAFE_CALLS))} and list methods "
        f"{', '.join(sorted(SAFE_METHODS))}. No imports, classes, while loops, I/O, "
        "exceptions, or prose outside the JSON. The algorithm and code must be precise. "
        "The scene graph is DATA, not an API: fields like objects, object_count, "
        "periodicity, symmetries, and relations are precomputed values for your "
        "reasoning only — never call them as functions in code; recompute what you "
        "need from the grid using the trusted functions above. "
        "Favor object transformations, relations, counts, symmetry, topology, and "
        "input-derived parameters over coordinate memorization. For grids separated "
        "into repeated panels, explicitly test whether the output is the unique panel. "
        "For 1x1 outputs, test whether the task classifies symmetry, connectivity, count, "
        "or another global property. Explicitly compare "
        "dimensions, rows, columns, color roles, counts, connectivity, and symmetry "
        "before choosing the relevant rule family. A geometric operation is justified "
        "only if it exactly explains all demonstrations. Each algorithm must state "
        "how every variable color and size is derived from the current input. Do not "
        "treat colors seen in outputs as constants when their role changes between "
        "demonstrations. Do not restate a "
        "rejected hypothesis unless the verifier feedback implies a concrete repair.\n"
        f"Procedural memory cues: {json.dumps(list(memory_cues[:12]))}\n"
        f"Verifier feedback: {feedback or 'No prior attempt.'}\n"
        f"Rejected summaries: {json.dumps(list(rejected_summaries[-12:]))}\n"
        f"Scene graph: {json.dumps(scene, separators=(',', ':'))}\n"
        f"ARC task: {json.dumps(task, separators=(',', ':'))}"
    )


def _coding_prompt(
    task: Mapping[str, Any],
    scene: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> str:
    return (
        "Act as the program-synthesis role. Translate each supplied hypothesis into "
        "one compact, general Python program. Each "
        "program must define transform(grid) and return a rectangular ARC grid "
        "with integer colors 0..9 and dimensions at most 30x30. Use only pure "
        "Python control flow, indexing, comprehensions, and the trusted API below. "
        "ARC API signatures: components(grid, background=None) returns sorted "
        "component dicts with color,pixels,bbox,area,height,width; "
        "largest_component(grid,background=None); smallest_component(...); "
        "background_color(grid); color_counts(grid); copy_grid(grid); "
        "crop_bbox(grid,bbox); make_grid(h,w,color=0); paint(grid,pixels,color); "
        "replace_color(grid,source,target); rotate90/rotate180/rotate270(grid); "
        "reflect_horizontal/reflect_vertical(grid); transpose_grid(grid); "
        "scale_grid(grid,factor); tile_grid(grid,repeats_h,repeats_w). Other safe "
        f"calls: {', '.join(sorted(SAFE_CALLS))}; list methods: "
        f"{', '.join(sorted(SAFE_METHODS))}. No imports, other attributes, classes, "
        "exceptions, I/O, global mutation, or prose outside JSON. "
        "Return {\"hypotheses\":[{\"summary\":str,\"code\":str},...]}. "
        "Every candidate must correspond to a supplied plan and must be mentally "
        "executed against every demonstration before emission. Never perform arithmetic "
        "on color numbers: colors are categorical labels. Never hard-code a color that "
        "varies between demonstrations; bind it from grid positions, objects, or counts. "
        "For example, an alternating pattern using two input-derived colors should "
        "select those colors from the input and choose between them by coordinate parity, "
        "not add to a color value. Do not emit identity "
        "unless a plan explicitly justifies identity. JSON string newlines in code "
        "must be escaped.\n"
        f"Candidate plans: {json.dumps(list(plans), separators=(',', ':'))}\n"
        f"Scene graph: {json.dumps(scene, separators=(',', ':'))}\n"
        f"ARC task: {json.dumps(task, separators=(',', ':'))}"
    )


# Typed residual causes (from the symbolic world model's diagnose_residual)
# mapped to concrete, actionable repair guidance for the LLM. This turns the
# feedback from "you were N cells off" into "the OUTPUT SIZE is wrong" /
# "only the COLOR MAPPING is wrong" so the model repairs the actual defect.
_CAUSE_HINTS: dict[str, str] = {
    "wrong_canvas": (
        "OUTPUT GEOMETRY is wrong: the predicted grid is the wrong size/shape. "
        "Fix how the output dimensions are derived from the input before anything else."
    ),
    "wrong_color_binding": (
        "STRUCTURE is right but the COLOR MAPPING is wrong: the shapes/positions "
        "match, only colors differ. Rebind output colors from the input; do not "
        "change the geometry."
    ),
    "missing_content": (
        "Content is MISSING: expected objects/cells were not produced. The rule "
        "under-generates — add the missing objects or fill step."
    ),
    "extra_content": (
        "EXTRA content was produced: the rule over-generates. Remove or gate the "
        "spurious objects/cells."
    ),
    "local_or_structural": (
        "A LOCAL/STRUCTURAL detail is off (placement, ordering, phase, or a few "
        "cells). Keep the rule family and fix the specific detail."
    ),
}


def _diagnose_causes(
    predictions: Sequence[Sequence[Sequence[int]]],
    targets: Sequence[Sequence[Sequence[int]]],
) -> list[str]:
    """Per-demo typed failure causes via the shared symbolic diagnosis."""
    from hyper_arc.contender.repair import diagnose_residual

    causes: list[str] = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        try:
            causes.append(
                diagnose_residual(prediction, target, example_index=index).cause
            )
        except Exception:  # noqa: BLE001 - diagnosis is advisory, never fatal
            causes.append("local_or_structural")
    return causes


def _feedback(
    evaluated: Sequence[tuple[str, int, list[str], list[str]]],
) -> str:
    if not evaluated:
        return "No candidate could execute. Simplify the code and obey the contract."
    best = sorted(evaluated, key=lambda item: (item[1], len(item[0])))[0]
    code = best[0]
    causes = best[3] if len(best) > 3 else []
    # Surface the dominant typed cause as a specific repair directive.
    directive = ""
    non_exact = [c for c in causes if c and c != "exact"]
    if non_exact:
        dominant = Counter(non_exact).most_common(1)[0][0]
        hint = _CAUSE_HINTS.get(dominant)
        if hint:
            directive = f" DIAGNOSIS ({dominant}): {hint}"
    return (
        "None of the candidates exactly reproduced every demonstration. Repair or "
        "replace them using this verifier feedback. Do not repeat unchanged code. "
        f"Best total cell/shape error: {best[1]}; per-demo: {best[2]}."
        f"{directive} "
        f"Best failing code:\n{code[:2400]}\n"
        "Then propose a specifically repaired hypothesis (or a different family if "
        "the diagnosis implies the whole approach is wrong)."
    )


def _grid_error(actual: list[list[int]], expected: list[list[int]]) -> tuple[int, str]:
    if len(actual) != len(expected) or len(actual[0]) != len(expected[0]):
        delta = abs(len(actual) - len(expected)) + abs(len(actual[0]) - len(expected[0]))
        return 900 + delta, f"shape {len(actual)}x{len(actual[0])} expected {len(expected)}x{len(expected[0])}"
    coordinates = [
        (row, col, actual[row][col], expected[row][col])
        for row in range(len(expected))
        for col in range(len(expected[0]))
        if actual[row][col] != expected[row][col]
    ]
    sample = ",".join(
        f"({row},{col}):{value}->{target}"
        for row, col, value, target in coordinates[:24]
    )
    return len(coordinates), f"{len(coordinates)} mismatches [{sample}]"


class AgenticReasoner:
    """Generate, execute, critique, and refine task-specific transformation code."""

    def __init__(
        self,
        transport: Transport,
        *,
        model: str,
        rounds: int = 3,
        candidates_per_round: int = 4,
        max_output_tokens: int = 2_400,
        execution_timeout: float = 2.0,
        thinking: bool = False,
    ) -> None:
        if not 1 <= rounds <= 8:
            raise ValueError("rounds must be between 1 and 8")
        if not 1 <= candidates_per_round <= 8:
            raise ValueError("candidates_per_round must be between 1 and 8")
        self.transport = transport
        self.model = model
        self.rounds = rounds
        self.candidates_per_round = candidates_per_round
        self.max_output_tokens = max_output_tokens
        self.execution_timeout = execution_timeout
        self.thinking = thinking
        self.invocations = 0
        self.exact_invocations = 0
        self.generated_total = 0
        self.executed_total = 0
        self.unusable_invocations = 0

    def solve(
        self,
        task_id: str,
        task_data: Mapping[str, Any],
        *,
        memory_cues: Sequence[str] = (),
        failure_cues: Sequence[str] = (),
        deadline: float | None = None,
    ) -> AgenticResult:
        def remaining():
            return max(0.0, deadline - time.monotonic()) if deadline is not None else float("inf")

        train_inputs = [pair["input"] for pair in task_data.get("train", [])]
        train_outputs = [pair["output"] for pair in task_data.get("train", [])]
        test_inputs = [pair["input"] for pair in task_data.get("test", [])]
        if not train_inputs or not test_inputs:
            raise ValueError("agentic reasoning requires train and test inputs")
        task = _task_payload(task_data)
        scene = _scene_payload(task_id, task_data)
        total_output_cells = sum(
            len(grid) * len(grid[0]) for grid in train_outputs if grid and grid[0]
        )
        exact: list[CodeHypothesis] = []
        failures: list[str] = []
        seen: set[str] = set()
        rejected_summaries: list[str] = []
        feedback = ""
        best_residual: int | None = None
        # Refinement only earns rounds while it is making progress. The bakeoff
        # data shows residuals plateau (e.g. 12,12,12) far more often than they
        # descend; a refinable rule usually jumps straight to exact in one repair.
        # So we refine a near-miss, but if the best residual stops improving for
        # _REFINE_STALL_LIMIT consecutive rounds we abandon it and force a new
        # family instead of nudging a dead plateau.
        refine_stall = 0
        generated = 0
        executed = 0
        rounds_executed = 0
        active_trace: list[ReasoningTrace] = []
        for round_index in range(self.rounds):
            if remaining() <= 0:
                failures.append("deadline:reasoning budget exhausted")
                break
            rounds_executed += 1
            # Induce independent rules, then test each across every training
            # pair. Keep multiple initial proposals for the two-attempt slot.
            round_candidates = self.candidates_per_round
            common_options = {
                "temperature": min(0.2 + 0.12 * round_index, 0.7),
                "seed": int(
                    hashlib.sha256(f"{task_id}:{round_index}".encode()).hexdigest()[:8],
                    16,
                ),
                "num_predict": self.max_output_tokens,
            }
            if deadline is not None:
                common_options["max_time"] = remaining()
            try:
                planning_response = self.transport(
                    {
                    "model": self.model,
                    "stream": False,
                    "format": _planning_schema(round_candidates),
                    "think": self.thinking,
                    "options": common_options,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the planning role in an execution-guided "
                                "ARC agent. The exact verifier is authoritative."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _planning_prompt(
                                task,
                                scene,
                                memory_cues,
                                feedback,
                                rejected_summaries,
                                round_candidates,
                                prior_summaries=rejected_summaries,
                                failed_families=failure_cues,
                                best_residual=best_residual,
                                near_miss=(
                                    _is_near_miss(best_residual, total_output_cells)
                                    and refine_stall < _REFINE_STALL_LIMIT
                                ),
                            ),
                        },
                    ],
                    }
                )
                message = planning_response.get("message")
                content = (
                    message.get("content") if isinstance(message, Mapping) else None
                )
                if not isinstance(content, str):
                    raise AgenticReasoningError(
                        "model response is missing message.content"
                    )
                decoded = _extract_json_object(content)
                plans = decoded.get("hypotheses")
                if not isinstance(plans, list):
                    raise AgenticReasoningError("model JSON is missing hypotheses")
            except (AgenticReasoningError, RuntimeError, TimeoutError, ValueError) as exc:
                failures.append(
                    f"r{round_index}:model:{type(exc).__name__}:{str(exc)[:300]}"
                )
                feedback = (
                    "The prior response was unusable. Return one complete JSON object "
                    "that obeys the hypothesis schema and includes executable code."
                )
                continue
            hypotheses = []
            for plan in plans:
                if not isinstance(plan, Mapping):
                    continue
                hypotheses.append(
                    {
                        "summary": str(plan.get("rule", plan.get("id", "hypothesis"))),
                        "evidence": str(plan.get("evidence", "")),
                        "algorithm": str(plan.get("algorithm", "")),
                        "code": plan.get("code"),
                    }
                )
            evaluated: list[tuple[str, int, list[str], list[str]]] = []
            for index, payload in enumerate(hypotheses[:round_candidates]):
                if remaining() <= 0:
                    failures.append("deadline:candidate budget exhausted")
                    break
                generated += 1
                if not isinstance(payload, Mapping):
                    failures.append(f"r{round_index}c{index}:not-object")
                    continue
                summary = payload.get("summary")
                code = payload.get("code")
                if not isinstance(summary, str) or not isinstance(code, str):
                    failures.append(f"r{round_index}c{index}:bad-fields")
                    continue
                code = _normalize_code(code)
                rejected_summaries.append(summary[:300])
                digest = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
                if digest in seen:
                    failures.append(f"r{round_index}c{index}:duplicate")
                    continue
                seen.add(digest)
                try:
                    complexity = validate_code(code)
                    demo_predictions = execute_code(
                        code, train_inputs, timeout_seconds=min(self.execution_timeout, remaining())
                    )
                    executed += 1
                    errors = [
                        _grid_error(actual, target)
                        for actual, target in zip(demo_predictions, train_outputs)
                    ]
                    total_error = sum(item[0] for item in errors)
                    causes = (
                        _diagnose_causes(demo_predictions, train_outputs)
                        if total_error
                        else []
                    )
                    evaluated.append(
                        (code, total_error, [item[1] for item in errors], causes)
                    )
                    active_trace.append(
                        ReasoningTrace(
                            round_index=round_index,
                            rule=summary[:300],
                            evidence=str(payload["evidence"])[:300],
                            algorithm=str(payload["algorithm"])[:300],
                            code_digest=digest,
                            demo_residual=total_error,
                            causes=tuple(causes),
                        )
                    )
                    if total_error:
                        failures.append(
                            f"r{round_index}c{index}:non-exact:{total_error}:"
                            f"{summary[:100]}"
                        )
                        continue
                    test_predictions = execute_code(
                        code, test_inputs, timeout_seconds=min(self.execution_timeout, remaining())
                    )
                    exact.append(
                        CodeHypothesis(
                            digest=digest,
                            summary=summary[:500],
                            code=code,
                            complexity=complexity,
                            round_index=round_index,
                            test_predictions=tuple(
                                tuple(tuple(row) for row in grid)
                                for grid in test_predictions
                            ),
                        )
                    )
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    failures.append(
                        f"r{round_index}c{index}:{type(exc).__name__}:{str(exc)[:300]}"
                    )
            if exact:
                break
            feedback = _feedback(evaluated)
            # Track the best (lowest) demonstration residual seen so far so the
            # next round can switch from diversity to refine-first on a near-miss,
            # and count consecutive rounds without improvement so a stalled
            # refinement is abandoned rather than nudged indefinitely.
            round_best = min((item[1] for item in evaluated), default=None)
            if round_best is not None and (
                best_residual is None or round_best < best_residual
            ):
                best_residual = round_best
                refine_stall = 0
            elif _is_near_miss(best_residual, total_output_cells):
                # We were refining a near-miss but this round did not improve it.
                refine_stall += 1
        exact.sort(key=lambda item: (item.complexity, item.round_index, item.digest))
        # Distinct source strings can implement identical behavior. Do not spend
        # both output slots on the same predicted test grids.
        behaviors = set()
        distinct = []
        for hypothesis in exact:
            if hypothesis.test_predictions not in behaviors:
                behaviors.add(hypothesis.test_predictions)
                distinct.append(hypothesis)
        exact = distinct
        predictions = tuple(
            tuple(hypothesis.test_predictions[test_index] for hypothesis in exact[:2])
            for test_index in range(len(test_inputs))
        )
        result = AgenticResult(
            hypotheses=tuple(exact),
            test_predictions=predictions,
            rounds_executed=rounds_executed,
            candidates_generated=generated,
            candidates_executed=executed,
            failures=tuple(failures),
            active_trace=tuple(active_trace),
        )
        self.invocations += 1
        self.exact_invocations += int(bool(result.hypotheses))
        self.generated_total += result.candidates_generated
        self.executed_total += result.candidates_executed
        self.unusable_invocations += int(result.candidates_executed == 0)
        return result

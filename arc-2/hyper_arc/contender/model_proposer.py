"""Closed-schema model proposal boundary for the typed ARC executor."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from hyper_arc.contender.executor import ExecutionError, TypedExecutor, replay_program
from hyper_arc.contender.perception import canonical_grid
from hyper_arc.contender.relational_plans import generate_relational_programs
from hyper_arc.contender.schemas import (
    GridPair,
    Hypothesis,
    ProgramAST,
    program_from_dict,
    stable_digest,
)


class ProposalError(RuntimeError):
    """Raised when the model endpoint or response contract fails."""


@dataclass(frozen=True)
class ProposalBatch:
    model: str
    requested: int
    parsed: int
    exact: tuple[Hypothesis, ...]
    rejected: tuple[str, ...]


Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


GRAMMAR = """Return only JSON: {"programs":[ProgramAST,...]}.
Each ProgramAST has op, output_type="grid", optional arguments, and children.
Leaf: {"op":"input","output_type":"grid"}.
Safe grid wrappers (one grid child): identity; rotate {degrees:90|180|270};
reflect {axis:"horizontal"|"vertical"}; transpose; anti_transpose;
crop_mode; crop_to_bbox {background:0..9}; scale_integer {factor:2|3|4};
tile {repeats_h:1..4,repeats_w:1..4}; color_remap
{mapping:{source_color:target_color}}. Compose by nesting children. Do not emit
explanations, code, unknown fields, or guesses outside this grammar."""

RELATIONAL_PLAN_DESCRIPTIONS = {
    "edge-marker-motif-propagation": (
        "edge line markers direct copies of an existing aligned motif toward an edge"
    ),
    "keyed-object-frames": (
        "isolated two-color keys map object colors to exterior frame colors"
    ),
    "frame-signal-repeat": (
        "normalize the signal inside a rounded frame and repeat it at learned anchors"
    ),
}


def build_proposal_prompt(task_data: Mapping[str, Any], max_candidates: int) -> str:
    """Expose demonstrations only; test outputs and evaluator labels never enter."""
    demonstrations = [
        {"input": pair["input"], "output": pair["output"]}
        for pair in task_data.get("train", [])
    ]
    return (
        "Infer the single transformation that exactly reproduces every ARC "
        "demonstration. Propose up to "
        f"{max_candidates} distinct programs, simplest first.\n\n{GRAMMAR}\n\n"
        "Demonstrations:\n" + json.dumps(demonstrations, separators=(",", ":"))
    )


def build_relational_proposal_prompt(
    task_data: Mapping[str, Any], max_candidates: int
) -> str:
    """Ask only for closed relational roles; the system binds all parameters."""
    demonstrations = [
        {"input": pair["input"], "output": pair["output"]}
        for pair in task_data.get("train", [])
    ]
    descriptions = "\n".join(
        f"- {kind}: {description}"
        for kind, description in RELATIONAL_PLAN_DESCRIPTIONS.items()
    )
    return (
        "Select up to "
        f"{max_candidates} relational plan kinds that could explain every ARC "
        "demonstration. Return only the required JSON. The runtime will infer "
        "parameters and reject any plan that does not replay exactly.\n\n"
        f"Allowed plan kinds:\n{descriptions}\n\nDemonstrations:\n"
        + json.dumps(demonstrations, separators=(",", ":"))
    )


def _ollama_transport(model: str, endpoint: str, timeout_seconds: float) -> Transport:
    def send(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            raise ProposalError(f"Model endpoint failed: {exc}") from exc
        if not isinstance(result, Mapping):
            raise ProposalError("Model endpoint returned a non-object response")
        return result

    return send


def _response_content(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ProposalError("Model response is missing message.content")
    return message["content"]


def _response_schema(max_candidates: int) -> dict[str, Any]:
    empty_arguments = {"type": "object", "maxProperties": 0}
    child = {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {"$ref": "#/$defs/node"},
    }

    def unary(op: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        properties = {
            "op": {"type": "string", "const": op},
            "output_type": {"type": "string", "const": "grid"},
            "arguments": arguments or empty_arguments,
            "children": child,
        }
        required = ["op", "output_type", "children"]
        if arguments is not None:
            required.append("arguments")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def arguments(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    input_node_schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "const": "input"},
            "output_type": {"type": "string", "const": "grid"},
            "arguments": empty_arguments,
            "children": {"type": "array", "maxItems": 0},
        },
        "required": ["op", "output_type"],
        "additionalProperties": False,
    }
    no_argument_ops = (
        "identity",
        "transpose",
        "anti_transpose",
        "crop_mode",
    )
    branches = [input_node_schema, *(unary(op) for op in no_argument_ops)]
    branches.extend(
        (
            unary(
                "rotate",
                arguments(
                    {"degrees": {"type": "integer", "enum": [90, 180, 270]}},
                    ["degrees"],
                ),
            ),
            unary(
                "reflect",
                arguments(
                    {"axis": {"type": "string", "enum": ["horizontal", "vertical"]}},
                    ["axis"],
                ),
            ),
            unary(
                "crop_to_bbox",
                arguments(
                    {"background": {"type": "integer", "minimum": 0, "maximum": 9}},
                    ["background"],
                ),
            ),
            unary(
                "scale_integer",
                arguments(
                    {"factor": {"type": "integer", "enum": [2, 3, 4]}}, ["factor"]
                ),
            ),
            unary(
                "tile",
                arguments(
                    {
                        "repeats_h": {"type": "integer", "minimum": 1, "maximum": 4},
                        "repeats_w": {"type": "integer", "minimum": 1, "maximum": 4},
                    },
                    ["repeats_h", "repeats_w"],
                ),
            ),
            unary(
                "color_remap",
                arguments(
                    {
                        "mapping": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 9,
                            },
                        }
                    },
                    ["mapping"],
                ),
            ),
        )
    )
    return {
        "type": "object",
        "properties": {
            "programs": {
                "type": "array",
                "maxItems": max_candidates,
                "items": {"$ref": "#/$defs/node"},
            }
        },
        "required": ["programs"],
        "additionalProperties": False,
        "$defs": {"node": {"oneOf": branches}},
    }


def _relational_response_schema(max_candidates: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "plans": {
                "type": "array",
                "maxItems": max_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": list(RELATIONAL_PLAN_DESCRIPTIONS),
                        }
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["plans"],
        "additionalProperties": False,
    }


def propose_exact_hypotheses(
    task_id: str,
    task_data: Mapping[str, Any],
    *,
    model: str,
    max_candidates: int = 4,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_seconds: float = 120.0,
    max_output_tokens: int = 2_400,
    transport: Transport | None = None,
) -> ProposalBatch:
    """Ask for typed programs and retain only exact demonstration replays."""
    if not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be between 1 and 8")
    if not 128 <= max_output_tokens <= 4_096:
        raise ValueError("max_output_tokens must be between 128 and 4096")
    pairs = tuple(
        GridPair(
            input=canonical_grid(pair["input"]),
            output=canonical_grid(pair["output"]),
        )
        for pair in task_data.get("train", [])
    )
    if not pairs:
        raise ValueError("Model proposal requires at least one training pair")
    request_payload = {
        "model": model,
        "stream": False,
        "format": _response_schema(max_candidates),
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": max_output_tokens,
        },
        "messages": [
            {
                "role": "system",
                "content": "You emit safe typed transformation ASTs for ARC puzzles.",
            },
            {
                "role": "user",
                "content": build_proposal_prompt(task_data, max_candidates),
            },
        ],
    }
    response = (transport or _ollama_transport(model, endpoint, timeout_seconds))(
        request_payload
    )
    content = _response_content(response)
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        done_reason = response.get("done_reason", "unknown")
        raise ProposalError(
            "Model content is not valid JSON: "
            f"{exc}; length={len(content)}; done_reason={done_reason}"
        ) from exc
    programs = decoded.get("programs") if isinstance(decoded, Mapping) else None
    if not isinstance(programs, list):
        raise ProposalError("Model JSON must contain a programs list")
    executor = TypedExecutor()
    exact: list[Hypothesis] = []
    rejected: list[str] = []
    seen: set[str] = set()
    parsed = 0
    for index, payload in enumerate(programs[:max_candidates]):
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("program is not an object")
            program: ProgramAST = program_from_dict(payload)
            parsed += 1
            if program.digest in seen:
                rejected.append(f"{index}:duplicate")
                continue
            seen.add(program.digest)
            traces, residuals, fits = replay_program(
                executor, program, pairs, task_id=task_id
            )
            if not fits:
                rejected.append(f"{index}:non-exact")
                continue
            hypothesis_id = stable_digest((task_id, model, program.digest))
            exact.append(
                Hypothesis(
                    hypothesis_id=hypothesis_id,
                    program=program,
                    traces=traces,
                    residuals=residuals,
                    exact_replay=True,
                    complexity=program.complexity,
                    confidence=1.0,
                    channel=f"model:{model}",
                    provenance={"provider": "ollama", "proposal_index": index},
                )
            )
        except (ExecutionError, TypeError, ValueError) as exc:
            detail = str(exc).replace("\n", " ")[:160]
            rejected.append(f"{index}:invalid:{type(exc).__name__}:{detail}")
    exact.sort(key=lambda item: (item.complexity, item.hypothesis_id))
    return ProposalBatch(
        model=model,
        requested=min(len(programs), max_candidates),
        parsed=parsed,
        exact=tuple(exact),
        rejected=tuple(rejected),
    )


def propose_exact_relational_hypotheses(
    task_id: str,
    task_data: Mapping[str, Any],
    *,
    model: str,
    max_candidates: int = 3,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_seconds: float = 120.0,
    max_output_tokens: int = 512,
    transport: Transport | None = None,
) -> ProposalBatch:
    """Let a model select closed plan kinds; bind and verify them deterministically."""
    if not 1 <= max_candidates <= len(RELATIONAL_PLAN_DESCRIPTIONS):
        raise ValueError("max_candidates exceeds the closed relational plan set")
    pairs = tuple(
        GridPair(
            input=canonical_grid(pair["input"]),
            output=canonical_grid(pair["output"]),
        )
        for pair in task_data.get("train", [])
    )
    if not pairs:
        raise ValueError("Relational proposal requires at least one training pair")
    request_payload = {
        "model": model,
        "stream": False,
        "format": _relational_response_schema(max_candidates),
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": max_output_tokens,
        },
        "messages": [
            {
                "role": "system",
                "content": "You select closed relational plans; the runtime is authoritative.",
            },
            {
                "role": "user",
                "content": build_relational_proposal_prompt(task_data, max_candidates),
            },
        ],
    }
    response = (transport or _ollama_transport(model, endpoint, timeout_seconds))(
        request_payload
    )
    content = _response_content(response)
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProposalError(
            f"Model content is not valid relational JSON: {exc}"
        ) from exc
    plans = decoded.get("plans") if isinstance(decoded, Mapping) else None
    if not isinstance(plans, list):
        raise ProposalError("Model JSON must contain a plans list")
    available = dict(generate_relational_programs(task_data))
    executor = TypedExecutor()
    exact: list[Hypothesis] = []
    rejected: list[str] = []
    seen: set[str] = set()
    parsed = 0
    for index, payload in enumerate(plans[:max_candidates]):
        if not isinstance(payload, Mapping) or set(payload) != {"kind"}:
            rejected.append(f"{index}:invalid:closed-plan-object-required")
            continue
        kind = payload.get("kind")
        if not isinstance(kind, str) or kind not in RELATIONAL_PLAN_DESCRIPTIONS:
            rejected.append(f"{index}:invalid:unknown-plan-kind")
            continue
        parsed += 1
        if kind in seen:
            rejected.append(f"{index}:duplicate")
            continue
        seen.add(kind)
        program = available.get(kind)
        if program is None:
            rejected.append(f"{index}:unbound")
            continue
        try:
            traces, residuals, fits = replay_program(
                executor, program, pairs, task_id=task_id
            )
        except ExecutionError as exc:
            rejected.append(f"{index}:invalid:{str(exc)[:120]}")
            continue
        if not fits:
            rejected.append(f"{index}:non-exact")
            continue
        exact.append(
            Hypothesis(
                hypothesis_id=stable_digest(
                    (task_id, model, "relational-v2", program.digest)
                ),
                program=program,
                traces=traces,
                residuals=residuals,
                exact_replay=True,
                complexity=program.complexity,
                confidence=1.0,
                channel=f"model:{model}:relational-v2",
                provenance={
                    "provider": "ollama",
                    "proposal_index": index,
                    "plan_kind": kind,
                    "parameters_bound_by": "deterministic-demonstration-analysis",
                },
            )
        )
    exact.sort(key=lambda item: (item.complexity, item.hypothesis_id))
    return ProposalBatch(
        model=model,
        requested=min(len(plans), max_candidates),
        parsed=parsed,
        exact=tuple(exact),
        rejected=tuple(rejected),
    )

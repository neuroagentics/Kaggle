"""Transformers 4-bit deliberator backend — R11 implementation.

Concrete DeliberatorInterface backed by a local HuggingFace causal-LM checkpoint
(Qwen2.5-7B-Instruct, Apache-2.0) loaded 4-bit via bitsandbytes NF4 on CUDA.

This is the architecture-native backend the blueprint specified: a single
PyTorch/CUDA stack shared with the simulator, one quantizer (bitsandbytes int4),
no second engine. It implements only `_call_model` and `_repair`; the base
DeliberatorInterface owns token budgets, parsing, the one-repair rule, and the
fallback-to-last-valid contract.

Loading contract:
  - 4-bit load REQUIRES CUDA (bitsandbytes has no CPU int4 path). On CPU-only
    machines this backend cannot load; that is expected and reported clearly.
    Local unit tests exercise the prompt/parse plumbing with a fake generator
    (see tests/test_deliberator_transformers.py) so no GPU is needed for CI.
  - The checkpoint directory must contain HF safetensors + tokenizer + config.

The model PROPOSES structured hypotheses; it never sends actions, never writes
executable code, and never certifies truth (enforced by the typed schema and the
controller, not by trusting the model).
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from agent.deliberator import (
    DeliberatorConfig,
    DeliberatorInput,
    DeliberatorInterface,
)
from agent.model import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a game-mechanics analyst for an interactive grid puzzle. "
    "You observe recent grid states (colors are integers 0 to 15), the actions "
    "taken, and outcomes. You propose FALSIFIABLE hypotheses about the game's "
    "mechanics and goals. You do NOT choose actions. Respond with STRICT JSON "
    "ONLY, no prose. All numbers must be DECIMAL integers (never hex, never "
    "letters). Keep affected_colors to the specific colors involved, not all of "
    "them. Match exactly this schema:\n"
    "{\n"
    '  "mechanic_hypotheses": [\n'
    "    {\n"
    '      "mechanic_type": "movement|interaction|transformation|constraint|goal_condition|spawn|environmental|unknown",\n'
    '      "description": "short human-readable description",\n'
    '      "object_type": "optional string or null",\n'
    '      "direction_vector": [dx, dy] with dx,dy in {-1,0,1}, or null,\n'
    '      "affected_colors": [ints 0-15] or null,\n'
    '      "precondition_tags": ["tag", ...],\n'
    '      "effect_tags": ["tag", ...],\n'
    '      "confidence": 0.0-1.0\n'
    "    }\n"
    "  ],\n"
    '  "goal_signals": [\n'
    "    {\n"
    '      "description": "short description",\n'
    '      "target_color": int 0-15 or null,\n'
    '      "target_region": [x0,y0,x1,y1] within 0-63 or null,\n'
    '      "priority": 0.0-1.0 (exclusive of 0),\n'
    '      "status": "provisional|active"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Keep at most 3 hypotheses and 2 goals. Output JSON only."
)


def _grid_digest(grid: list[list[int]], max_rows: int = 16, max_cols: int = 16) -> str:
    """Compact textual rendering of a grid (bounded for prompt size)."""
    if not grid:
        return "(empty)"
    rows = grid[:max_rows]
    return "\n".join(
        "".join(f"{c:x}" for c in row[:max_cols]) for row in rows
    )


def build_prompt(inp: DeliberatorInput) -> str:
    """Build a user prompt from structured observations (no chat history)."""
    lines: list[str] = [f"game_id={inp.game_id} level={inp.level_id}"]

    if inp.current_grid:
        lines.append("\nCurrent observation (unscaled coordinates):")
        lines.append(_grid_digest(inp.current_grid, max_rows=64, max_cols=64))
    if inp.recent_grids:
        lines.append(f"\nMost recent {len(inp.recent_grids)} grid state(s) "
                     f"(hex colors, cropped):")
        for i, grid in enumerate(inp.recent_grids):
            lines.append(f"grid[{i}]:")
            lines.append(_grid_digest(grid))

    if inp.recent_actions:
        acts = ", ".join(
            f"(id={a[0]},x={a[1]},y={a[2]})" for a in inp.recent_actions
        )
        lines.append(f"\nRecent actions: {acts}")
    if inp.recent_outcomes:
        lines.append("Recent outcomes (success?): "
                     + ", ".join("Y" if o else "N" for o in inp.recent_outcomes))
    if inp.prediction_errors:
        lines.append("Recent prediction errors: "
                     + ", ".join("unknown" if e is None else f"{e:.2f}" for e in inp.prediction_errors))
    if inp.active_belief_summaries:
        lines.append("\nCurrent beliefs:\n- " + "\n- ".join(inp.active_belief_summaries))
    if inp.active_goal_summaries:
        lines.append("Current goals:\n- " + "\n- ".join(inp.active_goal_summaries))

    lines.append("\nPropose hypotheses and goals as STRICT JSON now.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction + ID injection
# ---------------------------------------------------------------------------

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _new_hex_id() -> str:
    return uuid.uuid4().hex


# Bare hex letters (a-f) appearing as array elements, e.g. "[0, 1, a, b]".
# LLMs prompted on hex grids sometimes emit these; JSON requires decimal.
_BARE_HEX_ELEM = re.compile(r"(?<=[\[,\s])([a-f])(?=[\s,\]])")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _repair_json_text(text: str) -> str:
    """Fix common LLM-JSON defects so a second parse can succeed.

    - Convert bare hex letters (a-f) used as array elements to their decimal
      value (a->10 ... f->15). Only single letters bounded by array delimiters
      are touched, so words in strings are not corrupted.
    - Remove trailing commas before } or ].
    """
    def _hex_to_dec(mobj: re.Match) -> str:
        return str(int(mobj.group(1), 16))

    fixed = _BARE_HEX_ELEM.sub(_hex_to_dec, text)
    fixed = _TRAILING_COMMA.sub(r"\1", fixed)
    fixed = _close_truncated_json(fixed)
    return fixed


def _close_truncated_json(text: str) -> str:
    """Best-effort completion of a JSON object truncated mid-generation.

    Token-limited model output often ends partway through an array/object. We
    drop any trailing incomplete token after the last complete element and close
    the open brackets so the salvageable prefix parses. String-aware so braces
    inside strings are ignored.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    last_safe = -1  # index just after the last complete top-level-ish element
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack:
                stack.pop()
        elif ch == "," and len(stack) <= 2:
            last_safe = i  # a comma at shallow depth ends a complete element
    if not stack:
        return text
    # Cut back to the last complete element boundary, drop the dangling comma.
    if last_safe > 0:
        head = text[:last_safe]
    else:
        head = text.rstrip().rstrip(",")
    # Recompute open brackets for the trimmed head.
    stack = []
    in_str = False
    escape = False
    for ch in head:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack:
                stack.pop()
    return head + "".join(reversed(stack))


def _clamp_colors(value):
    """Coerce an affected_colors value to a list of valid 0..15 ints, or None."""
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out = []
    for v in value:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= iv <= 15 and iv not in out:
            out.append(iv)
    return out or None


def normalize_model_json(raw_text: str) -> str:
    """Extract the first JSON object and inject required IDs / schema_version.

    The prompt does not ask the model to emit the 32-hex IDs (it would get them
    wrong); we assign them here so validate_*_dict passes. belief_id is a fresh
    ID per hypothesis (the controller links it to a real MechanicBelief when it
    records evidence). This keeps the typed schema intact without trusting the
    model to produce valid UUIDs.

    Returns a JSON string ready for DeliberatorInterface._parse_output. If no
    JSON object can be found, returns the raw text unchanged (parse will fail,
    triggering the one-repair path).
    """
    m = _JSON_OBJ.search(raw_text or "")
    if not m:
        return raw_text or ""
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        # One defensive repair pass (bare hex elements, trailing commas).
        try:
            data = json.loads(_repair_json_text(blob))
        except (json.JSONDecodeError, TypeError):
            return raw_text or ""
    if not isinstance(data, dict):
        return raw_text or ""

    hyps = data.get("mechanic_hypotheses", []) or []
    goals = data.get("goal_signals", []) or []

    fixed_hyps = []
    for h in hyps[:3]:
        if not isinstance(h, dict):
            continue
        h = dict(h)
        h.setdefault("hypothesis_id", _new_hex_id())
        h.setdefault("belief_id", _new_hex_id())
        h.setdefault("schema_version", SCHEMA_VERSION)
        h.setdefault("supporting_ids", [])
        h.setdefault("precondition_tags", [])
        h.setdefault("effect_tags", [])
        # Clamp affected_colors to valid 0..15 ints (models sometimes emit hex,
        # out-of-range values, or every color).
        if "affected_colors" in h:
            h["affected_colors"] = _clamp_colors(h["affected_colors"])
        # Coerce common issues
        if "confidence" not in h:
            h["confidence"] = 0.3
        fixed_hyps.append(h)

    fixed_goals = []
    for g in goals[:2]:
        if not isinstance(g, dict):
            continue
        g = dict(g)
        g.setdefault("goal_id", _new_hex_id())
        g.setdefault("schema_version", SCHEMA_VERSION)
        g.setdefault("status", "provisional")
        if "priority" not in g:
            g["priority"] = 0.5
        fixed_goals.append(g)

    return json.dumps({
        "mechanic_hypotheses": fixed_hyps,
        "goal_signals": fixed_goals,
    })


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class TransformersDeliberator(DeliberatorInterface):
    """Deliberator backed by a local HF causal LM (4-bit NF4 on CUDA).

    Parameters
    ----------
    config : DeliberatorConfig
    model_path : str | Path
        Directory with HF safetensors + tokenizer + config.
    device : str
        'cuda' / 'cuda:0'. 4-bit load requires CUDA.
    load_now : bool
        If True (default), load the model in __init__. If False, defer to
        `ensure_loaded()` (useful for tests that inject a fake generator).
    generate_fn : callable | None
        Test seam: if provided, used instead of the real model. Signature:
        generate_fn(prompt: str, max_new_tokens: int) -> str. When set, no
        weights are loaded (CPU-friendly unit tests).
    """

    def __init__(
        self,
        config: DeliberatorConfig,
        model_path: str | Path,
        *,
        device: str = "cuda",
        load_now: bool = True,
        generate_fn: Callable[[str, int], str] | None = None,
        clock: Callable[[], float] = monotonic,
    ):
        super().__init__(config, clock)
        self.model_path = Path(model_path)
        self.device = device
        self._generate_fn = generate_fn
        self._model = None
        self._tokenizer = None
        if generate_fn is None and load_now:
            self.ensure_loaded()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load tokenizer + 4-bit model. Raises with a clear message on failure."""
        if self._generate_fn is not None or self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Deliberator checkpoint dir not found: {self.model_path}"
            )
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers + bitsandbytes are required for the 4-bit "
                f"deliberator backend: {exc}"
            ) from exc

        if not self.device.startswith("cuda"):
            raise RuntimeError(
                "4-bit (bitsandbytes NF4) load requires CUDA; "
                f"got device={self.device!r}. Use a CUDA device on Kaggle."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; cannot load the 4-bit deliberator."
            )

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            quantization_config=quant_cfg,
            device_map={"": self.device},
            torch_dtype=torch.float16,
        )
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        return self._generate_fn is not None or self._model is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int, *, deadline: float = float("inf")) -> str:
        """Run the model (or the injected test generator) and return raw text."""
        if self._clock() >= deadline:
            raise TimeoutError("Deliberator deadline exhausted before generation")
        if self._generate_fn is not None:
            text = self._generate_fn(prompt, max_new_tokens)
            if self._clock() >= deadline:
                raise TimeoutError("Deliberator exceeded deadline")
            return text

        self.ensure_loaded()
        import torch  # noqa: PLC0415
        from transformers import StoppingCriteria, StoppingCriteriaList
        clock = self._clock

        class DeadlineStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return clock() >= deadline

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([DeadlineStop()]),
            )
        if self._clock() >= deadline:
            raise TimeoutError("Deliberator generation reached deadline")
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # DeliberatorInterface hooks
    # ------------------------------------------------------------------

    def _call_model(self, inp: DeliberatorInput, *, deadline: float) -> str:
        prompt = build_prompt(inp)
        raw = self._generate(prompt, self.config.max_new_tokens, deadline=deadline)
        return normalize_model_json(raw)

    def _repair(
        self, inp: DeliberatorInput, raw: str, error: str, *, deadline: float
    ) -> str | None:
        """One repair pass: re-ask with the parse error, then re-normalize."""
        if self._clock() >= deadline:
            return None
        repair_prompt = (
            build_prompt(inp)
            + f"\n\nYour previous output could not be parsed ({error}). "
            "Output STRICT JSON ONLY, matching the schema exactly, no prose."
        )
        try:
            raw2 = self._generate(repair_prompt, self.config.max_new_tokens, deadline=deadline)
        except Exception:
            return None
        return normalize_model_json(raw2)

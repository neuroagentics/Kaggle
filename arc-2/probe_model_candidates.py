"""Fast candidate-health probe for the ARC code agent's model bakeoff.

This does NOT measure solve rate. It measures whether a model can emit
well-formed, sandbox-valid, executable, distinct candidate programs under our
contract at all -- the thing that actually gates bakeoff yield (most candidates
in prior runs died to invalid JSON / syntax / duplicates, not to being wrong).

For each model it runs a single planning round on one trivial task and reports:
  json_ok        -- response parsed and carried a hypotheses array
  candidates     -- number of hypotheses returned
  sandbox_valid  -- fraction whose code passed validate_code
  executed       -- fraction that ran in the worker and returned a valid grid
  distinct       -- number of distinct code digests
  solved_demo    -- did any candidate exactly reproduce the demo (bonus, not required)

A model with high sandbox_valid/executed/distinct has potential even if it does
not solve the probe; a model near zero there will not improve with more rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from hyper_arc.contender.agentic_reasoner import AgenticReasoner
from hyper_arc.contender.model_proposer import _ollama_transport


# A deliberately easy, unambiguous task: reflect each row (horizontal mirror).
PROBE_TASK = {
    "train": [
        {"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]},
        {"input": [[7, 0], [0, 8]], "output": [[0, 7], [8, 0]]},
    ],
    "test": [{"input": [[9, 1, 2]]}],
}

DEFAULT_MODELS = ["gemma4:latest", "qwen2.5-coder:7b", "qwen3:8b", "qwen3-vl:8b"]


def _probe_model(
    model: str,
    *,
    candidates: int,
    endpoint: str,
    timeout: float,
    max_output_tokens: int,
) -> dict:
    reasoner = AgenticReasoner(
        _ollama_transport(model, endpoint, timeout),
        model=model,
        rounds=1,
        candidates_per_round=candidates,
        max_output_tokens=max_output_tokens,
    )
    started = time.perf_counter()
    try:
        result = reasoner.solve("probe-reflect-h", PROBE_TASK)
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        return {
            "model": model,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
    elapsed = time.perf_counter() - started
    generated = result.candidates_generated
    executed = result.candidates_executed
    # Distinct executable candidates == generated minus duplicate rejections.
    duplicates = sum(1 for item in result.failures if item.endswith(":duplicate"))
    syntax = sum(1 for item in result.failures if "invalid Python" in item)
    forbidden = sum(1 for item in result.failures if "not allowed" in item)
    worker_err = sum(1 for item in result.failures if "worker rejected" in item)
    non_exact = sum(1 for item in result.failures if ":non-exact:" in item)
    return {
        "model": model,
        "json_ok": generated > 0,
        "candidates": generated,
        "executed": executed,
        "executed_rate": round(executed / generated, 2) if generated else 0.0,
        "duplicates": duplicates,
        "syntax_errors": syntax,
        "forbidden_api": forbidden,
        "worker_errors": worker_err,
        "non_exact_but_ran": non_exact,
        "solved_demo": bool(result.hypotheses),
        "sample_rejections": list(result.failures[:6]),
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:11434/api/chat"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--output")
    args = parser.parse_args()

    reports = []
    for model in args.models:
        report = _probe_model(
            model,
            candidates=args.candidates,
            endpoint=args.endpoint,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )
        print(json.dumps(report, indent=2), flush=True)
        reports.append(report)

    # A model "has potential" if it returns JSON and at least one candidate both
    # passed the sandbox and executed to a valid grid.
    promising = [
        r["model"]
        for r in reports
        if r.get("json_ok") and r.get("executed", 0) > 0
    ]
    summary = {
        "probe": "reflect-horizontal",
        "candidates_per_model": args.candidates,
        "promising_models": promising,
        "reports": reports,
    }
    print(json.dumps({"promising_models": promising}, indent=2), flush=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

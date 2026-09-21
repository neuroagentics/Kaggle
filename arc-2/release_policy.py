"""Shared fail-closed runtime qualification (not a claim of solver quality)."""

RELEASE_ID = "hyper-arc-rc-20260920"
MODEL_ID = "google/gemma-4-e4b-it"
MODEL_ATTACHMENT = "google/gemma-4/Transformers/gemma-4-e4b-it/1"
# Conservative admission floor, not a guarantee for every context length.
# E4B BF16 did not fit the 14.56 GiB T4 allocated in private qualification v2.
MODEL_MIN_GPU_MEMORY_GIB = 20.0
MODEL_MIN_GPUS = 2  # Real ARC prompt generation OOMed on one L4 in v7.
TASK_SECONDS = 120.0
RUN_SECONDS = 34200.0  # 9.5 hours; outer notebook watchdog adds a small margin.
AGENTIC_ROUNDS = 3
AGENTIC_CANDIDATES = 4
OUTPUT_TOKENS = 3000


def qualify_run(report):
    """Reject structurally valid artifacts with missing or ineffective AI paths.

    Exact hidden answers are unavailable here. This proves runtime health only;
    independent quality, licensing, rules, and clean-run gates remain mandatory.
    """
    failures = []
    config, channels = report.get("configuration", {}), report.get("channels", {})
    if config.get("release_id") != RELEASE_ID:
        failures.append("release identity mismatch")
    if not config.get("agentic_ai_enabled"):
        failures.append("AI disabled")
    for key in ("agentic_invocations", "agentic_candidates_generated", "agentic_candidates_executed"):
        if channels.get(key, 0) < 1:
            failures.append(f"no {key}")
    if not report.get("model_preflight", {}).get("passed"):
        failures.append("live model preflight not passed")
    if channels.get("agentic_unusable_invocations", 0):
        failures.append("neural tasks produced no executable candidates")
    tasks = report.get("tasks", {})
    if tasks.get("total", 0) < 1 or tasks.get("outputs", 0) < 1:
        failures.append("empty run")
    if tasks.get("failures") or tasks.get("timed_out"):
        failures.append("task failures or timeouts")
    if report.get("elapsed_seconds", float("inf")) > config.get("global_timeout_seconds", 0):
        failures.append("global budget exceeded")
    if failures:
        raise RuntimeError("Run is not qualified: " + "; ".join(failures))


def model_preflight(transport, model, seconds=60.0):
    """Use the real transport, not a stub; keep counters separate from tasks."""
    import time
    from hyper_arc.contender.agentic_reasoner import AgenticReasoner

    task = {"train": [
        {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
        {"input": [[4, 5], [6, 7]], "output": [[5, 4], [7, 6]]},
    ], "test": [{"input": [[8, 0, 9]]}]}
    result = AgenticReasoner(transport, model=model, rounds=2, candidates_per_round=1,
                            max_output_tokens=1600).solve(
        "runtime-preflight-reflection", task, deadline=time.monotonic() + seconds)
    expected = ((9, 0, 8),)
    passed = bool(result.test_predictions and expected in result.test_predictions[0])
    evidence = {"passed": passed, "scope": "synthetic-runtime-only",
                "generated": result.candidates_generated, "executed": result.candidates_executed,
                "failures": list(result.failures)}
    if not passed:
        raise RuntimeError(f"Live model preflight failed: {evidence}")
    return evidence

"""Regression evidence for the September alignment corrections; no real weights."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from release_policy import RELEASE_ID, qualify_run, model_preflight
from hyper_arc.contender.agentic_reasoner import AgenticReasoner
from hyper_arc.contender.offline_model import OfflineTransformersTransport, OfflineModelError
from hyper_arc.contender.session_memory import SessionMemory


def good_report():
    return {"configuration": {"release_id": RELEASE_ID, "agentic_ai_enabled": True,
                               "global_timeout_seconds": 100},
            "channels": {"agentic_invocations": 1, "agentic_candidates_generated": 2,
                         "agentic_candidates_executed": 1},
            "model_preflight": {"passed": True},
            "tasks": {"total": 1, "outputs": 1, "failures": [], "timed_out": []},
            "elapsed_seconds": 50}


def test_runtime_qualification_does_not_require_hidden_answers():
    qualify_run(good_report())


@pytest.mark.parametrize("key", ["agentic_invocations", "agentic_candidates_generated", "agentic_candidates_executed"])
def test_calls_without_usable_candidates_do_not_qualify(key):
    report = good_report()
    report["channels"][key] = 0
    with pytest.raises(RuntimeError, match=key):
        qualify_run(report)


@pytest.mark.parametrize("mutation", ["preflight", "timeout", "failure", "deadline", "identity"])
def test_unhealthy_runs_fail_closed(mutation):
    report = good_report()
    if mutation == "preflight": report["model_preflight"]["passed"] = False
    if mutation == "timeout": report["tasks"]["timed_out"] = ["a"]
    if mutation == "failure": report["tasks"]["failures"] = ["a"]
    if mutation == "deadline": report["elapsed_seconds"] = 101
    if mutation == "identity": report["configuration"]["release_id"] = "v10"
    with pytest.raises(RuntimeError): qualify_run(report)


def test_historical_false_promotion_is_rejected():
    path = Path(__file__).parents[1] / "artifacts/v10_terminal_audit_20260910/run_manifest.json"
    with pytest.raises(RuntimeError, match="no agentic_candidates_executed"):
        qualify_run(json.loads(path.read_text()))


def response(codes):
    return {"message": {"content": json.dumps({"hypotheses": [
        {"rule": "candidate", "code": code} for code in codes]})}}


def test_expired_reasoning_budget_never_calls_model():
    def forbidden(_): raise AssertionError("called model after deadline")
    result = AgenticReasoner(forbidden, model="stub").solve("a", {
        "train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]]}]
    }, deadline=time.monotonic() - 1)
    assert result.rounds_executed == 0
    assert "deadline" in result.failures[0]


def test_duplicate_behaviors_leave_room_for_a_distinct_second_attempt():
    codes = ["def transform(grid):\n    return grid",
             "def transform(grid):\n    return [row[:] for row in grid]",
             "def transform(grid):\n    return [row[::-1] for row in grid]"]
    result = AgenticReasoner(lambda _: response(codes), model="stub", rounds=1,
                            candidates_per_round=3).solve("a", {
        "train": [{"input": [[1, 1]], "output": [[1, 1]]}],
        "test": [{"input": [[2, 3]]}]})
    assert len(result.hypotheses) == 2
    assert len(set(result.test_predictions[0])) == 2


def test_session_procedure_transfers_only_after_current_demo_replay():
    code = "def transform(grid):\n    return [row[::-1] for row in grid]"
    task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}]}
    result = AgenticReasoner(lambda _: response([code]), model="stub", rounds=1).solve("a", task)
    bank = SessionMemory()
    bank.remember(result.hypotheses)
    next_task = {"train": [{"input": [[5, 6]], "output": [[6, 5]]}],
                 "test": [{"input": [[7, 8]]}]}
    assert bank.predict(next_task, time.monotonic() + 5) == [[[[8, 7]]]]
    assert bank.hits == 1
    next_task["train"][0]["output"] = [[9, 9]]
    assert bank.predict(next_task, time.monotonic() + 5) == []


def fake_transport(prompt_tokens=2):
    transport = OfflineTransformersTransport.__new__(OfflineTransformersTransport)
    transport.torch, transport.device, transport.path = torch, "cpu", Path("stub")
    transport.max_input_tokens = 10
    captured = {}
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, **kwargs):
            captured["tokenization"] = kwargs
            return {"input_ids": torch.zeros((1, prompt_tokens), dtype=torch.long)}
    def template(messages, **kwargs):
        captured["messages"] = messages
        return "prompt"
    def generate(**kwargs):
        captured["generation"] = kwargs
        return torch.zeros((1, prompt_tokens + 1), dtype=torch.long)
    transport.tokenizer = Tokenizer()
    transport.processor = SimpleNamespace(apply_chat_template=template, decode=lambda *a, **k: '{}')
    transport.model = SimpleNamespace(generate=generate)
    return transport, captured


def test_offline_transport_passes_budget_and_schema_without_silent_truncation():
    transport, captured = fake_transport()
    payload = {"messages": [{"role": "user", "content": "task"}],
               "format": {"type": "object"}, "options": {"max_time": 5}}
    saved = copy.deepcopy(payload)
    transport(payload)
    assert payload == saved
    assert "schema" in captured["messages"][-1]["content"]
    assert 0 < captured["generation"]["max_time"] <= 5
    assert captured["tokenization"]["truncation"] is False


def test_oversized_prompt_is_rejected_before_generation():
    transport, captured = fake_transport(prompt_tokens=11)
    with pytest.raises(OfflineModelError, match="refusing to truncate"):
        transport({"messages": [{"role": "user", "content": "task"}]})
    assert "generation" not in captured


def test_live_preflight_rejects_unusable_model_output():
    with pytest.raises(RuntimeError, match="preflight failed"):
        model_preflight(lambda _: {"message": {"content": "not JSON"}}, "stub")

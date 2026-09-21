"""Ephemeral demo-supported procedures. Never labeled as known test successes."""

import subprocess
import time
from dataclasses import dataclass, field

from hyper_arc.contender.agentic_reasoner import execute_code


@dataclass
class SessionMemory:
    # Hypotheses are code proposals supported by source demonstrations only.
    records: dict = field(default_factory=dict)
    max_records: int = 128
    hits: int = 0

    def remember(self, hypotheses):
        for hypothesis in hypotheses:
            self.records[hypothesis.digest] = hypothesis
        while len(self.records) > self.max_records:
            del self.records[next(iter(self.records))]

    def predict(self, task, deadline):
        inputs = [p["input"] for p in task["train"]]
        outputs = [p["output"] for p in task["train"]]
        tests = [p["input"] for p in task["test"]]
        matches = []
        for hypothesis in reversed(list(self.records.values())):
            def budget():
                return min(2.0, max(0.001, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                break
            try:
                if execute_code(hypothesis.code, inputs, timeout_seconds=budget()) != outputs:
                    continue
                if time.monotonic() >= deadline:
                    break
                prediction = execute_code(hypothesis.code, tests, timeout_seconds=budget())
                if prediction not in matches:
                    matches.append(prediction)
                if len(matches) == 2:
                    break
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
        self.hits += bool(matches)
        return matches

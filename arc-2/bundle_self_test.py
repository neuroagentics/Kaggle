"""Fail-closed runtime sentinel for an extracted Hyper-ARC bundle."""

from __future__ import annotations

import json
from pathlib import Path

from hyper_arc.contender.agentic_memory import AgenticMemoryBank
from hyper_arc.contender.agentic_reasoner import AgenticReasoner
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)
from hyper_arc.hpm import GlobalMemoryBank
from main import solve_task, validate_submission


ROOT = Path(__file__).resolve().parent


def main() -> int:
    procedural_path = ROOT / "hyper_arc" / "procedural_memory_v1.json"
    agentic_memory_path = ROOT / "hyper_arc" / "agentic_memory_builder_v2.json"

    seed_memory = GlobalMemoryBank(k=5)
    procedural_memory = ProceduralMemoryBank.load(procedural_path)
    agentic_memory = AgenticMemoryBank.load(agentic_memory_path)
    if not procedural_memory.records:
        raise RuntimeError("Required procedural-memory artifact is empty")
    if any(r.validation_scope != "builder" for r in agentic_memory.records):
        raise RuntimeError("Static agentic memory contains non-builder evidence")

    fixture = {
        "train": [
            {"input": [[1, 0]], "output": [[1, 0]]},
            {"input": [[2, 0]], "output": [[2, 0]]},
        ],
        "test": [{"input": [[3, 0]]}],
    }
    attempts, score, exact, _ = solve_task(
        "bundle-sentinel",
        fixture,
        seed_memory,
        global_prior_weight=0.2,
        timeout_sec=1.0,
        max_iterations=1,
        experience_bank=None,
        world_model=None,
        procedural_memory=procedural_memory,
    )
    validate_submission({"bundle-sentinel": attempts}, {"bundle-sentinel": fixture})
    if not exact or score != 1.0:
        raise RuntimeError("Deterministic exact-replay sentinel failed")

    structural_fixture = {
        "train": [
            {
                "input": [
                    [1, 0, 0, 2, 2, 0],
                    [0, 0, 0, 2, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [1, 0, 0, 3, 3, 0],
                    [0, 0, 0, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [4, 4, 0, 0, 5, 0],
                    [4, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                "output": [
                    [3, 3, 0, 0, 5, 0],
                    [3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            },
        ],
        "test": [
            {
                "input": [
                    [6, 0, 0, 7, 7, 0],
                    [0, 0, 0, 7, 7, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            }
        ],
    }
    promoted_world_model = RecursiveReasoningWorldModel(
        HyperbolicWorldMemory(),
        config=WorldModelConfig(
            max_depth=0,
            max_memory_seeds=0,
            max_object_seeds=0,
            max_relational_seeds=0,
            retrieve_memory=False,
            learn_memory=False,
        ),
    )
    world_result = promoted_world_model.solve("structural-sentinel", structural_fixture)
    expected = (
        (6, 0, 0, 3, 3, 0),
        (0, 0, 0, 3, 3, 0),
        (0, 0, 0, 0, 0, 0),
    )
    if expected not in world_result.test_predictions[0]:
        raise RuntimeError("Promoted structural world-model sentinel failed")
    if not any(
        item.provenance.get("source") == "structural-world"
        for item in world_result.hypotheses
    ):
        raise RuntimeError("Structural hypothesis lineage is missing")

    neural_fixture = {
        "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
        "test": [{"input": [[0, 0, 1]]}],
    }

    def stub_transport(_payload):
        return {
            "message": {
                "content": json.dumps(
                    {
                        "hypotheses": [
                            {
                                "id": "invert",
                                "rule": "invert binary colors",
                                "evidence": "exact training relation",
                                "algorithm": "replace each v with 1-v",
                                "code": (
                                    "def transform(grid):\n"
                                    "    return [[1-v for v in row] for row in grid]"
                                ),
                            }
                        ]
                    }
                )
            }
        }

    neural_result = AgenticReasoner(
        stub_transport,
        model="bundle-stub",
        rounds=1,
        candidates_per_round=1,
    ).solve("neural-sentinel", neural_fixture)
    if not neural_result.hypotheses:
        raise RuntimeError("Agentic perception-planning-execution sentinel failed")
    if neural_result.test_predictions != (((((1, 1, 0),),),)):
        raise RuntimeError("Agentic sentinel produced an unexpected prediction")

    print(
        json.dumps(
            {
                "status": "PASS",
                "procedural_records": len(procedural_memory.records),
                "deterministic_exact": exact,
                "recursive_world_model_exact": True,
                "agentic_pipeline_exact": True,
                "neural_inference_tested": False,
                "sentinel_scope": "stub-backed-structural-only",
                "agentic_memory_records": len(agentic_memory.records),
                "inactive_channels_packaged": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

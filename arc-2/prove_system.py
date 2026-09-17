"""Generate machine-checkable evidence that Hyper-ARC exists and runs end to end."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
from hyper_arc.contender.recursive_specialist import (
    RecursiveGridSpecialist,
    SpecialistConfig,
)
from main import load_challenges, main as solver_main, validate_submission


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _neural_proof() -> dict:
    checkpoint_path = ROOT / "artifacts" / "recursive_specialist_full_v2.pt"
    validation_path = ROOT / "artifacts" / "recursive_specialist_validation_v1.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model = RecursiveGridSpecialist(SpecialistConfig(**metadata["config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    demonstrations = [([[1, 0], [0, 0]], [[1, 0], [0, 0]])]
    query = [[0, 2], [0, 0]]
    first = model.predict(demonstrations, query)
    second = model.predict(demonstrations, query)
    validation = _load_json(validation_path)
    return {
        "checkpoint": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "class": type(model).__name__,
        "parameter_count": model.parameter_count,
        "strict_state_load": True,
        "inference_executed": True,
        "deterministic_repeat": first == second,
        "output_shape": [len(first), len(first[0])],
        "promoted_to_submission_solver": False,
        "promotion_blocker": {
            "leave_one_out_eligible_tasks": validation["leave_one_out_eligible_tasks"],
            "specialist_pass_at_2": validation["specialist_pass_at_2"],
            "unique_exact_contribution": validation["ensemble_unique"],
        },
    }


def _fixtures() -> tuple[dict, dict]:
    challenges = {
        "identity-proof": {
            "train": [{"input": [[1, 0]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 0]]}],
        },
        "world-model-proof": {
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
        },
    }
    expected = {
        "identity-proof": [[[2, 0]]],
        "world-model-proof": [
            [
                [6, 0, 0, 3, 3, 0],
                [0, 0, 0, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ],
    }
    return challenges, expected


def _end_to_end_proof() -> dict:
    challenges, expected = _fixtures()
    with tempfile.TemporaryDirectory(prefix="hyper-arc-proof-") as directory:
        root = Path(directory)
        challenge_path = root / "arc-agi_test_challenges.json"
        output_path = root / "submission.json"
        manifest_path = root / "run_manifest.json"
        challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
        exit_code = solver_main(
            [
                "--challenge-file",
                str(challenge_path),
                "--output",
                str(output_path),
                "--run-manifest",
                str(manifest_path),
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--procedural-memory",
                str(ROOT / "hyper_arc" / "procedural_memory_v1.json"),
                "--enable-procedural-memory",
                "--enable-world-model",
                "--max-iterations",
                "1",
                "--task-timeout",
                "5",
                "--global-timeout",
                "30",
                "--no-resume",
            ]
        )
        submission = _load_json(output_path)
        validate_submission(submission, load_challenges(challenge_path))
        exact = {
            task_id: all(
                target in submission[task_id][index].values()
                for index, target in enumerate(outputs)
            )
            for task_id, outputs in expected.items()
        }
        manifest = _load_json(manifest_path)
        return {
            "exit_code": exit_code,
            "submission_schema_valid": True,
            "submission_sha256_verified": manifest["submission"]["sha256"]
            == _sha256(output_path),
            "exact_fixture_results": exact,
            "all_fixtures_exact": all(exact.values()),
            "failures": manifest["tasks"]["failures"],
            "timeouts": manifest["tasks"]["timed_out"],
            "channels": manifest["configuration"],
        }


def main() -> int:
    memory_path = ROOT / "hyper_arc" / "procedural_memory_v1.json"
    memory = ProceduralMemoryBank.load(memory_path)
    validation = _load_json(
        ROOT / "artifacts" / "phase_c_final_validation_v1.json"
    )
    holdout = _load_json(ROOT / "artifacts" / "phase_c_final_holdout_v1.json")
    report = {
        "schema_version": 1,
        "claim": "Hyper-ARC implementation and end-to-end runtime proof",
        "neural_specialist": _neural_proof(),
        "procedural_memory": {
            "path": str(memory_path),
            "sha256": _sha256(memory_path),
            "bank_id": memory.bank_id,
            "records": len(memory.records),
            "exact_success_episodes": sum(
                record.exact_successes for record in memory.records
            ),
            "transfer_failure_episodes": sum(
                record.transfer_failures for record in memory.records
            ),
            "integrity_verified": True,
        },
        "recursive_solver_exact_evidence": {
            "validation": {
                "tasks": validation["evaluated_tasks"],
                "world_model_pass_at_2": validation["world_model_pass_at_2"],
                "combined_pass_at_2": validation["combined_pass_at_2"],
                "combined_regressions": validation["combined_regressions_vs_owned"],
            },
            "holdout": {
                "tasks": holdout["evaluated_tasks"],
                "world_model_pass_at_2": holdout["world_model_pass_at_2"],
                "combined_pass_at_2": holdout["combined_pass_at_2"],
                "combined_regressions": holdout["combined_regressions_vs_owned"],
            },
            "metric": "full-task exact pass@2",
        },
        "end_to_end": _end_to_end_proof(),
    }
    destination = ROOT / "artifacts" / "system_proof_v1.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["end_to_end"]["all_fixtures_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Kaggle-compatible orchestration for the Hyper-ARC solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from hyper_arc.esb import ESB
from hyper_arc.contender.experience_bank import ExperienceBank, experience_candidates
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.contender.world_model import (
    RecursiveReasoningWorldModel,
    WorldModelConfig,
)
from hyper_arc.deterministic import exact_candidates
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MAX_ITERATIONS, MCTSEngine


IS_KAGGLE = Path("/kaggle/working").exists()
DEFAULT_DATA_DIR = (
    Path("/kaggle/input/arc-prize-2026-arc-agi-2")
    if IS_KAGGLE
    else Path("./data/arc-agi-2")
)
DEFAULT_OUTPUT = (
    Path("/kaggle/working/submission.json") if IS_KAGGLE else Path("./submission.json")
)
DEFAULT_CHECKPOINT = (
    Path("/kaggle/working/checkpoint.pt") if IS_KAGGLE else Path("./checkpoint.pt")
)
DEFAULT_RUN_MANIFEST = (
    Path("/kaggle/working/run_manifest.json")
    if IS_KAGGLE
    else Path("./run_manifest.json")
)
DEFAULT_SEED_BANK = Path("./hyper_arc/seed_bank.json")
DEFAULT_EXPERIENCE_BANK = Path("./hyper_arc/experience_bank_v2.json")
DEFAULT_WORLD_MEMORY = Path("./hyper_arc/world_memory_v1.json")


def log(message: str) -> None:
    print(message, flush=True)


def load_challenges(file_path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {task["task_id"]: task for task in data}
    raise ValueError("Unrecognized challenge file format.")


def resolve_challenge_file(
    challenge_file: str | None,
    data_dir: str | Path,
) -> Path:
    if challenge_file:
        path = Path(challenge_file)
        if not path.is_file():
            raise FileNotFoundError(f"Challenge file not found: {path}")
        return path

    root = Path(data_dir)
    candidates = (
        sorted(root.glob("**/arc-agi_test_challenges.json")) if root.exists() else []
    )
    if not candidates:
        raise FileNotFoundError(
            f"No arc-agi_test_challenges.json found under {root}. "
            "Attach the ARC competition data or pass --challenge-file."
        )
    return candidates[0]


def grid_to_esb(grid_2d: list[list[int]]) -> ESB:
    return ESB.from_grid(grid_2d)


def stable_task_seed(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16)


def _valid_grid(grid: Any) -> bool:
    if (
        not isinstance(grid, list)
        or not grid
        or not all(isinstance(row, list) and row for row in grid)
    ):
        return False
    width = len(grid[0])
    if len(grid) > 30 or width > 30:
        return False
    return all(
        len(row) == width
        and all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9
            for value in row
        )
        for row in grid
    )


def _copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _identity_supported(task_data: dict[str, Any]) -> bool:
    pairs = task_data.get("train", [])
    return bool(pairs) and all(pair.get("input") == pair.get("output") for pair in pairs)


def _two_attempts(
    ranked: list[list[list[int]]], input_grid: list[list[int]]
) -> dict[str, list[list[int]]]:
    unique: list[list[list[int]]] = []
    for prediction in ranked:
        if _valid_grid(prediction) and prediction not in unique:
            unique.append(prediction)
        if len(unique) == 2:
            break
    if not unique:
        unique.append(_copy_grid(input_grid))
    if len(unique) == 1:
        # A duplicate honest best candidate is preferable to injecting an
        # unsupported identity transformation and calling it a second theory.
        unique.append(_copy_grid(unique[0]))
    return {"attempt_1": unique[0], "attempt_2": unique[1]}


def failure_fallback(task_data: dict[str, Any]) -> list[dict[str, list[list[int]]]]:
    """Declared last-resort policy used only after a recorded task failure."""
    return [
        {
            "attempt_1": _copy_grid(pair["input"]),
            "attempt_2": _copy_grid(pair["input"]),
        }
        for pair in task_data.get("test", [])
    ]


def validate_submission(
    submission: dict[str, list[dict[str, list[list[int]]]]],
    challenges: dict[str, dict[str, Any]],
) -> None:
    if set(submission) != set(challenges):
        missing = sorted(set(challenges) - set(submission))
        extra = sorted(set(submission) - set(challenges))
        raise ValueError(
            f"Submission task IDs mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    for task_id, task in challenges.items():
        entries = submission[task_id]
        expected = len(task.get("test", []))
        if not isinstance(entries, list) or len(entries) != expected:
            raise ValueError(
                f"Task {task_id}: expected {expected} test predictions, got {len(entries)}"
            )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"Task {task_id} test {index}: invalid attempt object")
            for name in ("attempt_1", "attempt_2"):
                if not _valid_grid(entry[name]):
                    raise ValueError(
                        f"Task {task_id} test {index}: invalid {name} grid"
                    )


def save_checkpoint(
    path: Path,
    submission: dict[str, Any],
    completed_ids: set[str],
    global_memory: GlobalMemoryBank,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "submission": submission,
            "completed_ids": sorted(completed_ids),
            "global_memory_state": global_memory.state_dict_entries(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    global_memory: GlobalMemoryBank,
) -> tuple[dict[str, Any], set[str]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older Kaggle torch images
        payload = torch.load(path, map_location="cpu")
    global_memory.restore_from_state_dict(payload.get("global_memory_state", []))
    return dict(payload.get("submission", {})), set(payload.get("completed_ids", []))


def solve_task(
    task_id: str,
    task_data: dict[str, Any],
    global_memory: GlobalMemoryBank,
    global_prior_weight: float,
    timeout_sec: float,
    max_iterations: int,
    experience_bank: ExperienceBank | None = None,
    world_model: RecursiveReasoningWorldModel | None = None,
) -> tuple[list[dict[str, list[list[int]]]], float, bool, int]:
    raw_train_pairs = [
        (pair["input"], pair["output"]) for pair in task_data.get("train", [])
    ]
    train_pairs = [
        (grid_to_esb(pair["input"]), grid_to_esb(pair["output"]))
        for pair in task_data.get("train", [])
    ]
    if not train_pairs:
        raise ValueError("Task has no training pairs")

    test_inputs = [pair["input"] for pair in task_data.get("test", [])]
    deterministic = exact_candidates(raw_train_pairs, test_inputs)
    relational = exact_relational_candidates(task_data)
    memory = (
        experience_candidates(experience_bank, task_data)
        if experience_bank is not None
        else []
    )
    world_result = world_model.solve(task_id, task_data) if world_model else None
    if (
        deterministic
        or relational
        or memory
        or (world_result is not None and world_result.hypotheses)
    ):
        attempts = []
        for index, input_grid in enumerate(test_inputs):
            baseline_ranked = [
                candidate.predict(input_grid) for candidate in deterministic
            ]
            baseline = []
            for prediction in baseline_ranked:
                if _valid_grid(prediction) and prediction not in baseline:
                    baseline.append(prediction)
                if len(baseline) == 2:
                    break
            relational_predictions = [
                candidate.predict(input_grid) for candidate in relational
            ]
            memory_predictions = [candidate.predict(index) for candidate in memory]
            world_predictions = (
                [
                    [list(row) for row in prediction]
                    for prediction in world_result.test_predictions[index]
                ]
                if world_result is not None
                else []
            )
            ranked = [
                *baseline[:1],
                *world_predictions,
                *memory_predictions,
                *relational_predictions,
                *baseline[1:],
            ]
            attempts.append(_two_attempts(ranked, input_grid))
        complexity = (
            world_result.hypotheses[0].complexity
            if world_result is not None and world_result.hypotheses
            else memory[0].complexity
            if memory
            else relational[0].complexity
            if relational
            else deterministic[0].complexity
            if deterministic
            else 0
        )
        return attempts, 1.0, True, complexity

    engine = MCTSEngine(
        global_memory=global_memory,
        local_memory=LocalTaskBuffer(),
        C=1.41,
        global_prior_weight=global_prior_weight,
        max_iterations=max_iterations,
        random_seed=stable_task_seed(task_id),
    )
    program = engine.solve(train_pairs, timeout_sec=timeout_sec)
    score, exact = engine.score_program(program, train_pairs)

    attempts: list[dict[str, list[list[int]]]] = []
    for test_pair in task_data.get("test", []):
        input_grid = test_pair["input"]
        test_esb = grid_to_esb(input_grid)
        try:
            prediction = engine.apply_program(program, test_esb).to_grid()
        except (ValueError, IndexError, RuntimeError) as exc:
            raise RuntimeError("MCTS program failed on a test input") from exc
        alternatives = [prediction]
        if _identity_supported(task_data):
            alternatives.append(input_grid)
        attempts.append(_two_attempts(alternatives, input_grid))
    return attempts, score, exact, len(program)


def write_submission(path: Path, submission: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(submission, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 production solver")
    parser.add_argument("--challenge-file", help="Explicit aggregate challenge JSON")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--run-manifest")
    parser.add_argument("--seed-bank", default=str(DEFAULT_SEED_BANK))
    parser.add_argument("--experience-bank", default=str(DEFAULT_EXPERIENCE_BANK))
    parser.add_argument("--world-memory", default=str(DEFAULT_WORLD_MEMORY))
    parser.add_argument("--task-timeout", type=float, default=90.0)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--global-timeout", type=float, default=36_000.0)
    parser.add_argument("--enable-world-model", action="store_true")
    parser.add_argument(
        "--allow-missing-seed-bank",
        action="store_true",
        help="Development-only escape hatch; Kaggle bundles must not use it.",
    )
    parser.add_argument(
        "--task-failure-policy",
        choices=("abort", "record-input"),
        default="record-input",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    challenge_path = resolve_challenge_file(args.challenge_file, args.data_dir)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    run_manifest_path = (
        Path(args.run_manifest)
        if args.run_manifest
        else output_path.parent / DEFAULT_RUN_MANIFEST.name
    )
    seed_bank_path = Path(args.seed_bank)
    experience_bank_path = Path(args.experience_bank)
    world_memory_path = Path(args.world_memory)

    log(f"[INFO] Loading challenges from {challenge_path}")
    challenges = load_challenges(challenge_path)
    log(f"[INFO] Found {len(challenges)} tasks")

    if not experience_bank_path.is_file():
        raise FileNotFoundError(
            f"Required experience bank not found: {experience_bank_path}"
        )
    experience_bank = ExperienceBank.load(experience_bank_path)
    log(
        f"[INFO] Loaded experience bank {experience_bank.bank_id[:12]} "
        f"with {len(experience_bank.records)} records"
    )
    world_model = None
    world_memory_items = 0
    if args.enable_world_model:
        if not world_memory_path.is_file():
            raise FileNotFoundError(
                f"Enabled world memory not found: {world_memory_path}"
            )
        world_memory = HyperbolicWorldMemory.load(world_memory_path)
        world_memory_items = len(world_memory.items)
        if not world_memory_items:
            raise ValueError("Enabled world memory is empty")
        world_model = RecursiveReasoningWorldModel(
            world_memory,
            config=WorldModelConfig(learn_memory=False),
        )
        log(
            f"[INFO] Enabled recursive world model with "
            f"{world_memory_items} executable memories"
        )
    else:
        log("[INFO] Recursive world model disabled: not promoted by exact-score gate")

    global_memory = GlobalMemoryBank(k=5)
    global_prior_weight = 0.0
    if seed_bank_path.is_file():
        try:
            global_memory.load(seed_bank_path)
            global_prior_weight = 0.2 if len(global_memory) else 0.0
            if not len(global_memory):
                raise ValueError("seed bank is empty")
            log(
                f"[INFO] Loaded {len(global_memory)} seed programs from {seed_bank_path}"
            )
        except Exception as exc:
            if not args.allow_missing_seed_bank:
                raise RuntimeError(f"Required seed bank is invalid: {seed_bank_path}") from exc
            log(f"[WARN] Development run without usable seed bank: {exc}")
    else:
        if not args.allow_missing_seed_bank:
            raise FileNotFoundError(f"Required seed bank not found: {seed_bank_path}")
        log("[WARN] Development run without seed bank; using uniform search priors")

    submission: dict[str, Any] = {}
    completed_ids: set[str] = set()
    if checkpoint_path.is_file() and not args.no_resume:
        try:
            submission, completed_ids = load_checkpoint(checkpoint_path, global_memory)
            if len(global_memory):
                global_prior_weight = 0.2
            log(f"[INFO] Resuming with {len(completed_ids)} completed tasks")
        except Exception as exc:
            log(f"[WARN] Ignoring unreadable checkpoint {checkpoint_path}: {exc}")
            submission, completed_ids = {}, set()

    total = len(challenges)
    started = time.monotonic()
    failures: list[dict[str, str]] = []
    timed_out_tasks: list[str] = []
    demonstration_exact_tasks = 0
    for index, (task_id, task_data) in enumerate(challenges.items(), start=1):
        if task_id in completed_ids and task_id in submission:
            log(f"[SKIP] {task_id} ({index}/{total})")
            continue
        remaining = args.global_timeout - (time.monotonic() - started)
        if remaining <= 0:
            log(f"[TIMEOUT] Global solver deadline reached at task {task_id}")
            for pending_id, pending_task in list(challenges.items())[index - 1 :]:
                if pending_id not in submission:
                    submission[pending_id] = failure_fallback(pending_task)
                    timed_out_tasks.append(pending_id)
            break
        try:
            attempts, score, exact, program_len = solve_task(
                task_id,
                task_data,
                global_memory,
                global_prior_weight,
                min(args.task_timeout, remaining),
                args.max_iterations,
                experience_bank,
                world_model,
            )
            submission[task_id] = attempts
            demonstration_exact_tasks += int(exact)
            log(
                f"[TASK] {task_id} exact_match={exact} score={score:.4f} "
                f"program_len={program_len} ({index}/{total})"
            )
        except Exception as exc:
            log(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc()
            if args.task_failure_policy == "abort":
                raise
            failures.append({"task_id": task_id, "error": type(exc).__name__})
            submission[task_id] = failure_fallback(task_data)
        completed_ids.add(task_id)
        save_checkpoint(checkpoint_path, submission, completed_ids, global_memory)

    validate_submission(submission, challenges)
    write_submission(output_path, submission)
    identity_attempts = 0
    distinct_attempt_pairs = 0
    output_count = 0
    for task_id, task in challenges.items():
        for test_index, pair in enumerate(task.get("test", [])):
            entry = submission[task_id][test_index]
            output_count += 1
            identity_attempts += int(entry["attempt_1"] == pair["input"])
            identity_attempts += int(entry["attempt_2"] == pair["input"])
            distinct_attempt_pairs += int(entry["attempt_1"] != entry["attempt_2"])
    run_manifest = {
        "schema_version": 1,
        "solver": "Hyper-ARC",
        "result_type": "submission-structure-and-runtime-only",
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": sys.platform,
        },
        "configuration": {
            "global_timeout_seconds": args.global_timeout,
            "task_timeout_seconds": args.task_timeout,
            "max_iterations": args.max_iterations,
            "world_model_enabled": args.enable_world_model,
            "task_failure_policy": args.task_failure_policy,
        },
        "channels": {
            "seed_records": len(global_memory),
            "experience_records": len(experience_bank.records),
            "world_memory_records": world_memory_items,
        },
        "tasks": {
            "total": total,
            "outputs": output_count,
            "demonstration_exact": demonstration_exact_tasks,
            "failures": failures,
            "timed_out": timed_out_tasks,
        },
        "submission": {
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "identity_attempts": identity_attempts,
            "distinct_attempt_pairs": distinct_attempt_pairs,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_submission(run_manifest_path, run_manifest)
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()
    log(f"[SUCCESS] Saved validated submission to {output_path}")
    log(f"[SUCCESS] Saved run manifest to {run_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

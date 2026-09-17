"""Kaggle-compatible orchestration for the Hyper-ARC solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from hyper_arc.esb import ESB
from hyper_arc.contender.relational_plans import exact_relational_candidates
from hyper_arc.deterministic import exact_candidates
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MAX_ITERATIONS, MCTSEngine
from hyper_arc.verified_adapter import solve_verified


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
DEFAULT_SEED_BANK = Path("./hyper_arc/seed_bank.json")


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
    verified_predictions, fitting_rules = solve_verified(task_data)
    if fitting_rules or deterministic or relational:
        attempts = []
        for index, input_grid in enumerate(test_inputs):
            baseline_ranked = list(verified_predictions[index]) if fitting_rules else []
            baseline_ranked.extend(
                candidate.predict(input_grid) for candidate in deterministic
            )
            baseline = []
            for prediction in baseline_ranked:
                if _valid_grid(prediction) and prediction not in baseline:
                    baseline.append(prediction)
                if len(baseline) == 2:
                    break
            while len(baseline) < 2:
                baseline.append(input_grid)
            relational_predictions = [
                candidate.predict(input_grid) for candidate in relational
            ]
            ranked = [baseline[0], *relational_predictions, baseline[1]]
            unique = []
            for prediction in ranked:
                if _valid_grid(prediction) and prediction not in unique:
                    unique.append(prediction)
                if len(unique) == 2:
                    break
            while len(unique) < 2:
                unique.append(input_grid)
            attempts.append({"attempt_1": unique[0], "attempt_2": unique[1]})
        complexity = (
            relational[0].complexity
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
        except (ValueError, IndexError, RuntimeError):
            prediction = input_grid
        attempts.append({"attempt_1": prediction, "attempt_2": input_grid})
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
    parser.add_argument("--seed-bank", default=str(DEFAULT_SEED_BANK))
    parser.add_argument("--task-timeout", type=float, default=90.0)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    challenge_path = resolve_challenge_file(args.challenge_file, args.data_dir)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    seed_bank_path = Path(args.seed_bank)

    log(f"[INFO] Loading challenges from {challenge_path}")
    challenges = load_challenges(challenge_path)
    log(f"[INFO] Found {len(challenges)} tasks")

    global_memory = GlobalMemoryBank(k=5)
    global_prior_weight = 0.0
    if seed_bank_path.is_file():
        try:
            global_memory.load_seed_bank(seed_bank_path)
            global_prior_weight = 0.2 if len(global_memory) else 0.0
            log(
                f"[INFO] Loaded {len(global_memory)} seed programs from {seed_bank_path}"
            )
        except Exception as exc:
            log(f"[WARN] Could not load seed bank {seed_bank_path}: {exc}")
    else:
        log("[WARN] No seed bank found; using uniform search priors")

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
    for index, (task_id, task_data) in enumerate(challenges.items(), start=1):
        if task_id in completed_ids and task_id in submission:
            log(f"[SKIP] {task_id} ({index}/{total})")
            continue
        try:
            attempts, score, exact, program_len = solve_task(
                task_id,
                task_data,
                global_memory,
                global_prior_weight,
                args.task_timeout,
                args.max_iterations,
            )
            submission[task_id] = attempts
            log(
                f"[TASK] {task_id} exact_match={exact} score={score:.4f} "
                f"program_len={program_len} ({index}/{total})"
            )
        except Exception as exc:
            log(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc()
            submission[task_id] = [
                {"attempt_1": pair["input"], "attempt_2": pair["input"]}
                for pair in task_data.get("test", [])
            ]
        completed_ids.add(task_id)
        save_checkpoint(checkpoint_path, submission, completed_ids, global_memory)

    validate_submission(submission, challenges)
    write_submission(output_path, submission)
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()
    log(f"[SUCCESS] Saved validated submission to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

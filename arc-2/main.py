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
from hyper_arc.contender.agentic_reasoner import AgenticReasoner, validate_code
from hyper_arc.contender.agentic_memory import AgenticMemoryBank
from hyper_arc.contender.session_memory import SessionMemory
from release_policy import (
    RELEASE_ID, MODEL_ID, MODEL_MIN_GPU_MEMORY_GIB, MODEL_MIN_GPUS, TASK_SECONDS, RUN_SECONDS, AGENTIC_ROUNDS,
    AGENTIC_CANDIDATES, OUTPUT_TOKENS, model_preflight, qualify_run,
)
from hyper_arc.contender.hyperbolic_memory import HyperbolicWorldMemory
from hyper_arc.contender.failure_memory import (
    FailureMemoryBank,
    extract_failed_families,
)
from hyper_arc.contender.offline_model import OfflineTransformersTransport
from hyper_arc.contender.procedural_memory import ProceduralMemoryBank
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
DEFAULT_PROCEDURAL_MEMORY = Path("./hyper_arc/procedural_memory_v1.json")
DEFAULT_AGENTIC_MEMORY = Path("./hyper_arc/agentic_memory_builder_v2.json")
DEFAULT_FAILURE_MEMORY = Path("./hyper_arc/failure_memory_builder_v2.json")


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


def _zero_grid(grid: list[list[int]]) -> list[list[int]]:
    """A same-shape all-zero (ARC background) grid."""
    if not grid or not grid[0]:
        # Degenerate input; emit a minimal valid 1x1 grid.
        return [[0]]
    return [[0] * len(grid[0]) for _ in grid]


def failure_fallback(
    task_data: dict[str, Any], mode: str = "input"
) -> list[dict[str, list[list[int]]]]:
    """Declared last-resort policy used only after a recorded task failure.

    mode="input" (default): copy the test input. On ARC this is a strictly
        stronger blind guess than zeros (many tasks are near-identity) and is
        always shape-valid.
    mode="zero": emit a same-shape all-zero (background) grid. Provided for
        explicit request only; expected to score no better and often worse.
    """
    filler = _zero_grid if mode == "zero" else _copy_grid
    return [
        {
            "attempt_1": filler(pair["input"]),
            "attempt_2": filler(pair["input"]),
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
    run_signature: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "submission": submission,
            "completed_ids": sorted(completed_ids),
            "global_memory_state": global_memory.state_dict_entries(),
            "run_signature": run_signature,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    global_memory: GlobalMemoryBank,
    *,
    expected_run_signature: str | None = None,
    restore_memory: bool = True,
) -> tuple[dict[str, Any], set[str]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older Kaggle torch images
        payload = torch.load(path, map_location="cpu")
    if (
        expected_run_signature is not None
        and payload.get("run_signature") != expected_run_signature
    ):
        raise ValueError("Checkpoint does not match this challenge/configuration")
    if restore_memory:
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
    procedural_memory: ProceduralMemoryBank | None = None,
    agentic_memory: AgenticMemoryBank | None = None,
    agentic_reasoner: AgenticReasoner | None = None,
    failure_memory: "FailureMemoryBank | None" = None,
    learned_failures: list | None = None,
    session_memory: SessionMemory | None = None,
) -> tuple[list[dict[str, list[list[int]]]], float, bool, int]:
    deadline = time.monotonic() + timeout_sec
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
    if procedural_memory is not None:
        deterministic = procedural_memory.rank(deterministic, task_data)
    relational = exact_relational_candidates(task_data)
    memory = (
        experience_candidates(experience_bank, task_data)
        if experience_bank is not None
        else []
    )
    # Induction and verifier-guided iteration get the first opportunity on
    # unresolved tasks. Recursive repair is a conditional, bounded fallback.
    # Exact-fit baselines remain cheap and available without a neural call.
    # Retain the historical second-attempt world candidate on that path so
    # changing the unresolved-task loop cannot silently erase its upside.
    if deterministic or relational or memory:
        baseline_world = (
            world_model.solve(task_id, task_data, deadline=deadline)
            if world_model is not None and time.monotonic() < deadline
            else None
        )
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
                    for prediction in baseline_world.test_predictions[index]
                ]
                if baseline_world is not None
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
            baseline_world.hypotheses[0].complexity
            if baseline_world is not None and baseline_world.hypotheses
            else memory[0].complexity
            if memory
            else relational[0].complexity
            if relational
            else deterministic[0].complexity
            if deterministic
            else 0
        )
        return attempts, 1.0, True, complexity

    if session_memory is not None:
        predictions = session_memory.predict(task_data, deadline)
        if predictions:
            return [
                _two_attempts([p[i] for p in predictions], grid)
                for i, grid in enumerate(test_inputs)
            ], 1.0, True, 0
    recalled = agentic_memory.exact_candidates(task_data, deadline=deadline) if agentic_memory else ()
    if recalled:
        attempts = []
        for index, input_grid in enumerate(test_inputs):
            ranked = [
                [list(row) for row in candidate.test_predictions[index]]
                for candidate in recalled[:2]
            ]
            attempts.append(_two_attempts(ranked, input_grid))
        return attempts, 1.0, True, validate_code(recalled[0].record.code)

    if agentic_reasoner is not None:
        # A stalled model must not consume the entire task budget. Keep time
        # for recursive repair and a schema-valid final fallback.
        agentic_deadline = deadline - max(0.2, timeout_sec * 0.25)
        cues = (
            procedural_memory.retrieve_cues(task_data)
            if procedural_memory is not None
            else ()
        )
        # Recall approach families that already failed on structurally similar
        # tasks so the reasoner steers away from known dead ends (learn-from-
        # failure). Previously wired only in the eval harness, not production.
        failure_cues = (
            failure_memory.recall(task_data) if failure_memory is not None else ()
        )
        agentic = agentic_reasoner.solve(
            task_id, task_data, memory_cues=cues, failure_cues=failure_cues, deadline=agentic_deadline
        )
        if learned_failures is not None:
            learned_failures.extend(extract_failed_families(task_id, task_data, agentic.failures))
        if agentic.hypotheses:
            if session_memory is not None:
                session_memory.remember(agentic.hypotheses)
            attempts = []
            for index, input_grid in enumerate(test_inputs):
                ranked = [
                    [list(row) for row in prediction]
                    for prediction in agentic.test_predictions[index]
                ]
                attempts.append(_two_attempts(ranked, input_grid))
            return attempts, 1.0, True, agentic.hypotheses[0].complexity

    # Recursive symbolic composition is useful when induction/iteration did
    # not produce an exact demonstration fit. It must never preempt that loop.
    world_result = world_model.solve(task_id, task_data, deadline=deadline) if world_model and time.monotonic() < deadline else None
    if world_result is not None and world_result.hypotheses:
        attempts = []
        for index, input_grid in enumerate(test_inputs):
            ranked = [
                [list(row) for row in prediction]
                for prediction in world_result.test_predictions[index]
            ]
            attempts.append(_two_attempts(ranked, input_grid))
        return attempts, 1.0, True, world_result.hypotheses[0].complexity

    if time.monotonic() >= deadline:
        raise TimeoutError("Task reasoning budget exhausted")

    engine = MCTSEngine(
        global_memory=global_memory,
        local_memory=LocalTaskBuffer(),
        C=1.41,
        global_prior_weight=global_prior_weight,
        max_iterations=max_iterations,
        random_seed=stable_task_seed(task_id),
    )
    program = engine.solve(train_pairs, timeout_sec=max(0.001, deadline - time.monotonic()))
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
    parser.add_argument(
        "--procedural-memory", default=str(DEFAULT_PROCEDURAL_MEMORY)
    )
    parser.add_argument("--task-timeout", type=float, default=TASK_SECONDS)
    parser.add_argument(
        "--timeout-fallback",
        choices=("input", "zero"),
        default="input",
        help=(
            "Grid submitted when a task fails or times out. 'input' (default) "
            "copies the test input; 'zero' emits a same-shape blank grid."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--global-timeout", type=float, default=RUN_SECONDS)
    parser.add_argument("--enable-world-model", action="store_true")
    parser.add_argument("--enable-world-memory", action="store_true")
    parser.add_argument("--enable-experience-memory", action="store_true")
    parser.add_argument("--enable-procedural-memory", action="store_true")
    parser.add_argument("--enable-legacy-seed-memory", action="store_true")
    parser.add_argument("--enable-agentic-ai", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--agentic-model-name", default=MODEL_ID)
    parser.add_argument("--agentic-memory", default=str(DEFAULT_AGENTIC_MEMORY))
    parser.add_argument("--enable-agentic-memory", action="store_true")
    parser.add_argument("--failure-memory", default=str(DEFAULT_FAILURE_MEMORY))
    parser.add_argument(
        "--enable-failure-memory",
        action="store_true",
        help="Recall tried-and-failed approach families to steer the agentic "
        "reasoner away from known dead ends, and persist new ones after each task.",
    )
    parser.add_argument(
        "--agentic-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--agentic-rounds", type=int, default=AGENTIC_ROUNDS)
    parser.add_argument("--agentic-candidates", type=int, default=AGENTIC_CANDIDATES)
    parser.add_argument("--agentic-max-output-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument("--agentic-thinking", action="store_true")
    parser.add_argument("--competition-run", action="store_true")
    parser.add_argument("--session-transfer", action="store_true",
                        help="Development only: reuse demo-supported procedures and failures within this run")
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
    started = time.monotonic()
    if args.task_timeout <= 0 or args.global_timeout <= 0:
        raise ValueError("Runtime budgets must be positive")
    if args.competition_run and args.session_transfer:
        raise ValueError("Cross-task competition transfer requires verified rules; use task-local memory")
    if args.session_transfer and not args.no_resume:
        raise ValueError("Session transfer requires --no-resume; ephemeral evidence is not a checkpoint")
    if args.competition_run and not args.enable_agentic_ai:
        raise ValueError("Competition runs require --enable-agentic-ai")
    if args.competition_run and not args.enable_agentic_memory:
        raise ValueError("Competition runs require --enable-agentic-memory")
    if args.competition_run and not args.no_resume:
        raise ValueError("Competition qualification requires a clean --no-resume run")
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
    procedural_memory_path = Path(args.procedural_memory)
    agentic_memory_path = Path(args.agentic_memory)

    log(f"[INFO] Loading challenges from {challenge_path}")
    challenges = load_challenges(challenge_path)
    log(f"[INFO] Found {len(challenges)} tasks")

    experience_bank = None
    experience_records = 0
    if args.enable_experience_memory:
        if not experience_bank_path.is_file():
            raise FileNotFoundError(
                f"Enabled experience bank not found: {experience_bank_path}"
            )
        experience_bank = ExperienceBank.load(experience_bank_path)
        experience_records = len(experience_bank.records)
        if not experience_records:
            raise ValueError("Enabled experience bank is empty")
        log(
            f"[INFO] Enabled experience bank {experience_bank.bank_id[:12]} "
            f"with {experience_records} records"
        )
    else:
        log(
            "[GATE] Experience memory inactive (expected): exact-score ablation "
            "added zero owned solves"
        )
    procedural_memory = None
    procedural_records = 0
    if args.enable_procedural_memory:
        if not procedural_memory_path.is_file():
            raise FileNotFoundError(
                f"Enabled procedural memory not found: {procedural_memory_path}"
            )
        procedural_memory = ProceduralMemoryBank.load(procedural_memory_path)
        procedural_records = len(procedural_memory.records)
        if not procedural_records:
            raise ValueError("Enabled procedural memory is empty")
        log(
            f"[INFO] Enabled procedural memory {procedural_memory.bank_id[:12]} "
            f"with {procedural_records} exact-replay chains"
        )
    else:
        log("[INFO] Procedural memory disabled")
    world_model = None
    world_memory_items = 0
    if args.enable_world_model:
        if args.enable_world_memory:
            if not world_memory_path.is_file():
                raise FileNotFoundError(
                    f"Enabled world memory not found: {world_memory_path}"
                )
            world_memory = HyperbolicWorldMemory.load(world_memory_path)
            world_memory_items = len(world_memory.items)
            if not world_memory_items:
                raise ValueError("Enabled world memory is empty")
        else:
            world_memory = HyperbolicWorldMemory()
        world_model = RecursiveReasoningWorldModel(
            world_memory,
            config=WorldModelConfig(
                retrieve_memory=args.enable_world_memory,
                learn_memory=False,
            ),
        )
        log("[INFO] Enabled promoted recursive world model")
        if args.enable_world_memory:
            log(f"[INFO] Enabled {world_memory_items} executable world memories")
        else:
            log(
                "[GATE] World-memory retrieval inactive (expected): exact-score "
                "ablation added zero solves"
            )
    elif args.enable_world_memory:
        raise ValueError("--enable-world-memory requires --enable-world-model")
    else:
        log("[GATE] Recursive world model inactive: not selected for this run")

    agentic_memory = None
    agentic_memory_records = 0
    if args.enable_agentic_memory:
        if not agentic_memory_path.is_file():
            raise FileNotFoundError(
                f"Enabled agentic memory not found: {agentic_memory_path}"
            )
        agentic_memory = AgenticMemoryBank.load(agentic_memory_path)
        agentic_memory_records = len(agentic_memory.records)
        if any(record.validation_scope != "builder" for record in agentic_memory.records):
            raise ValueError("Active static agentic memory must contain builder-only evidence")
        log(
            f"[INFO] Enabled exact model-authored memory with "
            f"{agentic_memory_records} records"
        )
    else:
        log("[GATE] Model-authored memory inactive")

    agentic_reasoner = None
    preflight = {"passed": False}
    if args.enable_agentic_ai:
        if not args.model_path:
            raise ValueError("--enable-agentic-ai requires --model-path")
        transport = OfflineTransformersTransport(
            args.model_path,
            device=args.agentic_device,
            minimum_gpu_memory_gib=MODEL_MIN_GPU_MEMORY_GIB if args.agentic_model_name == MODEL_ID else 0.0,
            minimum_gpu_count=MODEL_MIN_GPUS if args.agentic_model_name == MODEL_ID else 1,
        )
        preflight = model_preflight(transport, args.agentic_model_name,
                                   seconds=min(60.0, args.global_timeout - (time.monotonic() - started)))
        agentic_reasoner = AgenticReasoner(
            transport,
            model=args.agentic_model_name,
            rounds=args.agentic_rounds,
            candidates_per_round=args.agentic_candidates,
            max_output_tokens=args.agentic_max_output_tokens,
            thinking=args.agentic_thinking,
        )
        log(
            f"[INFO] Enabled offline agentic AI {args.agentic_model_name} "
            f"on {transport.device} from {transport.path}"
        )
    else:
        log("[GATE] Agentic AI inactive; this configuration is not submission-eligible")

    failure_memory = None
    failure_memory_path = Path(args.failure_memory)
    if args.enable_failure_memory:
        failure_memory = (
            FailureMemoryBank.load(failure_memory_path)
            if failure_memory_path.is_file()
            else FailureMemoryBank()
        )
        log(
            f"[INFO] Failure memory enabled with "
            f"{len(failure_memory.records)} learned dead-end families"
        )
    else:
        log("[GATE] Failure memory inactive")

    global_memory = GlobalMemoryBank(k=5)
    global_prior_weight = 0.0
    if args.enable_legacy_seed_memory and seed_bank_path.is_file():
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
    elif args.enable_legacy_seed_memory:
        if not args.allow_missing_seed_bank:
            raise FileNotFoundError(f"Required seed bank not found: {seed_bank_path}")
        log("[WARN] Development run without seed bank; using uniform search priors")
    else:
        log(
            "[GATE] Legacy seed memory inactive (expected): exact-score ablation "
            "added zero solves"
        )

    submission: dict[str, Any] = {}
    completed_ids: set[str] = set()
    signature_payload = {
        "release_id": RELEASE_ID,
        "challenge_sha256": hashlib.sha256(challenge_path.read_bytes()).hexdigest(),
        "max_iterations": args.max_iterations,
        "task_timeout": args.task_timeout,
        "world_model_enabled": args.enable_world_model,
        "world_memory_enabled": args.enable_world_memory,
        "experience_memory_enabled": args.enable_experience_memory,
        "procedural_memory_enabled": args.enable_procedural_memory,
        "legacy_seed_memory_enabled": args.enable_legacy_seed_memory,
        "agentic_ai_enabled": args.enable_agentic_ai,
        "agentic_model_name": args.agentic_model_name if args.enable_agentic_ai else None,
        "agentic_model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
        "agentic_rounds": args.agentic_rounds if args.enable_agentic_ai else 0,
        "agentic_candidates": args.agentic_candidates if args.enable_agentic_ai else 0,
        "agentic_thinking": args.agentic_thinking if args.enable_agentic_ai else False,
        "agentic_memory_enabled": args.enable_agentic_memory,
        "agentic_memory_sha256": (
            hashlib.sha256(agentic_memory_path.read_bytes()).hexdigest()
            if args.enable_agentic_memory
            else None
        ),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if checkpoint_path.is_file() and not args.no_resume:
        try:
            submission, completed_ids = load_checkpoint(
                checkpoint_path,
                global_memory,
                expected_run_signature=run_signature,
                restore_memory=args.enable_legacy_seed_memory,
            )
            if args.enable_legacy_seed_memory and len(global_memory):
                global_prior_weight = 0.2
            log(f"[INFO] Resuming with {len(completed_ids)} completed tasks")
        except Exception as exc:
            log(f"[WARN] Ignoring unreadable checkpoint {checkpoint_path}: {exc}")
            submission, completed_ids = {}, set()

    total = len(challenges)
    failures: list[dict[str, str]] = []
    timed_out_tasks: list[str] = []
    demonstration_exact_tasks = 0
    learned_failures: list = []
    session_memory = SessionMemory() if args.session_transfer else None
    for index, (task_id, task_data) in enumerate(challenges.items(), start=1):
        if task_id in completed_ids and task_id in submission:
            log(f"[SKIP] {task_id} ({index}/{total})")
            continue
        remaining = args.global_timeout - (time.monotonic() - started)
        if remaining <= 0:
            log(f"[TIMEOUT] Global solver deadline reached at task {task_id}")
            for pending_id, pending_task in list(challenges.items())[index - 1 :]:
                if pending_id not in submission:
                    submission[pending_id] = failure_fallback(
                        pending_task, args.timeout_fallback
                    )
                    timed_out_tasks.append(pending_id)
            break
        try:
            task_started = time.monotonic()
            task_budget = min(args.task_timeout, remaining)
            attempts, score, exact, program_len = solve_task(
                task_id,
                task_data,
                global_memory,
                global_prior_weight,
                task_budget,
                args.max_iterations,
                experience_bank,
                world_model,
                procedural_memory,
                agentic_memory,
                agentic_reasoner,
                failure_memory,
                learned_failures,
                session_memory,
            )
            if time.monotonic() - task_started > task_budget:
                raise TimeoutError("Task exceeded its end-to-end budget")
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
            if isinstance(exc, TimeoutError):
                timed_out_tasks.append(task_id)
            submission[task_id] = failure_fallback(task_data, args.timeout_fallback)
        if args.session_transfer and failure_memory is not None:
            failure_memory = failure_memory.merged(learned_failures)
        learned_failures.clear()
        completed_ids.add(task_id)
        save_checkpoint(
            checkpoint_path,
            submission,
            completed_ids,
            global_memory,
            run_signature=run_signature,
        )

    validate_submission(submission, challenges)
    write_submission(output_path, submission)

    # Static inputs are immutable. Session learning is ephemeral; exporting it
    # requires a separate builder-only evaluation, never hidden test guesses.
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
        "result_type": (
            "agentic-competition-submission"
            if args.competition_run
            else "development-submission-artifact"
        ),
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
            "release_id": RELEASE_ID,
            "session_transfer": args.session_transfer,
            "run_signature": run_signature,
            "global_timeout_seconds": args.global_timeout,
            "task_timeout_seconds": args.task_timeout,
            "timeout_fallback": args.timeout_fallback,
            "max_iterations": args.max_iterations,
            "world_model_enabled": args.enable_world_model,
            "world_memory_enabled": args.enable_world_memory,
            "experience_memory_enabled": args.enable_experience_memory,
            "procedural_memory_enabled": args.enable_procedural_memory,
            "legacy_seed_memory_enabled": args.enable_legacy_seed_memory,
            "task_failure_policy": args.task_failure_policy,
            "agentic_ai_enabled": args.enable_agentic_ai,
            "agentic_model_name": args.agentic_model_name if args.enable_agentic_ai else None,
            "agentic_model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
            "agentic_device": (
                agentic_reasoner.transport.device
                if agentic_reasoner is not None
                and hasattr(agentic_reasoner.transport, "device")
                else None
            ),
            "agentic_rounds": args.agentic_rounds if args.enable_agentic_ai else 0,
            "agentic_candidates": (
                args.agentic_candidates if args.enable_agentic_ai else 0
            ),
            "agentic_max_output_tokens": (
                args.agentic_max_output_tokens if args.enable_agentic_ai else 0
            ),
            "agentic_thinking": (
                args.agentic_thinking if args.enable_agentic_ai else False
            ),
            "agentic_memory_enabled": args.enable_agentic_memory,
            "competition_run": args.competition_run,
        },
        "channels": {
            "agentic_unusable_invocations": agentic_reasoner.unusable_invocations if agentic_reasoner else 0,
            "session_procedures": len(session_memory.records) if session_memory else 0,
            "session_memory_hits": session_memory.hits if session_memory else 0,
            "seed_records": len(global_memory),
            "experience_records": experience_records,
            "procedural_records": procedural_records,
            "world_memory_records": world_memory_items,
            "agentic_memory_records": agentic_memory_records,
            "agentic_invocations": (
                agentic_reasoner.invocations if agentic_reasoner is not None else 0
            ),
            "agentic_demo_exact": (
                agentic_reasoner.exact_invocations
                if agentic_reasoner is not None
                else 0
            ),
            "agentic_candidates_generated": (
                agentic_reasoner.generated_total if agentic_reasoner is not None else 0
            ),
            "agentic_candidates_executed": (
                agentic_reasoner.executed_total if agentic_reasoner is not None else 0
            ),
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
        "model_preflight": preflight,
    }
    write_submission(run_manifest_path, run_manifest)
    if args.competition_run:
        try:
            qualify_run(run_manifest)
        except RuntimeError:
            output_path.unlink(missing_ok=True)
            raise
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()
    log(f"[SUCCESS] Saved validated submission to {output_path}")
    log(f"[SUCCESS] Saved run manifest to {run_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

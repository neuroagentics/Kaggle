"""Score a complete ARC-AGI-2 submission and optionally append a run ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_arc.contender.evaluation import (
    append_ledger,
    build_ledger_entry,
    evaluate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--solver-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--channel", action="append", default=[])
    parser.add_argument("--channel-attribution")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--peak-memory-bytes", type=int)
    parser.add_argument(
        "--telemetry",
        help="Optional ResourceTelemetry JSON produced by a contender runner.",
    )
    parser.add_argument(
        "--task-families",
        help="Optional JSON object mapping every scored task ID to a family label.",
    )
    parser.add_argument("--report")
    parser.add_argument("--ledger")
    args = parser.parse_args()

    submission = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    channel_attribution = (
        json.loads(Path(args.channel_attribution).read_text(encoding="utf-8"))
        if args.channel_attribution
        else None
    )
    telemetry = (
        json.loads(Path(args.telemetry).read_text(encoding="utf-8"))
        if args.telemetry
        else None
    )
    task_families = (
        json.loads(Path(args.task_families).read_text(encoding="utf-8"))
        if args.task_families
        else None
    )
    report = evaluate_submission(submission, solutions, channel_attribution)
    payload = report.to_dict()
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "tasks"}, indent=2
        )
    )

    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.ledger:
        append_ledger(
            args.ledger,
            build_ledger_entry(
                report,
                run_id=args.run_id,
                solver_id=args.solver_id,
                dataset_id=args.dataset_id,
                channels=args.channel,
                duration_seconds=args.duration_seconds,
                peak_memory_bytes=args.peak_memory_bytes,
                telemetry=telemetry,
                task_families=task_families,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

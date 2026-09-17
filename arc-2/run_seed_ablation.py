"""Mine seed-bank v1 on development data and evaluate the sealed holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_arc.contender.data_protocol import SplitManifest, load_verified_split
from hyper_arc.contender.seed_bank import mine_seed_bank, run_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenges",
        default="data/arc-agi-2/arc-agi_training_challenges.json",
    )
    parser.add_argument(
        "--solutions",
        default="data/arc-agi-2/arc-agi_training_solutions.json",
    )
    parser.add_argument("--manifest", default="config/arc2_split_v1.json")
    parser.add_argument("--retrieval-limit", type=int, default=8)
    parser.add_argument("--seed-bank")
    parser.add_argument("--report")
    args = parser.parse_args()

    manifest = SplitManifest.load(args.manifest)
    development, holdout = load_verified_split(args.challenges, args.manifest)
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    development_solutions = {task_id: solutions[task_id] for task_id in development}
    holdout_solutions = {task_id: solutions[task_id] for task_id in holdout}
    bank = mine_seed_bank(
        development,
        development_solutions,
        split_sha256=manifest.split_sha256,
        training_dataset_sha256=manifest.dataset_sha256,
    )
    report = run_ablation(
        bank,
        holdout,
        holdout_solutions,
        retrieval_limit=args.retrieval_limit,
    )
    summary = {
        "bank_id": bank.bank_id,
        "schemas": len(bank.schemas),
        "development_evidence": sum(
            schema.evidence_count for schema in bank.schemas
        ),
        **report,
    }
    print(json.dumps(summary, indent=2))
    if args.seed_bank:
        bank.save(args.seed_bank)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build one deterministic, self-describing Hyper-ARC Kaggle bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable
from release_policy import RELEASE_ID, MODEL_ATTACHMENT


ROOT = Path(__file__).resolve().parent
BUNDLE_PREFIX = "hyper_arc_solver_bundle"
MANIFEST_NAME = "bundle_manifest.json"

RUNTIME_FILES = (
    "release_policy.py",
    "main.py",
    "bundle_self_test.py",
    "agentic_sandbox.py",
    "agentic_primitives.py",
    "hyper_arc/__init__.py",
    "hyper_arc/cost.py",
    "hyper_arc/deterministic.py",
    "hyper_arc/esb.py",
    "hyper_arc/hpm.py",
    "hyper_arc/mcts.py",
    "hyper_arc/spatial_dsl.py",
    "hyper_arc/contender/__init__.py",
    "hyper_arc/contender/schemas.py",
    "hyper_arc/contender/structural_rules.py",
    "hyper_arc/contender/perception.py",
    "hyper_arc/contender/procedural_memory.py",
    "hyper_arc/contender/executor.py",
    "hyper_arc/contender/experience_bank.py",
    "hyper_arc/contender/hyperbolic_memory.py",
    "hyper_arc/contender/object_programs.py",
    "hyper_arc/contender/agentic_reasoner.py",
    "hyper_arc/contender/agentic_memory.py",
    "hyper_arc/contender/agentic_worker.py",
    "hyper_arc/contender/offline_model.py",
    "hyper_arc/contender/relational_primitives.py",
    "hyper_arc/contender/relational_plans.py",
    "hyper_arc/contender/repair.py",
    "hyper_arc/contender/world_model.py",
    "hyper_arc/contender/failure_memory.py",
    "hyper_arc/contender/session_memory.py",
    "hyper_arc/procedural_memory_v1.json",
    "hyper_arc/agentic_memory_builder_v2.json",
    "hyper_arc/failure_memory_builder_v2.json",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _read_runtime_files(paths: Iterable[str]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    missing = []
    for relative in sorted(paths):
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        # Stable across Windows/Linux checkouts; runtime assets are text only.
        payloads[relative] = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    if missing:
        raise FileNotFoundError(f"Required bundle files are missing: {missing}")
    return payloads


def _item_count(payload: bytes, key: str | None = None) -> int:
    value = json.loads(payload.decode("utf-8"))
    if key is not None:
        value = value[key]
    if not isinstance(value, list):
        raise ValueError(f"Expected a list while counting {key or 'asset'}")
    return len(value)


def build_manifest(payloads: dict[str, bytes]) -> dict:
    files = {
        name: {"sha256": _sha256(payload), "size": len(payload)}
        for name, payload in sorted(payloads.items())
    }
    source_tree_digest = _sha256(
        "".join(f"{name}:{entry['sha256']}\n" for name, entry in files.items()).encode()
    )
    return {
        "schema_version": 1,
        "project": "Hyper-ARC",
        "target": "ARC Prize 2026 ARC-AGI-2 Kaggle runtime",
        "source_revision": _source_revision(),
        "release_id": RELEASE_ID,
        "source_tree_sha256": source_tree_digest,
        "files": files,
        "runtime": {
            "network_required": False,
            "dependencies": [
                {"name": "python", "version": ">=3.10", "bundled": False},
                {"name": "numpy", "version": ">=1.26,<3", "bundled": False},
                {"name": "torch", "version": ">=2.3,<3", "bundled": False},
            ],
        },
        "channels": {
            "agentic_ai": {
                "required": True,
                "enabled": True,
                "model": MODEL_ATTACHMENT,
                "reason": (
                    "Offline neural perception, hypothesis planning, program "
                    "synthesis, verifier feedback, and reflection controller"
                ),
            },
            "agentic_memory": {
                "required": True,
                "enabled": True,
                "records": _item_count(
                    payloads["hyper_arc/agentic_memory_builder_v2.json"], "records"
                ),
                "reason": (
                    "Only model-authored procedures with exact demonstration and "
                    "held-out training-output verification; replay is re-gated on "
                    "every current task"
                ),
            },
            "deterministic": {"required": True, "enabled": True},
            "legacy_seed_memory": {
                "required": False,
                "enabled": False,
                "packaged": False,
                "reason": "Not promoted: 0/120 exact pass@2 with no unique solves",
            },
            "experience_memory": {
                "required": False,
                "enabled": False,
                "packaged": False,
                "reason": "Not promoted: clean evaluation added zero owned exact solves",
            },
            "procedural_memory": {
                "required": True,
                "enabled": True,
                "records": _item_count(
                    payloads["hyper_arc/procedural_memory_v1.json"], "records"
                ),
                "reason": (
                    "Builder-only exact successes and failures rank only hypotheses "
                    "that already replay the current demonstrations exactly"
                ),
            },
            "recursive_world_model": {
                "required": False,
                "enabled": True,
                "reason": (
                    "Promoted: combined exact pass@2 is 12/200 versus the 8/200 "
                    "owned deterministic baseline on the frozen holdout; the "
                    "labeled public-evaluation partition remains 0/120"
                ),
            },
            "world_memory": {
                "required": False,
                "enabled": False,
                "packaged": False,
                "reason": "Not promoted: memory retrieval added zero evaluation solves",
            },
            "external_symbolic": {
                "required": False,
                "enabled": False,
                "reason": "Reference-only external implementation is excluded",
            },
        },
        "release": {
            "public_license": "PENDING",
            "competition_release_ready": False,
            "reason": "Requires binding rules, owner-selected license, live offline model qualification, independent exact-score improvement and two clean rehearsals",
        },
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(output_dir: Path) -> Path:
    payloads = _read_runtime_files(RUNTIME_FILES)
    manifest = build_manifest(payloads)
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f"{BUNDLE_PREFIX}.tmp.zip"
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w") as archive:
        archive.writestr(_zip_info(MANIFEST_NAME), manifest_payload)
        for name, payload in sorted(payloads.items()):
            archive.writestr(_zip_info(name), payload)

    bundle_digest = _sha256(temporary.read_bytes())
    destination = output_dir / f"{BUNDLE_PREFIX}.{bundle_digest}.zip"
    if destination.exists():
        destination.unlink()
    temporary.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    destination = build_bundle(arguments.output_dir.resolve())
    print(
        f"built={destination} bytes={destination.stat().st_size} "
        f"sha256={destination.stem.rsplit('.', 1)[-1]} ownership=hyper-arc-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

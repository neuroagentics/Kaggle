"""Generate one hash-linked notebook/bundle/metadata release; never upload."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from build_solver_archive import ROOT, build_bundle
from release_policy import (
    RELEASE_ID, MODEL_ID, MODEL_ATTACHMENT, TASK_SECONDS, RUN_SECONDS,
    AGENTIC_ROUNDS, AGENTIC_CANDIDATES, OUTPUT_TOKENS,
)


def build_release(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(output_dir)
    # Keep historical bundles outside the upload directory; never mix versions.
    for old in output_dir.glob("hyper_arc_solver_bundle.*.zip"):
        if old != bundle:
            digest = hashlib.sha256(old.read_bytes()).hexdigest()
            if old.name != f"hyper_arc_solver_bundle.{digest}.zip":
                raise ValueError(f"Unrecognized artifact; refusing to move {old}")
            history = output_dir.parent / "history" / digest
            history.mkdir(parents=True, exist_ok=True)
            destination = history / old.name
            if destination.exists():
                raise ValueError(f"Historical destination already exists: {destination}")
            old.rename(destination)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
    identity = {"id": RELEASE_ID, "model": MODEL_ID,
                "rounds": AGENTIC_ROUNDS, "candidates": AGENTIC_CANDIDATES, "output_tokens": OUTPUT_TOKENS,
                "task_seconds": TASK_SECONDS, "run_seconds": RUN_SECONDS,
                "source_tree_sha256": manifest["source_tree_sha256"]}
    source = (ROOT / "kaggle_launcher.py").read_text(encoding="utf-8")
    source = source.replace("RELEASE = None", f"RELEASE = {identity!r}")
    compile(source, "generated-kaggle-launcher", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "code", "id": "hyper-arc-launcher", "metadata": {},
                           "execution_count": None, "outputs": [], "source": source.splitlines(keepends=True)}]}
    notebook_path = output_dir / "notebook6a96ea823f.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8", newline="\n")
    metadata = {"id": "jthomaslockhart/notebook6a96ea823f", "title": "Hyper-ARC recursive agent",
                "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": False,
                "dataset_sources": ["jthomaslockhart/arc-solver-code", "jthomaslockhart/hyper-arc-gemma4-runtime"],
                "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": [MODEL_ATTACHMENT]}
    metadata["machine_shape"] = "NvidiaL4"
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    record = {**identity, "source_revision": manifest["source_revision"], "bundle": bundle.name,
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "status": "UNQUALIFIED_CANDIDATE", "submission_authorized": False}
    (output_dir / "release.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release" / "current")
    args = parser.parse_args()
    print(json.dumps(build_release(args.output_dir), indent=2))

"""Hermetic Kaggle bundle regression tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile

from build_solver_archive import BUNDLE_PREFIX, MANIFEST_NAME, build_bundle


def test_bundle_is_named_by_digest_and_matches_internal_manifest(tmp_path):
    bundle = build_bundle(tmp_path)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert bundle.name == f"{BUNDLE_PREFIX}.{digest}.zip"

    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
        assert set(archive.namelist()) == set(manifest["files"]) | {MANIFEST_NAME}
        assert manifest["channels"]["legacy_seed_memory"]["enabled"] is False
        assert manifest["channels"]["legacy_seed_memory"]["packaged"] is False
        assert manifest["channels"]["experience_memory"]["enabled"] is False
        assert manifest["channels"]["experience_memory"]["packaged"] is False
        assert manifest["channels"]["procedural_memory"]["enabled"] is True
        assert manifest["channels"]["procedural_memory"]["records"] > 0
        assert manifest["channels"]["agentic_ai"]["enabled"] is True
        assert manifest["channels"]["agentic_ai"]["required"] is True
        assert manifest["channels"]["agentic_memory"]["records"] == 0
        assert "hyper_arc/agentic_memory_v1.json" not in archive.namelist()
        assert manifest["channels"]["recursive_world_model"]["enabled"] is True
        assert manifest["channels"]["world_memory"]["enabled"] is False
        assert manifest["channels"]["world_memory"]["packaged"] is False
        assert manifest["channels"]["external_symbolic"]["enabled"] is False
        assert manifest["release"]["competition_release_ready"] is False
        assert "hyper_arc/verified_adapter.py" not in archive.namelist()
        assert "hyper_arc/seed_bank.json" not in archive.namelist()
        assert "hyper_arc/experience_bank_v2.json" not in archive.namelist()
        assert "hyper_arc/world_memory_v1.json" not in archive.namelist()
        for name, metadata in manifest["files"].items():
            payload = archive.read(name)
            assert len(payload) == metadata["size"]
            assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]


def test_bundle_build_is_reproducible(tmp_path):
    first = build_bundle(tmp_path)
    first_payload = first.read_bytes()
    second = build_bundle(tmp_path)
    assert second == first
    assert second.read_bytes() == first_payload


def test_extracted_bundle_passes_its_runtime_sentinel(tmp_path):
    bundle = build_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(runtime)
    result = subprocess.run(
        [sys.executable, str(runtime / "bundle_self_test.py")],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "PASS"
    assert report["procedural_records"] > 0
    assert report["recursive_world_model_exact"] is True
    assert report["agentic_pipeline_exact"] is True
    assert report["agentic_memory_records"] == 0
    assert report["neural_inference_tested"] is False
    assert report["inactive_channels_packaged"] is False

"""Tests for scripts/preflight.py — R11-B: device/VRAM preflight checks.

All CUDA tests use mock objects so no GPU is required. PyTorch is mocked
where needed so the suite runs without torch installed.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from scripts.preflight import (
    CheckResult,
    DeviceReport,
    check_checkpoint_role,
    check_cuda_available,
    check_device_accessible,
    check_python_version,
    check_torch_available,
    check_torch_build_matches_device,
    check_vram_sufficient,
    estimate_vram_bytes,
    run_preflight,
)
from agent.model import CheckpointDescriptor, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_torch(cuda: bool = False, device_count: int = 0,
                free: int = 4 * 1024**3, total: int = 8 * 1024**3,
                cuda_build: bool = True) -> types.ModuleType:
    """Build a minimal fake torch module for patching sys.modules['torch'].

    cuda_build controls torch.version.cuda: a version string (CUDA build) when
    True, None (CPU-only wheel) when False. Version tag mirrors this so the
    torch-build guard can distinguish CPU wheels.
    """
    t = types.ModuleType("torch")
    t.__version__ = "2.3.0+cu121" if cuda_build else "2.3.0+cpu"
    t.version = types.SimpleNamespace(cuda="12.1" if cuda_build else None)
    t.cuda = MagicMock()
    t.cuda.is_available = MagicMock(return_value=cuda)
    t.cuda.device_count = MagicMock(return_value=device_count)
    t.cuda.get_device_name = MagicMock(return_value="FakeGPU-3090")
    t.cuda.mem_get_info = MagicMock(return_value=(free, total))
    return t


def _make_descriptor(
    role: str = "simulator",
    param_count: int = 7_000_000_000,
    quantization: str | None = "int4",
    device: str = "cpu",
) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        model_name="test-model",
        model_version="v1",
        role=role,
        parameter_count=param_count,
        quantization=quantization,
        device=device,
        artifact_hash="a" * 64,
    )


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------

def test_check_result_str_pass():
    c = CheckResult(name="x", passed=True, message="ok")
    assert "[PASS]" in str(c)


def test_check_result_str_fail():
    c = CheckResult(name="x", passed=False, message="bad")
    assert "[FAIL]" in str(c)


# ---------------------------------------------------------------------------
# estimate_vram_bytes
# ---------------------------------------------------------------------------

def test_estimate_vram_int4():
    # 7B params × 0.625 bytes + 512 MiB overhead
    result = estimate_vram_bytes(7_000_000_000, "int4")
    assert result > 512 * 1024**2
    assert result < 8 * 1024**3  # should fit in 8 GiB


def test_estimate_vram_fp32_larger_than_int4():
    fp32 = estimate_vram_bytes(1_000_000, "fp32")
    int4 = estimate_vram_bytes(1_000_000, "int4")
    assert fp32 > int4


def test_estimate_vram_zero_params_still_has_overhead():
    result = estimate_vram_bytes(0, None)
    assert result == 512 * 1024**2  # exactly _OVERHEAD_BYTES


def test_estimate_vram_unknown_quantization_defaults_to_fp16():
    unknown = estimate_vram_bytes(1_000_000, "q99_unknown")
    fp16 = estimate_vram_bytes(1_000_000, "fp16")
    assert unknown == fp16


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------

def test_python_version_current_passes():
    # We're running on 3.10+ by the project's own requirement
    result = check_python_version(min_minor=10)
    assert result.passed


def test_python_version_future_minor_fails():
    result = check_python_version(min_minor=99)
    assert not result.passed


# ---------------------------------------------------------------------------
# check_torch_available
# ---------------------------------------------------------------------------

def test_torch_available_when_installed():
    fake = _fake_torch()
    with patch.dict(sys.modules, {"torch": fake}):
        # Also patch find_spec to return a truthy spec
        with patch("scripts.preflight.importlib.util.find_spec", return_value=MagicMock()):
            result = check_torch_available()
    assert result.passed
    assert result.detail["version"] == "2.3.0+cu121"


def test_torch_not_available_when_absent():
    with patch("scripts.preflight.importlib.util.find_spec", return_value=None):
        result = check_torch_available()
    assert not result.passed


# ---------------------------------------------------------------------------
# check_cuda_available
# ---------------------------------------------------------------------------

def test_cuda_available_with_fake_gpu():
    fake = _fake_torch(cuda=True, device_count=2)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_cuda_available()
    assert result.passed
    assert result.detail["device_count"] == 2


def test_cuda_not_available_no_gpu():
    fake = _fake_torch(cuda=False, device_count=0)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_cuda_available()
    assert not result.passed


def test_cuda_check_fails_gracefully_without_torch():
    # Simulate torch being unimportable regardless of whether it is installed
    # in the dev venv: block the module so `import torch` raises ImportError.
    with patch.dict(sys.modules, {"torch": None}):
        result = check_cuda_available()
    assert not result.passed
    assert "not installed" in result.message


# ---------------------------------------------------------------------------
# check_device_accessible
# ---------------------------------------------------------------------------

def test_cpu_device_always_passes():
    result = check_device_accessible("cpu")
    assert result.passed


def test_cuda_device_passes_when_available():
    fake = _fake_torch(cuda=True, device_count=1)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_device_accessible("cuda:0")
    assert result.passed
    assert result.detail.get("name") == "FakeGPU-3090"


def test_cuda_device_fails_on_out_of_range_index():
    fake = _fake_torch(cuda=True, device_count=1)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_device_accessible("cuda:2")
    assert not result.passed
    assert "out of range" in result.message


def test_cuda_device_fails_when_cuda_unavailable():
    fake = _fake_torch(cuda=False)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_device_accessible("cuda:0")
    assert not result.passed


# ---------------------------------------------------------------------------
# check_vram_sufficient
# ---------------------------------------------------------------------------

def test_vram_sufficient_cpu_always_passes():
    result = check_vram_sufficient("cpu", 100 * 1024**3)
    assert result.passed


def test_vram_sufficient_passes_when_enough_free():
    # 4 GiB free, need 2 GiB
    fake = _fake_torch(cuda=True, device_count=1,
                       free=4 * 1024**3, total=8 * 1024**3)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_vram_sufficient("cuda:0", 2 * 1024**3)
    assert result.passed
    assert result.detail["free_bytes"] == 4 * 1024**3


def test_vram_sufficient_fails_when_insufficient():
    # 1 GiB free, need 4 GiB
    fake = _fake_torch(cuda=True, device_count=1,
                       free=1 * 1024**3, total=8 * 1024**3)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_vram_sufficient("cuda:0", 4 * 1024**3)
    assert not result.passed
    assert "INSUFFICIENT" in result.message


def test_vram_check_fails_gracefully_without_torch():
    # Block torch import so the check hits its ImportError path regardless of
    # whether a real CUDA torch is installed in the environment.
    with patch.dict(sys.modules, {"torch": None}):
        result = check_vram_sufficient("cuda:0", 1024)
    assert not result.passed


# ---------------------------------------------------------------------------
# check_checkpoint_role
# ---------------------------------------------------------------------------

def test_checkpoint_role_matches():
    desc = _make_descriptor(role="simulator")
    result = check_checkpoint_role(desc, "simulator")
    assert result.passed


def test_checkpoint_role_mismatch():
    desc = _make_descriptor(role="deliberator")
    result = check_checkpoint_role(desc, "simulator")
    assert not result.passed
    assert "deliberator" in result.message


def test_checkpoint_role_non_descriptor():
    result = check_checkpoint_role("not-a-descriptor", "simulator")
    assert not result.passed


# ---------------------------------------------------------------------------
# run_preflight — CPU (no torch required)
# ---------------------------------------------------------------------------

def test_run_preflight_cpu_passes():
    report = run_preflight("cpu")
    # CPU preflight must pass regardless of whether torch/CUDA are present:
    # CUDA availability is advisory for a cpu device.
    assert report.all_passed
    assert report.selected_device == "cpu"


def test_run_preflight_returns_device_report():
    report = run_preflight("cpu")
    assert isinstance(report, DeviceReport)
    assert len(report.checks) >= 2


def test_run_preflight_summary_contains_device():
    report = run_preflight("cpu")
    summary = report.summary()
    assert "cpu" in summary
    assert "PREFLIGHT" in summary


def test_run_preflight_with_descriptor_cpu():
    desc = _make_descriptor(device="cpu", role="simulator")
    report = run_preflight("cpu", descriptor=desc, expected_role="simulator")
    # VRAM check passes for cpu; role check passes
    assert report.all_passed


def test_run_preflight_with_wrong_role_fails():
    desc = _make_descriptor(device="cpu", role="deliberator")
    report = run_preflight("cpu", descriptor=desc, expected_role="simulator")
    assert not report.all_passed
    role_checks = [c for c in report.checks if c.name == "checkpoint_role"]
    assert role_checks and not role_checks[0].passed


def test_run_preflight_footprint_recorded():
    desc = _make_descriptor(param_count=7_000_000_000, quantization="int4")
    report = run_preflight("cpu", descriptor=desc)
    assert report.estimated_footprint_bytes is not None
    assert report.estimated_footprint_bytes > 0


# ---------------------------------------------------------------------------
# run_preflight — fake CUDA path
# ---------------------------------------------------------------------------

def test_run_preflight_cuda_with_sufficient_vram():
    fake = _fake_torch(cuda=True, device_count=1,
                       free=8 * 1024**3, total=16 * 1024**3)
    with patch.dict(sys.modules, {"torch": fake}), \
         patch("scripts.preflight.importlib.util.find_spec", return_value=MagicMock()):
        # 7B int4 needs ~4.9 GiB; 8 GiB free → should pass
        desc = _make_descriptor(
            device="cuda:0", role="simulator",
            param_count=7_000_000_000, quantization="int4",
        )
        report = run_preflight("cuda:0", descriptor=desc, expected_role="simulator")
    assert report.all_passed
    assert report.cuda_available
    assert report.free_vram_bytes == 8 * 1024**3


def test_run_preflight_cuda_with_insufficient_vram_fails():
    fake = _fake_torch(cuda=True, device_count=1,
                       free=1 * 1024**3, total=16 * 1024**3)
    with patch.dict(sys.modules, {"torch": fake}), \
         patch("scripts.preflight.importlib.util.find_spec", return_value=MagicMock()):
        desc = _make_descriptor(
            device="cuda:0", role="simulator",
            param_count=70_000_000_000, quantization="fp16",  # 140 GiB → fail
        )
        report = run_preflight("cuda:0", descriptor=desc)
    assert not report.all_passed
    vram_checks = [c for c in report.checks if c.name == "vram_sufficient"]
    assert vram_checks and not vram_checks[0].passed


# ---------------------------------------------------------------------------
# check_torch_build_matches_device — CPU wheel guard
# ---------------------------------------------------------------------------

def test_torch_build_guard_cpu_device_always_passes():
    # A cpu device accepts any torch build (including the real +cpu dev wheel).
    result = check_torch_build_matches_device("cpu")
    assert result.passed


def test_torch_build_guard_fails_for_cpu_wheel_on_cuda_device():
    fake = _fake_torch(cuda=False, cuda_build=False)  # CPU-only wheel
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_torch_build_matches_device("cuda:0")
    assert not result.passed
    assert "CPU-only" in result.message


def test_torch_build_guard_passes_for_cuda_wheel_on_cuda_device():
    fake = _fake_torch(cuda=True, device_count=1, cuda_build=True)
    with patch.dict(sys.modules, {"torch": fake}):
        result = check_torch_build_matches_device("cuda:0")
    assert result.passed


def test_torch_build_guard_fails_when_torch_absent_on_cuda_device():
    with patch.dict(sys.modules, {"torch": None}):
        result = check_torch_build_matches_device("cuda:0")
    assert not result.passed
    assert "not installed" in result.message


def test_run_preflight_cuda_with_cpu_wheel_fails_closed():
    """The critical case: a +cpu wheel must fail preflight for a cuda device."""
    fake = _fake_torch(cuda=False, cuda_build=False)  # CPU-only wheel, no GPU
    with patch.dict(sys.modules, {"torch": fake}), \
         patch("scripts.preflight.importlib.util.find_spec", return_value=MagicMock()):
        report = run_preflight("cuda:0")
    assert not report.all_passed
    build_checks = [c for c in report.checks if c.name == "torch_build_matches_device"]
    assert build_checks and not build_checks[0].passed

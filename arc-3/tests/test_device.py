"""Tests for agent/device.py — torch-build detection and device resolution.

Uses fake torch modules so behaviour is deterministic regardless of what is
installed in the dev venv (which is a real +cpu build).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.device import (
    TorchBuildInfo,
    DeviceResolution,
    detect_torch_build,
    resolve_device,
)


# ---------------------------------------------------------------------------
# Fake torch helpers
# ---------------------------------------------------------------------------

def _fake_torch(*, cuda_build: bool, cuda_runtime: bool) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.__version__ = "2.3.0+cu121" if cuda_build else "2.3.0+cpu"
    t.version = types.SimpleNamespace(cuda="12.1" if cuda_build else None)
    t.cuda = MagicMock()
    t.cuda.is_available = MagicMock(return_value=cuda_runtime)
    return t


# ---------------------------------------------------------------------------
# detect_torch_build
# ---------------------------------------------------------------------------

def test_detect_absent_torch():
    with patch.dict(sys.modules, {"torch": None}):
        info = detect_torch_build()
    assert not info.installed
    assert not info.cuda_build
    assert not info.cuda_runtime_ok


def test_detect_cpu_only_build():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=False, cuda_runtime=False)}):
        info = detect_torch_build()
    assert info.installed
    assert info.cpu_only
    assert not info.cuda_build
    assert "+cpu" in info.version


def test_detect_cuda_build_with_gpu():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=True, cuda_runtime=True)}):
        info = detect_torch_build()
    assert info.installed
    assert info.cuda_build
    assert not info.cpu_only
    assert info.cuda_runtime_ok


def test_detect_cuda_build_without_gpu():
    """A CUDA build on a machine with no GPU: build is CUDA, runtime unavailable."""
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=True, cuda_runtime=False)}):
        info = detect_torch_build()
    assert info.cuda_build
    assert not info.cpu_only
    assert not info.cuda_runtime_ok


def test_detect_never_raises_on_broken_torch():
    broken = types.ModuleType("torch")
    # No __version__, no version attr, cuda.is_available raises
    broken.cuda = MagicMock()
    broken.cuda.is_available = MagicMock(side_effect=RuntimeError("boom"))
    with patch.dict(sys.modules, {"torch": broken}):
        info = detect_torch_build()
    assert info.installed
    assert not info.cuda_runtime_ok  # exception swallowed → False


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_cpu_request_honoured():
    res = resolve_device("cpu")
    assert res.resolved == "cpu"
    assert not res.degraded


def test_resolve_auto_uses_cuda_when_available():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=True, cuda_runtime=True)}):
        res = resolve_device("auto")
    assert res.resolved == "cuda:0"
    assert not res.degraded


def test_resolve_auto_degrades_to_cpu_without_gpu():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=False, cuda_runtime=False)}):
        res = resolve_device("auto")
    assert res.resolved == "cpu"
    assert res.degraded
    assert res.reason  # explicit, non-empty reason (never silent)


def test_resolve_explicit_cuda_degrades_with_reason_on_cpu_wheel():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=False, cuda_runtime=False)}):
        res = resolve_device("cuda:0")
    assert res.resolved == "cpu"
    assert res.degraded
    assert "CPU-only" in res.reason


def test_resolve_explicit_cuda_honoured_when_available():
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda_build=True, cuda_runtime=True)}):
        res = resolve_device("cuda:1")
    assert res.resolved == "cuda:1"
    assert not res.degraded


def test_resolve_reports_missing_torch_reason():
    with patch.dict(sys.modules, {"torch": None}):
        res = resolve_device("auto")
    assert res.resolved == "cpu"
    assert res.degraded
    assert "not installed" in res.reason


def test_resolve_unknown_device_defaults_to_cpu_with_reason():
    res = resolve_device("tpu:0")
    assert res.resolved == "cpu"
    assert res.degraded
    assert "Unrecognised" in res.reason

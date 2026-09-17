"""Offline device and VRAM preflight checks — R11-B.

Run before loading any model checkpoint to confirm the execution environment
is suitable. All checks are non-destructive reads; no weights are loaded here.

Usage (CLI):
    python scripts/preflight.py --role simulator --device cuda:0
    python scripts/preflight.py --role deliberator --device cpu

Exits 0 on pass, 1 on any failure.

Design constraints
------------------
- PyTorch is optional at import time; checks degrade gracefully when it is
  absent (CPU-only report, no VRAM figures).
- No model weights are loaded.  Parameter count and quantization are read from
  a CheckpointDescriptor that the caller constructs from the manifest.
- Estimates are conservative: the footprint formula rounds UP so borderline
  configurations fail rather than OOM mid-run.
- Each check is a separate function returning a CheckResult so callers can
  act on individual failures rather than parsing exception messages.
"""
from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Allow running as a standalone script (`python scripts/preflight.py`): ensure
# the project root is importable so `agent.*` local imports resolve. When
# imported as a module (tests), conftest.py already handles this; the insert is
# idempotent.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single preflight check."""
    name: str
    passed: bool
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class DeviceReport:
    """Summary of device availability and estimated VRAM headroom.

    All fields are informational; only `all_passed` gates the preflight verdict.
    """
    platform_info: str
    python_version: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    device_count: int
    selected_device: str          # e.g. 'cuda:0' or 'cpu'
    device_name: str | None       # human-readable name from torch
    free_vram_bytes: int | None   # None when CUDA unavailable or torch absent
    total_vram_bytes: int | None
    estimated_footprint_bytes: int | None  # None when descriptor is absent
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [
            f"Platform : {self.platform_info}",
            f"Python   : {self.python_version}",
            f"PyTorch  : {self.torch_version or 'not installed'}",
            f"CUDA     : {'available' if self.cuda_available else 'not available'}",
            f"Device   : {self.selected_device}"
            + (f" ({self.device_name})" if self.device_name else ""),
        ]
        if self.free_vram_bytes is not None:
            lines.append(
                f"VRAM     : {self.free_vram_bytes / 1024**3:.2f} GiB free / "
                f"{self.total_vram_bytes / 1024**3:.2f} GiB total"  # type: ignore[operator]
            )
        if self.estimated_footprint_bytes is not None:
            lines.append(
                f"Footprint: {self.estimated_footprint_bytes / 1024**3:.2f} GiB estimated"
            )
        lines.append("")
        for c in self.checks:
            lines.append(str(c))
        verdict = "PREFLIGHT PASSED" if self.all_passed else "PREFLIGHT FAILED"
        lines.append(f"\n{verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# VRAM footprint estimation
# ---------------------------------------------------------------------------

# Bytes per parameter by quantization label.
# Conservative (ceiling) estimates; e.g. int4 packs 2 params/byte but
# we budget 0.625 to account for activation memory and KV cache.
_BYTES_PER_PARAM: dict[str | None, float] = {
    None:    4.0,    # fp32
    "fp32":  4.0,
    "fp16":  2.0,
    "bf16":  2.0,
    "int8":  1.25,   # 1 byte weights + ~25% overhead
    "int4":  0.625,  # 0.5 byte weights + ~25% overhead
    "q4_0":  0.625,
    "q4_k":  0.625,
    "q5_k":  0.75,
    "q6_k":  0.875,
    "q8_0":  1.25,
    "gguf":  1.0,    # unknown GGUF: assume mixed ~fp16 equivalent
}

# Minimum overhead added on top of weights (activations, KV cache, etc.)
_OVERHEAD_BYTES = 512 * 1024 * 1024   # 512 MiB


def estimate_vram_bytes(parameter_count: int, quantization: str | None) -> int:
    """Return a conservative VRAM footprint estimate in bytes.

    Parameters
    ----------
    parameter_count : int
        Approximate number of model parameters (use 0 if unknown).
    quantization : str | None
        Quantization label from CheckpointDescriptor, e.g. 'int4', 'fp32'.

    Returns
    -------
    int
        Estimated bytes needed, including a fixed 512 MiB overhead allowance.
        Always >= _OVERHEAD_BYTES even for parameter_count=0 so the check
        cannot falsely pass on an empty descriptor.
    """
    q_key = quantization.lower() if quantization else None
    bpp = _BYTES_PER_PARAM.get(q_key, 2.0)   # default fp16 if label unknown
    return int(parameter_count * bpp) + _OVERHEAD_BYTES


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version(min_minor: int = 10) -> CheckResult:
    """Require Python 3.{min_minor}+."""
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= min_minor
    return CheckResult(
        name="python_version",
        passed=ok,
        message=f"{major}.{minor} ({'ok' if ok else f'need 3.{min_minor}+'})",
        detail={"major": major, "minor": minor},
    )


def check_torch_available() -> CheckResult:
    """Confirm PyTorch can be imported."""
    torch_spec = importlib.util.find_spec("torch")
    available = torch_spec is not None
    version: str | None = None
    if available:
        try:
            import torch  # noqa: PLC0415
            version = torch.__version__
        except Exception as exc:
            available = False
            return CheckResult(
                name="torch_available",
                passed=False,
                message=f"torch import failed: {exc}",
            )
    return CheckResult(
        name="torch_available",
        passed=available,
        message=version or "not installed",
        detail={"version": version},
    )


def check_cuda_available() -> CheckResult:
    """Confirm at least one CUDA device is accessible."""
    try:
        import torch  # noqa: PLC0415
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        return CheckResult(
            name="cuda_available",
            passed=available,
            message=f"{count} device(s)" if available else "CUDA not available (CPU fallback)",
            detail={"device_count": count},
        )
    except ImportError:
        return CheckResult(
            name="cuda_available",
            passed=False,
            message="torch not installed; CUDA unavailable",
        )


def check_device_accessible(device: str) -> CheckResult:
    """Confirm the requested device string is usable by torch."""
    if device == "cpu":
        return CheckResult(
            name="device_accessible",
            passed=True,
            message="cpu always available",
            detail={"device": device},
        )
    try:
        import torch  # noqa: PLC0415
        idx = int(device.split(":")[-1]) if ":" in device else 0
        if not torch.cuda.is_available():
            return CheckResult(
                name="device_accessible",
                passed=False,
                message=f"{device}: CUDA not available",
                detail={"device": device},
            )
        count = torch.cuda.device_count()
        if idx >= count:
            return CheckResult(
                name="device_accessible",
                passed=False,
                message=f"{device}: index {idx} out of range (have {count})",
                detail={"device": device, "index": idx, "count": count},
            )
        name = torch.cuda.get_device_name(idx)
        return CheckResult(
            name="device_accessible",
            passed=True,
            message=f"{device}: {name}",
            detail={"device": device, "name": name},
        )
    except ImportError:
        return CheckResult(
            name="device_accessible",
            passed=False,
            message="torch not installed",
        )
    except Exception as exc:
        return CheckResult(
            name="device_accessible",
            passed=False,
            message=f"{device}: {exc}",
        )


def check_vram_sufficient(
    device: str,
    estimated_bytes: int,
) -> CheckResult:
    """Confirm free VRAM on device >= estimated_bytes.

    Always passes for CPU (no VRAM constraint). When torch is absent or CUDA
    is unavailable on a cuda: device, the check fails with a clear message
    rather than silently passing.
    """
    if device == "cpu":
        return CheckResult(
            name="vram_sufficient",
            passed=True,
            message="CPU: no VRAM constraint",
            detail={"device": device, "estimated_bytes": estimated_bytes},
        )

    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return CheckResult(
                name="vram_sufficient",
                passed=False,
                message="CUDA not available; cannot check VRAM",
                detail={"device": device},
            )
        idx = int(device.split(":")[-1]) if ":" in device else 0
        free, total = torch.cuda.mem_get_info(idx)
        ok = free >= estimated_bytes
        gib = 1024 ** 3
        return CheckResult(
            name="vram_sufficient",
            passed=ok,
            message=(
                f"{free / gib:.2f} GiB free / {total / gib:.2f} GiB total; "
                f"need {estimated_bytes / gib:.2f} GiB"
                + (" — OK" if ok else " — INSUFFICIENT")
            ),
            detail={
                "device": device,
                "free_bytes": free,
                "total_bytes": total,
                "estimated_bytes": estimated_bytes,
            },
        )
    except ImportError:
        return CheckResult(
            name="vram_sufficient",
            passed=False,
            message="torch not installed; cannot check VRAM",
            detail={"device": device},
        )
    except Exception as exc:
        return CheckResult(
            name="vram_sufficient",
            passed=False,
            message=f"VRAM check failed: {exc}",
            detail={"device": device},
        )


def check_checkpoint_role(descriptor: object, expected_role: str) -> CheckResult:
    """Confirm the CheckpointDescriptor role matches what the caller expects."""
    from agent.model import CheckpointDescriptor  # local import, no circular dep
    if not isinstance(descriptor, CheckpointDescriptor):
        return CheckResult(
            name="checkpoint_role",
            passed=False,
            message=f"expected CheckpointDescriptor, got {type(descriptor).__name__}",
        )
    ok = descriptor.role == expected_role
    return CheckResult(
        name="checkpoint_role",
        passed=ok,
        message=(
            f"role={descriptor.role!r} ({'ok' if ok else f'expected {expected_role!r}'})"
        ),
        detail={"role": descriptor.role, "expected": expected_role},
    )


def check_torch_build_matches_device(device: str) -> CheckResult:
    """Fail closed if a CPU-only torch build is used for a CUDA device.

    Catches the "wrong wheel shipped to Kaggle" mistake: a '+cpu' PyTorch would
    silently execute on CPU on a GPU notebook and blow the time budget. For a
    'cpu' device this check always passes (CPU build is expected there).
    """
    from agent.device import detect_torch_build  # local import, no circular dep

    if device == "cpu":
        return CheckResult(
            name="torch_build_matches_device",
            passed=True,
            message="cpu device: any torch build acceptable",
            detail={"device": device},
        )

    info = detect_torch_build()
    if not info.installed:
        return CheckResult(
            name="torch_build_matches_device",
            passed=False,
            message=f"{device}: PyTorch not installed",
            detail={"device": device},
        )
    if info.cpu_only:
        return CheckResult(
            name="torch_build_matches_device",
            passed=False,
            message=(
                f"{device}: PyTorch is a CPU-only build ({info.version}); "
                "a CUDA build is required for this device"
            ),
            detail={"device": device, "version": info.version, "cpu_only": True},
        )
    return CheckResult(
        name="torch_build_matches_device",
        passed=True,
        message=f"{device}: CUDA torch build ({info.version})",
        detail={"device": device, "version": info.version, "cpu_only": False},
    )


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------

def run_preflight(
    device: str,
    *,
    descriptor: object | None = None,
    expected_role: str | None = None,
) -> DeviceReport:
    """Run all applicable preflight checks and return a DeviceReport.

    Parameters
    ----------
    device : str
        Target device string, e.g. 'cuda:0' or 'cpu'.
    descriptor : CheckpointDescriptor | None
        If supplied, VRAM footprint and role checks are included.
    expected_role : str | None
        If supplied alongside descriptor, the role check verifies it matches.

    Returns
    -------
    DeviceReport
        DeviceReport.all_passed is True only when every check passes.
    """
    checks: list[CheckResult] = []

    # Gather environment metadata
    plat = platform.platform()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch_check = check_torch_available()
    # torch_available is required for CUDA devices; for CPU it is advisory only
    # (a CPU-only run does not need torch to enumerate the device).
    if device != "cpu" or torch_check.passed:
        checks.append(torch_check)
    else:
        # Record but mark passed so CPU preflight succeeds without torch
        checks.append(CheckResult(
            name=torch_check.name,
            passed=True,
            message=f"{torch_check.message} (advisory for cpu device)",
            detail=torch_check.detail,
        ))
    torch_version: str | None = torch_check.detail.get("version")
    torch_available = torch_check.passed

    cuda_available = False
    device_count = 0
    device_name: str | None = None
    free_vram: int | None = None
    total_vram: int | None = None
    estimated_footprint: int | None = None

    if torch_available:
        cuda_check = check_cuda_available()
        cuda_available = cuda_check.passed
        device_count = cuda_check.detail.get("device_count", 0)
        # CUDA availability only gates the verdict when a CUDA device is targeted.
        # For a CPU run, record the state as advisory so absent CUDA does not fail.
        if device == "cpu":
            checks.append(CheckResult(
                name=cuda_check.name,
                passed=True,
                message=f"{cuda_check.message} (advisory for cpu device)",
                detail=cuda_check.detail,
            ))
        else:
            checks.append(cuda_check)

    dev_check = check_device_accessible(device)
    checks.append(dev_check)
    device_name = dev_check.detail.get("name")

    # Guard against a CPU-only torch wheel being used for a CUDA device.
    # Gating for cuda:* devices; passes trivially for cpu.
    checks.append(check_torch_build_matches_device(device))

    # VRAM checks only when a descriptor is provided
    if descriptor is not None:
        from agent.model import CheckpointDescriptor  # noqa: PLC0415
        if isinstance(descriptor, CheckpointDescriptor):
            estimated_footprint = estimate_vram_bytes(
                descriptor.parameter_count, descriptor.quantization
            )
            vram_check = check_vram_sufficient(device, estimated_footprint)
            checks.append(vram_check)
            if vram_check.detail.get("free_bytes") is not None:
                free_vram = vram_check.detail["free_bytes"]
                total_vram = vram_check.detail["total_bytes"]

        if expected_role is not None:
            checks.append(check_checkpoint_role(descriptor, expected_role))

    checks.append(check_python_version())

    return DeviceReport(
        platform_info=plat,
        python_version=py_ver,
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        device_count=device_count,
        selected_device=device,
        device_name=device_name,
        free_vram_bytes=free_vram,
        total_vram_bytes=total_vram,
        estimated_footprint_bytes=estimated_footprint,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="ARC-3 offline device and VRAM preflight check"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Target device (e.g. 'cuda:0', 'cpu'). Default: cpu",
    )
    parser.add_argument(
        "--role", choices=["simulator", "deliberator"], default=None,
        help="Expected checkpoint role to verify",
    )
    args = parser.parse_args(argv)

    report = run_preflight(args.device, expected_role=args.role)
    print(report.summary())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(_main())

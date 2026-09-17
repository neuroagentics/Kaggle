"""Device resolution and torch-build detection.

Shared helpers used by both the runtime controller and scripts/preflight.py so
CPU/GPU behaviour is defined in exactly one place.

Two concerns:
  1. Detecting whether the installed PyTorch is a CPU-only build. A CPU-only
     wheel (e.g. 'torch==2.14.0+cpu') reports torch.version.cuda is None and
     carries a '+cpu' local-version tag. Shipping it to a CUDA Kaggle run would
     silently execute on CPU and blow the time budget.
  2. Resolving the device to actually use at runtime. When CUDA is requested and
     available, use it. Otherwise degrade to CPU with an EXPLICIT recorded
     warning — never a silent downgrade (blueprint §7).

No model weights are loaded here. torch is imported lazily so this module is
importable (and testable) without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TorchBuildInfo:
    """Facts about the installed PyTorch build.

    installed          : torch importable at all
    version            : torch.__version__ or None
    cuda_build         : True when this is a CUDA build (torch.version.cuda set)
    cpu_only           : True when this is a CPU-only wheel
    cuda_runtime_ok    : True when torch.cuda.is_available() (a GPU is usable now)
    """
    installed: bool
    version: str | None
    cuda_build: bool
    cpu_only: bool
    cuda_runtime_ok: bool


def detect_torch_build() -> TorchBuildInfo:
    """Inspect the installed torch and return a TorchBuildInfo.

    Never raises: if torch is absent, returns installed=False with safe defaults.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return TorchBuildInfo(
            installed=False,
            version=None,
            cuda_build=False,
            cpu_only=False,
            cuda_runtime_ok=False,
        )

    version = getattr(torch, "__version__", None)
    # torch.version.cuda is a CUDA toolkit version string for CUDA builds,
    # and None for CPU-only builds.
    cuda_build_tag = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_build = cuda_build_tag is not None
    cpu_only = (not cuda_build) or (version is not None and "+cpu" in version)

    try:
        cuda_runtime_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_runtime_ok = False

    return TorchBuildInfo(
        installed=True,
        version=version,
        cuda_build=cuda_build,
        cpu_only=cpu_only,
        cuda_runtime_ok=cuda_runtime_ok,
    )


@dataclass(frozen=True)
class DeviceResolution:
    """Result of resolve_device().

    requested   : the device the caller asked for (e.g. 'cuda:0', 'cpu', 'auto')
    resolved    : the device that will actually be used ('cuda:0' or 'cpu')
    degraded    : True when the request could not be honoured and CPU was chosen
    reason      : human-readable explanation; always populated when degraded
    """
    requested: str
    resolved: str
    degraded: bool
    reason: str


def resolve_device(
    requested: str = "auto",
    *,
    allow_cpu_fallback: bool = True,
) -> DeviceResolution:
    """Resolve the device to use, degrading to CPU with an explicit reason.

    Parameters
    ----------
    requested : str
        'auto'   → use CUDA if a usable GPU exists, else CPU.
        'cpu'    → always CPU.
        'cuda'   → CUDA device 0 if available; otherwise fall back per policy.
        'cuda:N' → that CUDA device if available; otherwise fall back per policy.
    allow_cpu_fallback : bool
        When False and CUDA was explicitly requested but unavailable, the
        resolution is marked degraded with resolved='cpu' but the caller is
        expected to treat degraded+not-allowed as a hard error (preflight does).
        We never raise here; the policy decision belongs to the caller.

    Returns
    -------
    DeviceResolution
        Never raises. degraded=True means the request was not honoured.
    """
    info = detect_torch_build()

    # Explicit CPU request: honoured directly.
    if requested == "cpu":
        return DeviceResolution(
            requested=requested, resolved="cpu", degraded=False, reason=""
        )

    wants_cuda = requested == "auto" or requested.startswith("cuda")

    if wants_cuda:
        if info.cuda_runtime_ok:
            resolved = "cuda:0" if requested in ("auto", "cuda") else requested
            return DeviceResolution(
                requested=requested, resolved=resolved, degraded=False, reason=""
            )
        # CUDA wanted but not usable — build the explicit reason.
        if not info.installed:
            reason = "PyTorch is not installed; cannot use CUDA"
        elif info.cpu_only:
            reason = (
                f"PyTorch is a CPU-only build ({info.version}); "
                "no CUDA runtime available"
            )
        else:
            reason = "CUDA runtime is not available on this machine"

        # 'auto' always allows CPU fallback. An explicit cuda request degrades
        # too, but the caller decides whether that is acceptable.
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            degraded=True,
            reason=reason,
        )

    # Unknown device string: treat conservatively as CPU with a reason.
    return DeviceResolution(
        requested=requested,
        resolved="cpu",
        degraded=True,
        reason=f"Unrecognised device request {requested!r}; defaulting to cpu",
    )

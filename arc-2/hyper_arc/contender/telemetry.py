"""Lean resource telemetry for contender evaluations."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


def _stdlib_peak_rss_bytes() -> int | None:
    """Return the process lifetime peak RSS without adding a dependency."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError):
        return None


@dataclass(frozen=True)
class ResourceTelemetry:
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int | None
    peak_gpu_memory_bytes: int | None
    timeout_count: int
    fallback_count: int
    failure_count: int
    sampler: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Measure one evaluation process and any live child processes.

    psutil is optional. When absent, the monitor still records wall/CPU time
    and the operating system's process peak RSS.
    """

    def __init__(self, sample_interval: float = 0.05) -> None:
        self.sample_interval = max(float(sample_interval), 0.01)
        self._started_wall = 0.0
        self._started_cpu = 0.0
        self._peak_rss = 0
        self._peak_gpu: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None
        self._sampler = "stdlib"

    def __enter__(self) -> "ResourceMonitor":
        self._started_wall = time.perf_counter()
        self._started_cpu = time.process_time()
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._sampler = "psutil-process-tree"
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except ImportError:
            self._process = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        return self

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            rss = 0
            try:
                processes = [self._process, *self._process.children(recursive=True)]
                for process in processes:
                    try:
                        rss += int(process.memory_info().rss)
                    except Exception:
                        continue
                self._peak_rss = max(self._peak_rss, rss)
            except Exception:
                pass
            self._stop.wait(self.sample_interval)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval * 2, 0.1))
        fallback_peak = _stdlib_peak_rss_bytes()
        if fallback_peak is not None:
            self._peak_rss = max(self._peak_rss, fallback_peak)
        try:
            import torch

            if torch.cuda.is_available():
                self._peak_gpu = int(torch.cuda.max_memory_allocated())
        except (ImportError, RuntimeError):
            self._peak_gpu = None

    def result(
        self,
        *,
        timeout_count: int = 0,
        fallback_count: int = 0,
        failure_count: int = 0,
    ) -> ResourceTelemetry:
        if not self._started_wall or not self._stop.is_set():
            raise RuntimeError(
                "ResourceMonitor result is available after its context exits"
            )
        return ResourceTelemetry(
            wall_seconds=time.perf_counter() - self._started_wall,
            cpu_seconds=time.process_time() - self._started_cpu,
            peak_rss_bytes=self._peak_rss or None,
            peak_gpu_memory_bytes=self._peak_gpu,
            timeout_count=int(timeout_count),
            fallback_count=int(fallback_count),
            failure_count=int(failure_count),
            sampler=self._sampler,
        )

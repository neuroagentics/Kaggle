"""Resource telemetry contract tests."""

from __future__ import annotations

import pytest

from hyper_arc.contender.telemetry import ResourceMonitor


def test_resource_monitor_records_runtime_and_counts():
    with ResourceMonitor(sample_interval=0.01) as monitor:
        payload = bytearray(1024 * 1024)
        assert len(payload) == 1024 * 1024

    result = monitor.result(timeout_count=2, fallback_count=3, failure_count=1)

    assert result.wall_seconds >= 0
    assert result.cpu_seconds >= 0
    assert result.peak_rss_bytes is None or result.peak_rss_bytes > 0
    assert result.timeout_count == 2
    assert result.fallback_count == 3
    assert result.failure_count == 1
    assert result.sampler in {"stdlib", "psutil-process-tree"}


def test_resource_monitor_rejects_early_result():
    monitor = ResourceMonitor()
    with pytest.raises(RuntimeError, match="after its context exits"):
        monitor.result()

"""Fail-closed tests for the offline Kaggle model boundary."""

from pathlib import Path

import pytest

from hyper_arc.contender.offline_model import (
    OfflineModelError,
    OfflineTransformersTransport,
    check_gpu_memory,
)


def test_offline_transport_rejects_missing_attachment(tmp_path: Path):
    with pytest.raises(OfflineModelError, match="does not exist"):
        OfflineTransformersTransport(tmp_path / "missing")


def test_offline_transport_rejects_directory_without_model_config(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(OfflineModelError, match="config.json"):
        OfflineTransformersTransport(model_dir)


@pytest.mark.parametrize('free_gib,passes', [(14.56, False), (23.0, True)])
def test_actual_gpu_memory_not_requested_accelerator_controls_admission(free_gib, passes):
    from types import SimpleNamespace
    torch = SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda index: (int(free_gib * 2**30), 24 * 2**30),
        get_device_name=lambda index: 'allocated device'))
    if passes:
        assert check_gpu_memory(torch, 20)['free_gib'] == free_gib
    else:
        with pytest.raises(OfflineModelError, match='Insufficient GPU memory'):
            check_gpu_memory(torch, 20)

"""Fail-closed tests for the offline Kaggle model boundary."""

from pathlib import Path

import pytest

from hyper_arc.contender.offline_model import (
    OfflineModelError,
    OfflineTransformersTransport,
    check_gpu_memory,
    gpu_allocation_plan,
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


def test_gpu_plan_reserves_generation_space_and_refuses_single_gpu():
    from types import SimpleNamespace
    torch = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 4,
        mem_get_info=lambda index: (22 * 2**30, 23 * 2**30)))
    assert gpu_allocation_plan(torch, minimum_gpus=2) == {
        0: '12GiB', 1: '18GiB', 2: '18GiB', 3: '18GiB'}
    torch.cuda.device_count = lambda: 1
    with pytest.raises(OfflineModelError, match='needs 2 GPUs'):
        gpu_allocation_plan(torch, minimum_gpus=2)

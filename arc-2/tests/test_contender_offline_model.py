"""Fail-closed tests for the offline Kaggle model boundary."""

from pathlib import Path

import pytest

from hyper_arc.contender.offline_model import (
    OfflineModelError,
    OfflineTransformersTransport,
)


def test_offline_transport_rejects_missing_attachment(tmp_path: Path):
    with pytest.raises(OfflineModelError, match="does not exist"):
        OfflineTransformersTransport(tmp_path / "missing")


def test_offline_transport_rejects_directory_without_model_config(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(OfflineModelError, match="config.json"):
        OfflineTransformersTransport(model_dir)


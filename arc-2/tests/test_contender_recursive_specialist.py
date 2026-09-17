"""Tiny recursive specialist tensor and optimization tests."""

from __future__ import annotations

import torch

from hyper_arc.contender.recursive_specialist import (
    EPISODE_CHANNELS,
    MAX_GRID,
    RecursiveGridSpecialist,
    SpecialistConfig,
    encode_episode,
    encode_target,
    specialist_loss,
)


def test_episode_encoding_has_fixed_shape_and_no_query_answer_slot():
    encoded = encode_episode([([[1]], [[2]])], [[3, 3]])

    assert encoded.shape == (EPISODE_CHANNELS, MAX_GRID, MAX_GRID)
    query_offset = EPISODE_CHANNELS - 11
    assert encoded[query_offset + 3, 0, :2].tolist() == [1.0, 1.0]
    assert encoded[query_offset + 10, 0, :2].tolist() == [1.0, 1.0]


def test_specialist_forward_loss_and_gradient_are_finite():
    model = RecursiveGridSpecialist(
        SpecialistConfig(hidden_channels=16, recurrent_steps=2)
    )
    episode = encode_episode([([[1, 0]], [[1, 0]])], [[2, 0]]).unsqueeze(0)
    target, height, width = encode_target([[2, 0]])

    outputs = model(episode)
    loss = specialist_loss(
        outputs,
        target.unsqueeze(0),
        torch.tensor([height]),
        torch.tensor([width]),
    )
    loss.backward()

    assert outputs[0].shape == (1, 10, MAX_GRID, MAX_GRID)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_default_specialist_is_small_enough_for_the_offline_budget():
    model = RecursiveGridSpecialist()

    assert model.parameter_count < 1_000_000

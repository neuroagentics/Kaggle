"""hyper_arc/cost.py — Exact-match and partial-reward cost functions."""
from __future__ import annotations

import torch

from hyper_arc.esb import ESB


def exact_match(candidate: ESB, target: ESB) -> bool:
    """Return True iff candidate and target have identical shape and values."""
    if candidate.data.shape != target.data.shape:
        return False
    return bool(torch.all(candidate.data == target.data).item())


def partial_reward(candidate: ESB, target: ESB) -> float:
    """Return fraction of correctly predicted cells; 0.0 if shapes differ."""
    if candidate.data.shape != target.data.shape:
        return 0.0
    correct = torch.sum(candidate.data == target.data).item()
    total = target.data.numel()
    return float(correct) / float(total)

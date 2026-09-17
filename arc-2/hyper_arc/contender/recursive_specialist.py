"""Tiny recurrent grid specialist for ARC meta-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from hyper_arc.contender.perception import canonical_grid


MAX_GRID = 30
MAX_DEMOS = 4
GRID_CHANNELS = 11
EPISODE_CHANNELS = (MAX_DEMOS * 2 + 1) * GRID_CHANNELS


@dataclass(frozen=True)
class SpecialistConfig:
    hidden_channels: int = 128
    recurrent_steps: int = 6
    max_demos: int = MAX_DEMOS
    max_grid: int = MAX_GRID


def _encode_grid(grid: Sequence[Sequence[int]] | None) -> Tensor:
    encoded = torch.zeros(GRID_CHANNELS, MAX_GRID, MAX_GRID, dtype=torch.float32)
    if grid is None:
        return encoded
    canonical = canonical_grid(grid)
    height, width = len(canonical), len(canonical[0])
    values = torch.tensor(canonical, dtype=torch.long)
    encoded[:10, :height, :width] = torch.nn.functional.one_hot(
        values, num_classes=10
    ).permute(2, 0, 1)
    encoded[10, :height, :width] = 1.0
    return encoded


def encode_episode(
    demonstrations: Sequence[tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]],
    query: Sequence[Sequence[int]],
) -> Tensor:
    """Encode context and query without any query-output channel."""
    if len(demonstrations) > MAX_DEMOS:
        demonstrations = demonstrations[-MAX_DEMOS:]
    slots: list[Tensor] = []
    for index in range(MAX_DEMOS):
        if index < len(demonstrations):
            source, target = demonstrations[index]
            slots.extend((_encode_grid(source), _encode_grid(target)))
        else:
            slots.extend((_encode_grid(None), _encode_grid(None)))
    slots.append(_encode_grid(query))
    return torch.cat(slots, dim=0)


def encode_target(grid: Sequence[Sequence[int]]) -> tuple[Tensor, int, int]:
    canonical = canonical_grid(grid)
    height, width = len(canonical), len(canonical[0])
    target = torch.full((MAX_GRID, MAX_GRID), -100, dtype=torch.long)
    target[:height, :width] = torch.tensor(canonical, dtype=torch.long)
    return target, height - 1, width - 1


class RecursiveGridSpecialist(nn.Module):
    """Shared-weight latent refinement with grid and output-shape heads."""

    def __init__(self, config: SpecialistConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpecialistConfig()
        hidden = self.config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(EPISODE_CHANNELS, hidden, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.norm = nn.GroupNorm(8, hidden)
        self.color_head = nn.Conv2d(hidden, 10, 1)
        self.shape_pool = nn.AdaptiveAvgPool2d(1)
        self.height_head = nn.Linear(hidden, MAX_GRID)
        self.width_head = nn.Linear(hidden, MAX_GRID)

    def forward(self, episodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.stem(episodes)
        for _ in range(self.config.recurrent_steps):
            latent = self.norm(latent + self.refine(latent))
        colors = self.color_head(latent)
        pooled = self.shape_pool(latent).flatten(1)
        return colors, self.height_head(pooled), self.width_head(pooled)

    @torch.inference_mode()
    def predict(
        self,
        demonstrations: Sequence[
            tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]
        ],
        query: Sequence[Sequence[int]],
        *,
        device: torch.device | str | None = None,
    ) -> list[list[int]]:
        target_device = device or next(self.parameters()).device
        episode = encode_episode(demonstrations, query).unsqueeze(0).to(target_device)
        colors, heights, widths = self(episode)
        height = int(heights.argmax(dim=1).item()) + 1
        width = int(widths.argmax(dim=1).item()) + 1
        grid = colors.argmax(dim=1)[0, :height, :width].cpu()
        return grid.tolist()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def specialist_loss(
    outputs: tuple[Tensor, Tensor, Tensor],
    target_grid: Tensor,
    target_height: Tensor,
    target_width: Tensor,
) -> Tensor:
    colors, heights, widths = outputs
    color_loss = nn.functional.cross_entropy(colors, target_grid, ignore_index=-100)
    height_loss = nn.functional.cross_entropy(heights, target_height)
    width_loss = nn.functional.cross_entropy(widths, target_width)
    return color_loss + 0.25 * (height_loss + width_loss)

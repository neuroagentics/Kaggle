"""Train the tiny recursive specialist on builder-only meta-learning episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from hyper_arc.contender.data_protocol import load_verified_split
from hyper_arc.contender.recursive_specialist import (
    RecursiveGridSpecialist,
    SpecialistConfig,
    encode_episode,
    encode_target,
    specialist_loss,
)


GridList = list[list[int]]


@dataclass(frozen=True)
class Episode:
    context: tuple[tuple[GridList, GridList], ...]
    query: GridList
    target: GridList


def _dihedral(grid: Sequence[Sequence[int]], variant: int) -> GridList:
    value = [list(row) for row in grid]

    def rotate(item: GridList) -> GridList:
        return [list(row) for row in zip(*item[::-1])]

    if variant >= 4:
        value = [row[::-1] for row in value]
        variant -= 4
    for _ in range(variant):
        value = rotate(value)
    return value


def _augment(grid: GridList, variant: int) -> GridList:
    transformed = _dihedral(grid, variant % 8)
    color_variant = variant // 8
    if color_variant == 0:
        return transformed
    colors = list(range(1, 10))
    random.Random(color_variant).shuffle(colors)
    mapping = {0: 0, **{source: target for source, target in zip(range(1, 10), colors)}}
    return [[mapping[value] for value in row] for row in transformed]


class ArcEpisodeDataset(Dataset):
    def __init__(
        self, tasks: Mapping[str, Mapping[str, Any]], *, augmentations: int
    ) -> None:
        if augmentations < 1:
            raise ValueError("augmentations must be positive")
        episodes: list[Episode] = []
        for task in tasks.values():
            demonstrations = task.get("train", [])
            for query_index, query_pair in enumerate(demonstrations):
                context = tuple(
                    (pair["input"], pair["output"])
                    for index, pair in enumerate(demonstrations)
                    if index != query_index
                )
                episodes.append(
                    Episode(context, query_pair["input"], query_pair["output"])
                )
        self.episodes = tuple(episodes)
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.episodes) * self.augmentations

    def __getitem__(self, index: int):
        episode = self.episodes[index // self.augmentations]
        variant = index % self.augmentations
        context = [
            (_augment(source, variant), _augment(target, variant))
            for source, target in episode.context
        ]
        query = _augment(episode.query, variant)
        target_grid = _augment(episode.target, variant)
        target, height, width = encode_target(target_grid)
        return encode_episode(context, query), target, height, width


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--validation-tasks", type=int, default=16)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--recurrent-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/recursive_specialist_smoke.pt")
    )
    return parser.parse_args()


@torch.inference_mode()
def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    exact = 0
    shape_exact = 0
    correct_cells = 0
    target_cells = 0
    correct_nonzero = 0
    target_nonzero = 0
    examples = 0
    for episode, target, height, width in loader:
        episode = episode.to(device)
        target = target.to(device)
        height = height.to(device)
        width = width.to(device)
        outputs = model(episode)
        total_loss += float(specialist_loss(outputs, target, height, width).item())
        colors, heights, widths = outputs
        predicted_grid = colors.argmax(dim=1)
        predicted_height = heights.argmax(dim=1)
        predicted_width = widths.argmax(dim=1)
        for index in range(len(target)):
            h, w = int(height[index].item()) + 1, int(width[index].item()) + 1
            correct_shape = (
                predicted_height[index] == height[index]
                and predicted_width[index] == width[index]
            )
            correct_grid = torch.equal(
                predicted_grid[index, :h, :w], target[index, :h, :w]
            )
            predicted_crop = predicted_grid[index, :h, :w]
            target_crop = target[index, :h, :w]
            nonzero = target_crop != 0
            shape_exact += bool(correct_shape)
            correct_cells += int((predicted_crop == target_crop).sum().item())
            target_cells += h * w
            correct_nonzero += int(
                ((predicted_crop == target_crop) & nonzero).sum().item()
            )
            target_nonzero += int(nonzero.sum().item())
            exact += bool(correct_shape and correct_grid)
            examples += 1
    return {
        "loss": total_loss / max(len(loader), 1),
        "exact_episodes": exact,
        "episodes": examples,
        "shape_accuracy": shape_exact / max(examples, 1),
        "cell_accuracy": correct_cells / max(target_cells, 1),
        "nonzero_cell_accuracy": correct_nonzero / max(target_nonzero, 1),
    }


def main() -> int:
    arguments = _arguments()
    torch.manual_seed(0)
    random.seed(0)
    development, _ = load_verified_split(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "config/arc2_split_v1.json",
    )
    ids = list(development)
    builder_ids, validation_ids = ids[:640], ids[640:]
    builder = {task_id: development[task_id] for task_id in builder_ids[: arguments.tasks]}
    validation = {
        task_id: development[task_id]
        for task_id in validation_ids[: arguments.validation_tasks]
    }
    train_data = ArcEpisodeDataset(builder, augmentations=arguments.augmentations)
    validation_data = ArcEpisodeDataset(validation, augmentations=1)
    train_loader = DataLoader(
        train_data, batch_size=arguments.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_data, batch_size=arguments.batch_size, shuffle=False, num_workers=0
    )
    config = SpecialistConfig(
        hidden_channels=arguments.hidden_channels,
        recurrent_steps=arguments.recurrent_steps,
    )
    model = RecursiveGridSpecialist(config)
    epoch_offset = 0
    if arguments.resume is not None:
        checkpoint = torch.load(arguments.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        epoch_offset = len(checkpoint.get("metadata", {}).get("epochs", []))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    started = time.perf_counter()
    epoch_reports = []
    best_state = deepcopy(model.state_dict())
    best_key = (float("inf"), float("inf"))
    for epoch in range(arguments.epochs):
        model.train()
        running_loss = 0.0
        for episode, target, height, width in train_loader:
            optimizer.zero_grad(set_to_none=True)
            episode = episode.to(device)
            target = target.to(device)
            height = height.to(device)
            width = width.to(device)
            loss = specialist_loss(model(episode), target, height, width)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        validation_metrics = _evaluate(model, validation_loader, device)
        report = {
            "epoch": epoch_offset + epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": validation_metrics["loss"],
            "validation_exact_episodes": validation_metrics["exact_episodes"],
            "validation_episodes": validation_metrics["episodes"],
            "validation_shape_accuracy": validation_metrics["shape_accuracy"],
            "validation_cell_accuracy": validation_metrics["cell_accuracy"],
            "validation_nonzero_cell_accuracy": validation_metrics[
                "nonzero_cell_accuracy"
            ],
        }
        epoch_reports.append(report)
        print(json.dumps(report), flush=True)
        key = (-validation_metrics["exact_episodes"], validation_metrics["loss"])
        if key < best_key:
            best_key = key
            best_state = deepcopy(model.state_dict())
    split_digest = hashlib.sha256(
        json.dumps(
            {"builder": builder_ids, "validation": validation_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": 1,
        "split_sha256": split_digest,
        "builder_tasks_used": len(builder),
        "validation_tasks_used": len(validation),
        "train_episodes": len(train_data),
        "validation_episodes": len(validation_data),
        "augmentations": arguments.augmentations,
        "config": asdict(config),
        "parameter_count": model.parameter_count,
        "device": str(device),
        "resumed_from": str(arguments.resume) if arguments.resume else None,
        "epochs": epoch_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "metadata": result}, arguments.output
    )
    arguments.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

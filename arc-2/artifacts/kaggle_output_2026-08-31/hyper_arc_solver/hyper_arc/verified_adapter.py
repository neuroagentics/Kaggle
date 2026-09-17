"""Adapter for the pinned MIT verified-symbolic ARC ensemble.

The external source is optional during development and bundled into the
offline solver archive for Kaggle. Every returned transform has already been
required by its own harness to replay all demonstrations exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PINNED_COMMIT = "e151937e34c8b34f953833a0dab75797fc737ba4"


def _source_root() -> Path | None:
    candidates = (
        Path(__file__).resolve().parents[1] / "verified_symbolic",
        Path(__file__).resolve().parents[1] / "reference" / "verified-symbolic",
    )
    return next((path for path in candidates if (path / "registry.py").is_file()), None)


def solve_verified(
    task_data: dict[str, Any],
) -> tuple[list[list[list[list[int]]]], int]:
    """Return ranked predictions per test input and number of fitting rules."""
    source = _source_root()
    if source is None:
        return [], 0
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import numpy as np
    from harness import solve_task as external_solve_task
    from registry import ALL

    task = {
        "train": [
            (
                np.asarray(pair["input"], dtype=int),
                np.asarray(pair["output"], dtype=int),
            )
            for pair in task_data.get("train", [])
        ],
        "test": [
            (np.asarray(pair["input"], dtype=int), None)
            for pair in task_data.get("test", [])
        ],
    }
    predictions, fitting = external_solve_task(task, ALL)
    return [
        [candidate.astype(int).tolist() for candidate in candidates]
        for candidates in predictions
    ], fitting

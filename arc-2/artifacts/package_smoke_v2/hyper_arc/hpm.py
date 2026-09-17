"""hyper_arc/hpm.py — Hyperbolic Program Memory.

Two classes:
  GlobalMemoryBank  — read-only seed bank loaded once at startup.
                      Never mutated during inference.
  LocalTaskBuffer   — per-task, instantiated fresh for every task.
                      Updated when programs succeed during inference.

Both share the same add() / query() interface.

Programs are embedded as mean-pooled one-hot vectors (dim=16) projected
onto a Poincaré Ball via geoopt. k-NN retrieval uses hyperbolic distance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import geoopt

from hyper_arc.spatial_dsl import DSLProgram

# ── Constants ─────────────────────────────────────────────────────────────

PRIMITIVE_INDEX: dict[str, int] = {
    "translate":      0,
    "rotate":         1,
    "reflect":        2,
    "extract_object": 3,
    "overlay":        4,
    "crop_to_bbox":   5,
    "scale_integer":  6,
    "tile":           7,
    "connect_points": 8,
    "color_remap":    9,
    "flood_fill":     10,
    "symmetrize":     11,
}

DIM = 16   # embedding dimension (>= 12 primitives)


# ── Encoding ──────────────────────────────────────────────────────────────

def _encode_program(program: DSLProgram) -> torch.Tensor:
    """Mean-pool one-hot primitive indices → raw vector in R^DIM."""
    if not program:
        return torch.zeros(DIM, dtype=torch.float32)
    vecs: list[torch.Tensor] = []
    for name, _ in program:
        idx = PRIMITIVE_INDEX.get(name, 0)
        oh = torch.zeros(DIM, dtype=torch.float32)
        oh[idx % DIM] = 1.0
        vecs.append(oh)
    return torch.stack(vecs).mean(0)


# ── Entry dataclass ───────────────────────────────────────────────────────

@dataclass
class HPMEntry:
    program:   DSLProgram
    embedding: torch.Tensor   # point on Poincaré Ball, norm < 1


# ── Shared mixin ─────────────────────────────────────────────────────────

class _MemoryBase:
    """Shared logic for both memory classes."""

    def __init__(self, k: int = 5, curvature: float = 1.0) -> None:
        self.k = k
        self.manifold = geoopt.PoincareBall(c=curvature)
        self._entries: list[HPMEntry] = []

    # ── Embedding ─────────────────────────────────────────────────────────

    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        """Project raw R^DIM vector onto Poincaré Ball (norm < 1)."""
        norm = raw.norm()
        if norm < 1e-8:
            # Zero vector → origin of the ball
            return torch.zeros(DIM, dtype=torch.float32)
        normed = raw / norm
        # Scale to 0.9 to stay well inside the boundary
        return self.manifold.expmap0(normed * 0.9)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add(self, program: DSLProgram) -> None:
        """Encode a program and store it."""
        raw = _encode_program(program)
        emb = self._project(raw)
        self._entries.append(HPMEntry(program=program, embedding=emb))

    def query(
        self,
        partial_program: DSLProgram,
    ) -> list[tuple[DSLProgram, float]]:
        """Return up to k (program, prior_weight) pairs via hyperbolic k-NN.

        Weights are exp(-dist) normalised to sum ≈ 1.0.
        Returns [] when store is empty.
        """
        if not self._entries:
            return []
        q_raw = _encode_program(partial_program)
        q_emb = self._project(q_raw)
        dists = [
            self.manifold.dist(q_emb, e.embedding).item()
            for e in self._entries
        ]
        k = min(self.k, len(self._entries))
        idxs = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
        raw_w = [float(torch.exp(-torch.tensor(dists[i])).item()) for i in idxs]
        total = sum(raw_w) or 1.0
        return [
            (self._entries[i].program, raw_w[j] / total)
            for j, i in enumerate(idxs)
        ]

    def __len__(self) -> int:
        return len(self._entries)


# ── GlobalMemoryBank ──────────────────────────────────────────────────────

class GlobalMemoryBank(_MemoryBase):
    """Read-only seed bank.

    Loaded once from seed_bank.json at startup.
    Never mutated during inference — only LocalTaskBuffer is updated.
    """

    def load_seed_bank(self, path: str | Path) -> None:
        """Populate from a JSON seed bank file.

        Expected format:
            [{"program": [{"name": ..., "kwargs": {...}}]}, ...]
        """
        data = json.loads(Path(path).read_text())
        for entry in data:
            prog: DSLProgram = [
                (p["name"], p.get("kwargs", p.get("params", {})))
                for p in entry["program"]
            ]
            self.add(prog)

    def save(self, path: str | Path) -> None:
        """Serialise all entries to JSON."""
        data = [
            {
                "program": [
                    {"name": name, "kwargs": kwargs}
                    for name, kwargs in e.program
                ],
                "embedding": e.embedding.tolist(),
            }
            for e in self._entries
        ]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Deserialise entries from JSON (replaces current state)."""
        data = json.loads(Path(path).read_text())
        self._entries.clear()
        for item in data:
            prog: DSLProgram = [
                (p["name"], p.get("kwargs", {}))
                for p in item["program"]
            ]
            emb = torch.tensor(item["embedding"], dtype=torch.float32)
            self._entries.append(HPMEntry(program=prog, embedding=emb))

    def state_dict_entries(self) -> list[dict]:
        """Return serialisable list of entries for checkpointing."""
        return [
            {
                "program":   e.program,
                "embedding": e.embedding.tolist(),
            }
            for e in self._entries
        ]

    def restore_from_state_dict(self, entries: list[dict]) -> None:
        """Restore entries from a checkpoint state dict."""
        self._entries.clear()
        for item in entries:
            emb = torch.tensor(item["embedding"], dtype=torch.float32)
            self._entries.append(
                HPMEntry(program=item["program"], embedding=emb)
            )


# ── LocalTaskBuffer ───────────────────────────────────────────────────────

class LocalTaskBuffer(_MemoryBase):
    """Per-task read/write buffer.

    Instantiated fresh for every new test task.
    Updated via add() whenever a program achieves exact match.
    """
    # Inherits add() and query() directly from _MemoryBase — no overrides needed.

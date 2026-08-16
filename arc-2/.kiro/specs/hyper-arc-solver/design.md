# Design Document — Hyper-ARC Neuro-Symbolic Solver

## Overview

Hyper-ARC is a Python 3.10+ neuro-symbolic solver for ARC-AGI-2. It frames every ARC task as a program synthesis problem: search for a sequence of spatial DSL primitives that maps each training-pair input to its exact output, then apply that program to the test input. The search is guided by a hyperbolic memory of past successes (HPM) and a UCB1-based MCTS engine.

```
┌─────────────────────────────────────────────────────┐
│                      main.py                        │
│  1. download_data.py (data acquisition)             │
│  2. HPM.load_seed_bank()                            │
│  3. for each Task:                                  │
│       MCTS(task, hpm) → program                    │
│  4. write submission.json                           │
└───────────────┬────────────────────────────────────┘
                │
    ┌───────────▼────────────┐
    │       MCTS Engine      │  ← select / expand / simulate / backprop
    │   MCTSNode tree        │
    └───────┬────────┬───────┘
            │        │
  ┌─────────▼──┐  ┌──▼──────────────────┐
  │  SpatialDSL │  │  Hyperbolic Program  │
  │  + ESB      │  │  Memory (HPM)        │
  └─────────────┘  └──────────────────────┘
```

---

## Package Layout

```
hyper-arc-solver/
├── main.py                    # Entry point — orchestrates full pipeline
├── download_data.py           # Standalone Kaggle CLI download helper
├── requirements.txt           # Pinned Python dependencies
├── hyper_arc/
│   ├── __init__.py
│   ├── esb.py                 # EideticSpatialBuffer
│   ├── spatial_dsl.py         # SpatialDSL (12 primitives) + DSLProgram
│   ├── hpm.py                 # GlobalMemoryBank + LocalTaskBuffer
│   ├── mcts.py                # MCTS Engine + MCTSNode
│   ├── cost.py                # exact_match + partial_reward
│   └── utils.py               # Grid I/O, JSON helpers
└── tests/
    ├── __init__.py
    ├── test_esb.py
    ├── test_spatial_dsl.py
    ├── test_cost.py
    └── test_hpm.py
```

---

## Component Designs

### 1. Data Acquisition (`download_data.py`)

A self-contained script with no imports from `hyper_arc/`. Uses `subprocess` to invoke the `kaggle` CLI.

```python
"""download_data.py — Kaggle ARC-AGI-2 dataset downloader."""
import argparse, subprocess, sys
from pathlib import Path

COMPETITION = "arc-prize-2025"
DEFAULT_DIR = "./data/arc-agi-2"

def download(dest: str = DEFAULT_DIR) -> int:
    dest_path = Path(dest)
    # Skip if already present
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"[INFO] Dataset already present at {dest_path}, skipping download.")
        return 0
    # Verify kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], check=True,
                       capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[ERROR] kaggle CLI not found. Install with: pip install kaggle\n"
              "        Then place your API key at ~/.kaggle/kaggle.json",
              file=sys.stderr)
        return 1
    dest_path.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(dest_path),
         "--unzip"],
        capture_output=False,
    )
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=DEFAULT_DIR)
    args = parser.parse_args()
    sys.exit(download(args.dest))
```

---

### 2. Eidetic Spatial Buffer (`hyper_arc/esb.py`)

All grid state is held as `torch.Tensor` of dtype `torch.int64`, shape `(C, H, W)`. Operations return new `ESB` instances (immutable style).

```python
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class ESB:
    """Eidetic Spatial Buffer — immutable tensor wrapper for ARC grids."""
    data: Tensor   # shape (C, H, W), dtype torch.int64

    # ── Constructors ───────────────────────────────────────────
    @classmethod
    def from_grid(cls, grid: list[list[int]]) -> "ESB":
        """Build a single-channel ESB from a 2-D ARC grid."""
        t = torch.tensor(grid, dtype=torch.int64).unsqueeze(0)  # (1, H, W)
        return cls(data=t)

    @property
    def C(self) -> int: return self.data.shape[0]
    @property
    def H(self) -> int: return self.data.shape[1]
    @property
    def W(self) -> int: return self.data.shape[2]

    # ── Core Operations ────────────────────────────────────────
    def pad(self, top: int, bottom: int, left: int, right: int,
            fill_value: int = 0) -> "ESB":
        new_h = self.H + top + bottom
        new_w = self.W + left + right
        if new_h < 1 or new_w < 1:
            raise ValueError(
                f"pad would produce shape ({new_h},{new_w}); both dims must be >= 1")
        out = torch.full((self.C, new_h, new_w), fill_value,
                         dtype=torch.int64)
        out[:, top:top + self.H, left:left + self.W] = self.data
        return ESB(out)

    def crop(self, row_start: int, row_end: int,
             col_start: int, col_end: int) -> "ESB":
        h = row_end - row_start
        w = col_end - col_start
        if h < 1 or w < 1:
            raise ValueError(
                f"crop region has size ({h},{w}); both dims must be >= 1")
        return ESB(self.data[:, row_start:row_end, col_start:col_end].clone())

    def mask(self, condition: Tensor) -> "ESB":
        """Zero out cells where condition is False; keep where True."""
        out = self.data.clone()
        out[:, ~condition] = 0
        return ESB(out)

    def shift(self, delta_row: int, delta_col: int,
              fill_value: int = 0) -> "ESB":
        out = torch.full_like(self.data, fill_value)
        src_r0 = max(0, -delta_row);  src_r1 = self.H - max(0, delta_row)
        dst_r0 = max(0,  delta_row);  dst_r1 = self.H - max(0, -delta_row)
        src_c0 = max(0, -delta_col);  src_c1 = self.W - max(0, delta_col)
        dst_c0 = max(0,  delta_col);  dst_c1 = self.W - max(0, -delta_col)
        if src_r0 < src_r1 and src_c0 < src_c1:
            out[:, dst_r0:dst_r1, dst_c0:dst_c1] = \
                self.data[:, src_r0:src_r1, src_c0:src_c1]
        return ESB(out)

    def to_grid(self) -> list[list[int]]:
        """Export single-channel ESB back to 2-D list of ints."""
        return self.data[0].tolist()
```

---

### 3. Spatial DSL (`hyper_arc/spatial_dsl.py`)

`SpatialDSL` is a stateless class in a dedicated file. All 12 primitives are pure deterministic tensor operations — zero learned weights. Each primitive accepts `(C, H, W)` ESB data or batched `(B, C, H, W)` tensors and returns a new ESB. A `DSLProgram` is a typed list of `(primitive_name, kwargs)` tuples.

#### Critical Constraints
- **Zero neural approximations**: all ops are pure math/tensor — no `nn.Module`, no gradients
- **Batch dimension**: all primitives handle `(B, C, H, W)` for parallel MCTS hypothesis evaluation
- **Out-of-bounds default**: `translate` and `rotate` pad with 0 (background) unless `wrap=True`
- **MCTS-enumerable parameters**: all parameter domains are finite and discrete

#### Parameter Domains (for MCTS enumeration)
| Primitive | Enumerable parameters |
|---|---|
| `translate` | `dx, dy ∈ [-2,2]×[-2,2]` excl. (0,0); `wrap ∈ {False, True}` |
| `rotate` | `degrees ∈ {90, 180, 270}` |
| `reflect` | `axis ∈ {"horizontal", "vertical"}` |
| `extract_object` | `(x, y)` from corners + center of grid |
| `overlay` | `blend_mode ∈ {"overwrite", "or"}` |
| `crop_to_bbox` | no parameters |
| `scale_integer` | `factor ∈ {2, 3, 4}` (upscale only in MCTS; downscale via 0.5 optional) |
| `tile` | `(repeats_h, repeats_w) ∈ {2,3}×{2,3}` |
| `connect_points` | seed point pairs from grid corners; `color ∈ colors_present` |
| `color_remap` | single-color swaps: `{src: dst}` for all `src ∈ colors_present`, `dst ∈ 0–9`, `src ≠ dst` |
| `flood_fill` | seed points (corners + center); `new_color ∈ 0–9` excl. current cell color |
| `symmetrize` | `axis ∈ {"horizontal", "vertical"}`; `mode ∈ {"copy", "priority"}` |

```python
from __future__ import annotations
from collections import deque
from typing import Any, Literal
import torch
from hyper_arc.esb import ESB

VALID_ROTATIONS  = {90, 180, 270}
VALID_AXES       = {"horizontal", "vertical"}
VALID_BLEND      = {"overwrite", "or"}
VALID_SYM_MODES  = {"copy", "priority"}
VALID_SCALES     = {0.5, 1, 2, 3, 4}

PrimitiveCall = tuple[str, dict[str, Any]]
DSLProgram    = list[PrimitiveCall]


class SpatialDSL:
    """12-primitive stateless Spatial DSL. All ops are pure tensor math."""

    # ── Group 1: Rigid Spatial Transformations ────────────────

    @staticmethod
    def translate(esb: ESB, dx: int, dy: int, wrap: bool = False) -> ESB:
        if wrap:
            shifted = torch.roll(esb.data, shifts=(dy, dx), dims=(1, 2))
        else:
            shifted = esb.shift(delta_row=dy, delta_col=dx, fill_value=0).data
        return ESB(shifted)

    @staticmethod
    def rotate(esb: ESB, degrees: int) -> ESB:
        if degrees not in VALID_ROTATIONS:
            raise ValueError(f"rotate degrees must be in {VALID_ROTATIONS}, got {degrees}")
        k = degrees // 90
        return ESB(torch.rot90(esb.data, k=-k, dims=[1, 2]).clone())

    @staticmethod
    def reflect(esb: ESB, axis: str) -> ESB:
        if axis not in VALID_AXES:
            raise ValueError(f"reflect axis must be in {VALID_AXES}, got {axis!r}")
        dim = 1 if axis == "horizontal" else 2
        return ESB(esb.data.flip(dims=[dim]).clone())

    # ── Group 2: Object Manipulation & Topology ──────────────

    @staticmethod
    def extract_object(esb: ESB, x: int, y: int) -> ESB:
        """Isolate the connected component at (row=y, col=x) via BFS CCL."""
        data   = esb.data.clone()
        C, H, W = data.shape
        target = data[0, y, x].item()
        visited = torch.zeros((H, W), dtype=torch.bool)
        queue   = deque([(y, x)])
        keep    = torch.zeros((H, W), dtype=torch.bool)
        while queue:
            r, c = queue.popleft()
            if r < 0 or r >= H or c < 0 or c >= W or visited[r, c]:
                continue
            if data[0, r, c].item() != target:
                continue
            visited[r, c] = True
            keep[r, c]    = True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                queue.append((r+dr, c+dc))
        out = torch.zeros_like(data)
        out[:, keep] = data[:, keep]
        return ESB(out)

    @staticmethod
    def overlay(esb: ESB, foreground: ESB,
                blend_mode: str = "overwrite") -> ESB:
        if blend_mode not in VALID_BLEND:
            raise ValueError(f"blend_mode must be in {VALID_BLEND}, got {blend_mode!r}")
        bg  = esb.data.clone()
        fg  = foreground.data
        if blend_mode == "overwrite":
            mask = fg != 0
        else:  # "or"
            mask = (fg != 0) | (bg != 0)
        out = bg.clone()
        out[mask] = fg[mask] if blend_mode == "overwrite" else (fg | bg)[mask]
        return ESB(out)

    @staticmethod
    def crop_to_bbox(esb: ESB) -> ESB:
        """Return minimal bounding box containing all non-zero cells."""
        nz = torch.nonzero(esb.data[0])
        if nz.numel() == 0:
            return esb  # all background — return as-is
        r0, c0 = nz[:, 0].min().item(), nz[:, 1].min().item()
        r1, c1 = nz[:, 0].max().item() + 1, nz[:, 1].max().item() + 1
        return ESB(esb.data[:, r0:r1, c0:c1].clone())

    # ── Group 3: Geometric Scaling & Pattern Generation ──────

    @staticmethod
    def scale_integer(esb: ESB, factor: float) -> ESB:
        if factor not in VALID_SCALES:
            raise ValueError(f"scale_integer factor must be in {VALID_SCALES}, got {factor}")
        if factor >= 1:
            f = int(factor)
            out = esb.data.repeat_interleave(f, dim=1).repeat_interleave(f, dim=2)
            return ESB(out)
        else:
            # Downscale by 0.5 — max-pool with kernel=2
            import torch.nn.functional as F
            d   = esb.data.unsqueeze(0).float()
            out = F.max_pool2d(d, kernel_size=2, stride=2).long().squeeze(0)
            return ESB(out)

    @staticmethod
    def tile(esb: ESB, repeats_h: int, repeats_w: int) -> ESB:
        out = torch.tile(esb.data, dims=(1, repeats_h, repeats_w))
        return ESB(out)

    @staticmethod
    def connect_points(esb: ESB, color: int,
                       row1: int, col1: int,
                       row2: int, col2: int) -> ESB:
        """Draw a line from (row1,col1) to (row2,col2) using Bresenham's algorithm."""
        data = esb.data.clone()
        # Bresenham's line
        r, c  = row1, col1
        dr    = abs(row2 - row1); sr = 1 if row2 > row1 else -1
        dc    = abs(col2 - col1); sc = 1 if col2 > col1 else -1
        err   = dr - dc
        H, W  = data.shape[1], data.shape[2]
        while True:
            if 0 <= r < H and 0 <= c < W:
                data[:, r, c] = color
            if r == row2 and c == col2:
                break
            e2 = 2 * err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc
        return ESB(data)

    # ── Group 4: Logic, Color & Constraints ──────────────────

    @staticmethod
    def color_remap(esb: ESB, mapping: dict[int, int]) -> ESB:
        """Remap colors using a 10-element lookup tensor (ARC has colors 0–9)."""
        lookup = torch.arange(10, dtype=torch.int64)
        for src, dst in mapping.items():
            lookup[src] = dst
        out = lookup[esb.data.clamp(0, 9)]
        return ESB(out)

    @staticmethod
    def flood_fill(esb: ESB, row: int, col: int, new_color: int) -> ESB:
        data  = esb.data.clone()
        C, H, W = data.shape
        target  = data[0, row, col].item()
        if target == new_color:
            return ESB(data)
        visited = torch.zeros((H, W), dtype=torch.bool)
        queue   = deque([(row, col)])
        while queue:
            r, c = queue.popleft()
            if r < 0 or r >= H or c < 0 or c >= W or visited[r, c]:
                continue
            if data[0, r, c].item() != target:
                continue
            visited[r, c] = True
            data[:, r, c] = new_color
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                queue.append((r+dr, c+dc))
        return ESB(data)

    @staticmethod
    def symmetrize(esb: ESB, axis: str,
                   mode: Literal["copy", "priority"] = "copy") -> ESB:
        if axis not in VALID_AXES:
            raise ValueError(f"axis must be in {VALID_AXES}, got {axis!r}")
        if mode not in VALID_SYM_MODES:
            raise ValueError(f"mode must be in {VALID_SYM_MODES}, got {mode!r}")
        data = esb.data.clone()
        if axis == "horizontal":
            flipped = data.flip(dims=[1])
        else:
            flipped = data.flip(dims=[2])
        if mode == "copy":
            # Copy second half from first half (flip of first half)
            out = flipped.clone()
        else:  # "priority" — non-zero wins
            out = torch.where(data != 0, data, flipped)
        return ESB(out)

    # ── Program Execution ────────────────────────────────────

    def apply_program(self, esb: ESB, program: DSLProgram) -> ESB:
        for name, kwargs in program:
            fn = getattr(self, name, None)
            if fn is None:
                raise ValueError(f"Unknown primitive: {name!r}")
            esb = fn(esb, **kwargs)
        return esb

    # ── Introspection ────────────────────────────────────────

    @staticmethod
    def list_primitives() -> list[dict]:
        return [
            {"name": "translate",      "params": {"dx": "int[-2,2]", "dy": "int[-2,2]", "wrap": "bool"}},
            {"name": "rotate",         "params": {"degrees": "int ∈ {90,180,270}"}},
            {"name": "reflect",        "params": {"axis": "str ∈ {horizontal,vertical}"}},
            {"name": "extract_object", "params": {"x": "int", "y": "int"}},
            {"name": "overlay",        "params": {"foreground": "ESB", "blend_mode": "str ∈ {overwrite,or}"}},
            {"name": "crop_to_bbox",   "params": {}},
            {"name": "scale_integer",  "params": {"factor": "float ∈ {0.5,1,2,3,4}"}},
            {"name": "tile",           "params": {"repeats_h": "int", "repeats_w": "int"}},
            {"name": "connect_points", "params": {"color": "int", "row1": "int", "col1": "int",
                                                   "row2": "int", "col2": "int"}},
            {"name": "color_remap",    "params": {"mapping": "dict[int,int]"}},
            {"name": "flood_fill",     "params": {"row": "int", "col": "int", "new_color": "int"}},
            {"name": "symmetrize",     "params": {"axis": "str ∈ {horizontal,vertical}",
                                                   "mode": "str ∈ {copy,priority}"}},
        ]
```

---

### 4. Hyperbolic Program Memory (`hyper_arc/hpm.py`)

Programs are encoded as fixed-length vectors (mean-pooled one-hot of primitive indices), then projected onto the Poincaré Ball using `geoopt`. k-NN is computed by iterating over hyperbolic distances (manageable for seed banks of a few thousand programs).

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, torch, geoopt
from hyper_arc.spatial_dsl import DSLProgram

PRIMITIVE_INDEX = {
    "translate": 0, "rotate": 1, "reflect": 2,
    "extract_object": 3, "overlay": 4, "crop_to_bbox": 5,
    "scale_integer": 6, "tile": 7, "connect_points": 8,
    "color_remap": 9, "flood_fill": 10, "symmetrize": 11,
}
DIM = 16   # embedding dimension (>= 12 primitives)

def _encode_program(program: DSLProgram) -> torch.Tensor:
    """Mean-pool one-hot primitive indices → raw vector in R^DIM."""
    if not program:
        return torch.zeros(DIM)
    vecs = []
    for name, _ in program:
        idx = PRIMITIVE_INDEX.get(name, 0)
        oh = torch.zeros(DIM)
        oh[idx % DIM] = 1.0
        vecs.append(oh)
    return torch.stack(vecs).mean(0)

@dataclass
class HPMEntry:
    program: DSLProgram
    embedding: torch.Tensor   # point on Poincaré Ball

class HyperbolicProgramMemory:

    def __init__(self, k: int = 5, curvature: float = 1.0):
        self.k = k
        self.manifold = geoopt.PoincareBall(c=curvature)
        self._entries: list[HPMEntry] = []

    # ── Embedding ────────────────────────────────────────────
    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        """Project raw vector onto Poincaré Ball (norm < 1)."""
        normed = raw / (raw.norm() + 1e-8)
        scale  = 0.9   # keep well inside boundary
        return self.manifold.expmap0(normed * scale)

    # ── Seed Bank ────────────────────────────────────────────
    def load_seed_bank(self, path: str | Path) -> None:
        """Load pre-built seed programs from a JSON file."""
        data = json.loads(Path(path).read_text())
        for entry in data:
            prog: DSLProgram = [(p["name"], p["kwargs"])
                                for p in entry["program"]]
            self.add(prog)

    # ── CRUD ─────────────────────────────────────────────────
    def add(self, program: DSLProgram) -> None:
        raw  = _encode_program(program)
        emb  = self._project(raw)
        self._entries.append(HPMEntry(program=program, embedding=emb))

    def query(self, partial_program: DSLProgram
              ) -> list[tuple[DSLProgram, float]]:
        """Return up to k (program, prior_weight) pairs by hyperbolic k-NN."""
        if not self._entries:
            return []
        q_raw = _encode_program(partial_program)
        q_emb = self._project(q_raw)
        dists = [
            self.manifold.dist(q_emb, e.embedding).item()
            for e in self._entries
        ]
        k     = min(self.k, len(self._entries))
        idxs  = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
        # Convert distances to weights: w_i = exp(-d_i) / Z
        raw_w = [torch.exp(-torch.tensor(dists[i])).item() for i in idxs]
        total = sum(raw_w) or 1.0
        return [(self._entries[i].program, raw_w[j] / total)
                for j, i in enumerate(idxs)]

    # ── Serialization ────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        data = [
            {"program": [{"name": n, "kwargs": kw}
                         for n, kw in e.program],
             "embedding": e.embedding.tolist()}
            for e in self._entries
        ]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._entries.clear()
        for item in data:
            prog: DSLProgram = [(p["name"], p["kwargs"])
                                for p in item["program"]]
            emb = torch.tensor(item["embedding"])
            self._entries.append(HPMEntry(program=prog, embedding=emb))
```

---

### 5. Exact-Match Cost Function (`hyper_arc/cost.py`)

```python
import torch
from hyper_arc.esb import ESB

def exact_match(candidate: ESB, target: ESB) -> bool:
    """True iff candidate and target have identical shape and all values equal."""
    if candidate.data.shape != target.data.shape:
        return False
    return bool(torch.all(candidate.data == target.data).item())

def partial_reward(candidate: ESB, target: ESB) -> float:
    """Fraction of correctly predicted cells; 0.0 if shapes differ."""
    if candidate.data.shape != target.data.shape:
        return 0.0
    correct = torch.sum(candidate.data == target.data).item()
    total   = target.data.numel()
    return float(correct) / float(total)
```

---

### 6. MCTS Engine (`hyper_arc/mcts.py`)

#### 6.1 MCTSNode

```python
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from hyper_arc.esb import ESB
from hyper_arc.dsl import PrimitiveCall

@dataclass
class MCTSNode:
    state:        ESB
    parent:       Optional["MCTSNode"]
    action_taken: Optional[PrimitiveCall]
    children:     list["MCTSNode"]     = field(default_factory=list)
    visits:       int                  = 0   # N
    value:        float                = 0.0  # Q

    def ucb1(self, C: float, prior: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_n = self.parent.visits if self.parent else 1
        exploit  = self.value
        explore  = C * math.sqrt(math.log(parent_n) / self.visits)
        return exploit + explore + prior
```

#### 6.2 Action Space

Two-tiered:
1. **Macro-step**: choose a DSL primitive (6 options).
2. **Micro-step**: deterministically enumerate valid parameter assignments for the chosen primitive given the current grid state.

```python
from hyper_arc.dsl import SpatialDSL, PrimitiveCall
from hyper_arc.esb import ESB

DSL = SpatialDSL()

def enumerate_actions(esb: ESB) -> list[PrimitiveCall]:
    """Generate all valid (primitive, kwargs) for the current ESB."""
    H, W = esb.H, esb.W
    actions: list[PrimitiveCall] = []
    # translate: small offsets only
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            actions.append(("translate", {"dx": dx, "dy": dy}))
    # rotate
    for deg in [90, 180, 270]:
        actions.append(("rotate", {"degrees": deg}))
    # reflect
    for axis in ["horizontal", "vertical"]:
        actions.append(("reflect", {"axis": axis}))
    # flood_fill: corners + center
    seeds = [(0,0),(0,W-1),(H-1,0),(H-1,W-1),(H//2,W//2)]
    colors_present = esb.data[0].unique().tolist()
    for r, c in seeds:
        for nc in range(10):
            if nc != esb.data[0, r, c].item():
                actions.append(("flood_fill",
                                {"row": r, "col": c, "new_color": nc}))
    # color_map: single-color remappings
    for src in [int(v) for v in colors_present]:
        for dst in range(10):
            if src != dst:
                actions.append(("color_map", {"mapping": {src: dst}}))
    return actions
```

#### 6.3 MCTS Algorithm

```python
import random, math
from hyper_arc.mcts import MCTSNode, enumerate_actions
from hyper_arc.esb import ESB
from hyper_arc.hpm import HyperbolicProgramMemory
from hyper_arc.cost import exact_match, partial_reward
from hyper_arc.dsl import SpatialDSL, PrimitiveCall, DSLProgram

MAX_ITERATIONS = 2000
MAX_ROLLOUT_DEPTH = 10
DSL = SpatialDSL()

class MCTSEngine:

    def __init__(self, hpm: HyperbolicProgramMemory, C: float = 1.41):
        self.hpm = hpm
        self.C   = C

    def _get_prior(self, node: MCTSNode, action: PrimitiveCall) -> float:
        partial = self._path_actions(node) + [action]
        neighbors = self.hpm.query(partial)
        return sum(w for _, w in neighbors) / len(neighbors) if neighbors else 0.0

    def _path_actions(self, node: MCTSNode) -> DSLProgram:
        path: DSLProgram = []
        cur = node
        while cur.parent is not None and cur.action_taken is not None:
            path.append(cur.action_taken)
            cur = cur.parent
        return list(reversed(path))

    # ── Four MCTS Phases ─────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            actions = [c.action_taken for c in node.children]
            scores  = [
                c.ucb1(self.C, self._get_prior(c, c.action_taken))
                for c in node.children
            ]
            node = node.children[scores.index(max(scores))]
        return node

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        actions = enumerate_actions(node.state)
        for act in actions:
            name, kwargs = act
            try:
                new_state = getattr(DSL, name)(node.state, **kwargs)
            except (ValueError, IndexError):
                continue
            child = MCTSNode(state=new_state, parent=node,
                             action_taken=act)
            node.children.append(child)
        return node.children

    def _rollout(self, node: MCTSNode,
                 target: ESB) -> tuple[float, DSLProgram | None]:
        state   = node.state
        history = self._path_actions(node)
        for _ in range(MAX_ROLLOUT_DEPTH):
            if exact_match(state, target):
                return 1.0, history
            actions  = enumerate_actions(state)
            if not actions:
                break
            # Memory-weighted selection
            priors   = [self._get_prior(node, a) for a in actions]
            total    = sum(priors)
            if total > 0:
                weights = [p / total for p in priors]
                action  = random.choices(actions, weights=weights, k=1)[0]
            else:
                action  = random.choice(actions)
            name, kwargs = action
            try:
                state = getattr(DSL, name)(state, **kwargs)
            except (ValueError, IndexError):
                continue
            history = history + [action]
        return partial_reward(state, target), None

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value  += (reward - cur.value) / cur.visits
            cur = cur.parent

    # ── Public API ───────────────────────────────────────────

    def solve(self, training_pairs: list[tuple[ESB, ESB]],
              ) -> DSLProgram:
        """Run MCTS for up to MAX_ITERATIONS; return best discovered program."""
        if not training_pairs:
            return []

        input_esb, target_esb = training_pairs[0]  # solve first pair
        root   = MCTSNode(state=input_esb, parent=None, action_taken=None)
        best_q = -1.0
        best_program: DSLProgram = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            leaf   = self._select(root)
            if not leaf.children:
                self._expand(leaf)

            # Pick a child to rollout (or leaf if no children)
            target = leaf.children[0] if leaf.children else leaf
            reward, solved_program = self._rollout(target, target_esb)

            if solved_program is not None:
                # Exact match — update HPM and return
                self.hpm.add(solved_program)
                self._backprop(target, 1.0)
                return solved_program

            self._backprop(target, reward)
            prog = self._path_actions(target)
            if reward > best_q:
                best_q       = reward
                best_program = prog

        # No exact match within budget — return highest-Q leaf path
        return best_program
```

---

### 7. Solver Orchestration (`main.py`)

```python
"""main.py — Hyper-ARC end-to-end solver."""
import json, sys, traceback
from pathlib import Path
from hyper_arc.esb import ESB
from hyper_arc.hpm import HyperbolicProgramMemory
from hyper_arc.mcts import MCTSEngine
from hyper_arc.dsl import SpatialDSL

DATA_DIR      = Path("./data/arc-agi-2")
SEED_BANK     = Path("./hyper_arc/seed_bank.json")
OUTPUT_FILE   = Path("./submission.json")

def load_tasks(data_dir: Path) -> dict[str, dict]:
    tasks = {}
    for fpath in sorted(data_dir.glob("**/*.json")):
        try:
            tasks[fpath.stem] = json.loads(fpath.read_text())
        except Exception:
            pass
    return tasks

def main() -> int:
    hpm = HyperbolicProgramMemory(k=5)
    if SEED_BANK.exists():
        hpm.load_seed_bank(SEED_BANK)
        print(f"[INFO] Loaded seed bank: {len(hpm._entries)} programs")
    else:
        print("[WARN] No seed bank found; HPM initialized empty.")

    engine = MCTSEngine(hpm=hpm)
    dsl    = SpatialDSL()
    tasks  = load_tasks(DATA_DIR)
    print(f"[INFO] Found {len(tasks)} tasks.")

    submission: dict[str, list] = {}

    for task_id, task in tasks.items():
        try:
            train_pairs = [
                (ESB.from_grid(p["input"]), ESB.from_grid(p["output"]))
                for p in task.get("train", [])
            ]
            program = engine.solve(train_pairs)
            exact   = program is not None and len(program) > 0

            predictions = []
            for test_pair in task.get("test", []):
                inp  = ESB.from_grid(test_pair["input"])
                pred = dsl.apply_program(inp, program) if program else inp
                predictions.append(pred.to_grid())

            submission[task_id] = predictions
            print(f"[TASK] {task_id}  exact_match={exact}  "
                  f"program_len={len(program)}")

        except Exception as exc:
            print(f"[ERROR] {task_id}: {exc}")
            traceback.print_exc()
            submission[task_id] = []

    OUTPUT_FILE.write_text(json.dumps(submission, indent=2))
    print(f"[INFO] Wrote {OUTPUT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

### 8. Requirements and Packaging

**`requirements.txt`:**
```
torch==2.3.*
geoopt==0.5.*
numpy==1.26.*
kaggle==1.6.*
pytest==8.2.*
```

**`hyper_arc/__init__.py`:**
```python
"""Hyper-ARC: Neuro-Symbolic ARC-AGI-2 Solver."""
from hyper_arc.esb import ESB
from hyper_arc.spatial_dsl import SpatialDSL, DSLProgram
from hyper_arc.hpm import HyperbolicProgramMemory
from hyper_arc.mcts import MCTSEngine
from hyper_arc.cost import exact_match, partial_reward

__all__ = [
    "ESB", "SpatialDSL", "DSLProgram",
    "HyperbolicProgramMemory", "MCTSEngine",
    "exact_match", "partial_reward",
]
```

---

## Data Flow

```
ARC JSON files
     │
     ▼
ESB.from_grid()          ← input/output grids become (1,H,W) int64 tensors
     │
     ▼
MCTSEngine.solve()       ← searches DSL program space for each training pair
  ├─ _select()           ← UCB1 + HPM prior → pick most promising node
  ├─ _expand()           ← enumerate_actions() → child nodes
  ├─ _rollout()          ← memory-weighted random walk, depth ≤ 10
  └─ _backprop()         ← update N, Q up to root
     │
     ▼
DSLProgram               ← sequence of (primitive_name, kwargs)
     │
     ▼
SpatialDSL.apply_program()   ← deterministic replay on test input
     │
     ▼
submission.json
```

---

## Error Handling

| Scenario | Module | Behavior |
|---|---|---|
| Kaggle CLI missing | `download_data.py` | Print actionable message, exit 1 |
| Data dir already populated | `download_data.py` | Skip download, log info |
| ESB dimension < 1 | `esb.py` | Raise `ValueError` with message |
| Invalid DSL parameter | `dsl.py` | Raise `ValueError` with parameter name + range |
| HPM store empty at query | `hpm.py` | Return empty list |
| HPM store < k at query | `hpm.py` | Return all entries, no error |
| Unhandled task exception | `main.py` | Log task ID + traceback, skip, continue |
| No exact match in budget | `mcts.py` | Return best partial program |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

---

### Property 1: ESB Shape Invariant

*For any* valid 2-D ARC grid (any height H and width W), constructing an ESB from that grid must yield a tensor of dtype `torch.int64` and shape `(1, H, W)`. Furthermore, for any pad operation with non-negative parameters `(top, bottom, left, right)`, the output ESB must have shape `(C, H+top+bottom, W+left+right)`. For any shift operation `(delta_row, delta_col)`, the output ESB must have identical shape to the input ESB.

**Validates: Requirements 2.1, 2.2, 2.5**

---

### Property 2: Crop Yields Correct Sub-region Shape

*For any* ESB of shape `(C, H, W)` and any valid crop parameters `(row_start, row_end, col_start, col_end)` where `row_end > row_start` and `col_end > col_start` and all indices are in-bounds, the output ESB must have shape `(C, row_end - row_start, col_end - col_start)`.

**Validates: Requirements 2.3**

---

### Property 3: Mask Zeros Out Non-Matching Cells

*For any* ESB and any boolean condition tensor of matching spatial dimensions, applying `mask(condition)` must produce an ESB where every cell at position `(c, r, col)` where `condition[r, col]` is `False` equals 0, and every cell where `condition[r, col]` is `True` retains its original value.

**Validates: Requirements 2.4**

---

### Property 4: Rotation Round-Trip (4 × 90°)

*For any* ESB, applying `rotate(90)` four times in succession must produce an ESB with data element-wise identical to the original ESB.

**Validates: Requirements 3.2**

---

### Property 5: Reflection Involution

*For any* ESB and any axis in `{"horizontal", "vertical"}`, applying `reflect(axis)` twice must produce an ESB with data element-wise identical to the original ESB.

**Validates: Requirements 3.3**

---

### Property 6: Flood Fill Connectivity

*For any* ESB with a connected region of uniform color `C` anchored at `(row, col)`, applying `flood_fill(row, col, new_color)` where `new_color ≠ C` must produce an ESB where every cell reachable from `(row, col)` via 4-connected cells of color `C` now equals `new_color`, and all other cells are unchanged.

**Validates: Requirements 3.4**

---

### Property 7: Color Map Bijective Round-Trip

*For any* ESB and any bijective color mapping `M: {0..9} → {0..9}`, applying `color_map(M)` followed by `color_map(M⁻¹)` must produce an ESB element-wise identical to the original ESB.

**Validates: Requirements 3.5**

---

### Property 8: Overlay Compositing Correctness

*For any* background ESB and foreground ESB of the same shape and any `mask_value`, applying `overlay(foreground, mask_value)` must produce an ESB where every cell at position `p` where `foreground.data[p] ≠ mask_value` equals `foreground.data[p]`, and every other cell equals `background.data[p]`.

**Validates: Requirements 3.6**

---

### Property 9: DSL Program Sequential Composition

*For any* ESB and any DSL program (ordered list of primitive calls), applying the full program via `apply_program` must produce the same result as manually applying each primitive call in sequence using direct method calls on the same starting ESB.

**Validates: Requirements 3.7**

---

### Property 10: HPM Embeddings Lie on the Poincaré Ball

*For any* DSL program added to the HPM, its computed embedding must have Euclidean norm strictly less than 1 (i.e., lie inside the open unit ball).

**Validates: Requirements 4.1**

---

### Property 11: HPM k-NN Ordering and Prior Validity

*For any* HPM with at least two entries at different hyperbolic distances from a query point, the entry with smaller hyperbolic distance to the query must receive a strictly higher prior probability weight than the entry with larger distance. Furthermore, for any query, the returned prior weights must all be positive and sum to approximately 1.0 (within floating-point tolerance).

**Validates: Requirements 4.4, 4.5**

---

### Property 12: HPM Serialization Round-Trip

*For any* HPM state (any number of stored programs with embeddings), serializing to a file and then loading from that file must produce an HPM with the same number of entries, the same programs in the same order, and embeddings element-wise identical (within floating-point tolerance) to the originals.

**Validates: Requirements 4.7**

---

### Property 13: UCB1 Selection Picks Maximum Score

*For any* set of MCTS child nodes with known `visits`, `value`, and `prior` values, the selection step must return the child node with the strictly highest `UCB1 = Q/N + C * sqrt(ln(parent_N)/N) + prior` score.

**Validates: Requirements 5.1, 5.2**

---

### Property 14: Rollout Depth Bounded by 10

*For any* MCTS rollout starting from any node, the number of DSL primitive applications performed during the rollout must not exceed `MAX_ROLLOUT_DEPTH = 10`.

**Validates: Requirements 5.4**

---

### Property 15: Backpropagation Formula Correctness

*For any* MCTS tree path from a terminal node to the root, after backpropagating reward `r`, each ancestor node along the path must have its `visits` incremented by exactly 1 and its `value` updated by the formula `Q_new = Q_old + (r - Q_old) / N_new` where `N_new` is the updated visit count.

**Validates: Requirements 5.6**

---

### Property 16: Exact Match Identity

*For any* ESB `g`, `exact_match(g, g)` must return `True`. For any two ESBs `g1` and `g2` with different shapes, `exact_match(g1, g2)` must return `False`. For any two ESBs with the same shape but at least one differing cell value, `exact_match(g1, g2)` must return `False`.

**Validates: Requirements 6.1**

---

### Property 17: Partial Reward Range and Correctness

*For any* candidate ESB and target ESB of the same shape, `partial_reward(candidate, target)` must return a value in the closed interval `[0.0, 1.0]` and must equal exactly `(number of cells where candidate == target) / (total cells in target)`.

**Validates: Requirements 6.3**

---

### Property 18: Extract Object Isolation

*For any* ESB and any seed coordinate `(x, y)` with a non-zero color `c`, applying `extract_object(x, y)` must produce an ESB where every cell in the 4-connected region of color `c` reachable from `(y, x)` retains its original value, and all other cells equal 0.

**Validates: Requirements 3.6**

---

### Property 19: Crop-to-BBox Minimality

*For any* ESB with at least one non-zero cell, applying `crop_to_bbox()` must produce an ESB where the first and last row and the first and last column each contain at least one non-zero cell (i.e., no all-zero border rows or columns remain).

**Validates: Requirements 3.8**

---

### Property 20: Scale Integer Upscale Shape

*For any* ESB of shape `(C, H, W)` and any integer upscale `factor ∈ {2, 3, 4}`, applying `scale_integer(factor)` must produce an ESB of shape `(C, H*factor, W*factor)`.

**Validates: Requirements 3.9**

---

### Property 21: Tile Shape

*For any* ESB of shape `(C, H, W)` and any `(repeats_h, repeats_w)`, applying `tile(repeats_h, repeats_w)` must produce an ESB of shape `(C, H*repeats_h, W*repeats_w)`.

**Validates: Requirements 3.10**

---

### Property 22: Symmetrize Idempotency

*For any* ESB and any `(axis, mode)`, applying `symmetrize(axis, mode)` twice in succession must produce an ESB element-wise identical to the result of applying it once (i.e., the operation is idempotent).

**Validates: Requirements 3.14**

---

### Property 23: Color Remap Lookup Correctness

*For any* ESB and any color mapping `M`, applying `color_remap(M)` must produce an ESB where every cell with color `src ∈ M` now equals `M[src]`, and all other cells are unchanged.

**Validates: Requirements 3.12**

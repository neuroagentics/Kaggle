# Product: Hyper-ARC Neuro-Symbolic Solver

Hyper-ARC is a Python-based solver for the ARC-AGI-2 benchmark (Kaggle competition `arc-prize-2025`). It treats every ARC task as a program synthesis problem: find a sequence of spatial DSL primitives that maps training-pair inputs to their outputs, then apply that program to the test inputs to generate a `submission.json`.

## Core Idea

The system combines three components:
- **SpatialDSL** — 12 pure-tensor, zero-neural transformation primitives (translate, rotate, reflect, flood_fill, etc.)
- **Hyperbolic Program Memory (HPM)** — stores successful programs embedded on a Poincaré Ball; biases future search via hyperbolic k-NN retrieval
- **Memory-guided MCTS** — UCB1 tree search over DSL program space, with HPM priors steering rollouts toward high-quality regions

## Deployment Context

Runs in a Kaggle Notebook environment. The entry point is `main.py` (invoked as `!python main.py`). It must survive Kaggle's 9-hour session limit via checkpoint/resume logic.

## Key Constraints

- All DSL primitives are **pure deterministic tensor math** — zero learned weights, no `nn.Module`, no gradients
- Grids are ARC color integers 0–9; ESB tensors are always `torch.int64`
- Output must be a `submission.json` mapping task IDs to predicted output grids

# Requirements Document

## Introduction

Hyper-ARC is a neuro-symbolic solver for the ARC-AGI-2 benchmark. The system combines a Spatial Domain-Specific Language (DSL) with a Hyperbolic Program Memory and a Memory-guided Monte Carlo Tree Search (MCTS) engine to discover transformation programs that map input grids to output grids. The solver downloads ARC-AGI-2 data via the Kaggle CLI, processes training/test pairs, and produces a `submission.json` in the Kaggle ARC submission format.

## Glossary

- **ARC-AGI-2**: The ARC (Abstraction and Reasoning Corpus) AGI-2 benchmark dataset hosted on Kaggle.
- **Grid**: A 2-D array of integer color values (0–9) representing an ARC task input or output.
- **Task**: A named ARC problem consisting of one or more training pairs and one or more test pairs.
- **Training Pair**: A `(input_grid, output_grid)` example used to infer the transformation rule.
- **Test Pair**: A `(input_grid, output_grid)` example on which the solver must predict the output.
- **DSL Primitive**: An atomic, parameterized transformation operation in the `SpatialDSL` (e.g., `translate`, `rotate`).
- **DSL Program**: An ordered sequence of DSL Primitives that transforms an input Grid into an output Grid.
- **Eidetic Spatial Buffer (ESB)**: A PyTorch tensor of shape `(C, H, W)` that holds the current intermediate grid state during program execution.
- **SpatialDSL**: The module implementing the twelve canonical DSL Primitives applied to the ESB, located in `hyper_arc/spatial_dsl.py`.
- **Connected Component Labeling (CCL)**: An algorithm that assigns unique integer labels to spatially contiguous regions of uniform color in a grid.
- **Bounding Box**: The minimal axis-aligned rectangle that contains all non-zero (non-background) cells of an object in the grid.
- **Hyperbolic Program Memory (HPM)**: A Poincaré Ball embedding store that holds DSL Programs, supports k-NN retrieval, and is updated online when new programs succeed.
- **MCTS Engine**: The Memory-guided Monte Carlo Tree Search component that selects, expands, simulates, and backpropagates over DSL action sequences.
- **MCTS Node**: A data structure that tracks `state` (grid tensor), `parent`, `children`, `action_taken`, `visits` (N), and `value` (Q).
- **UCB1 Score**: Upper Confidence Bound score used for MCTS node selection: `Q/N + C * sqrt(ln(parent_N) / N) + prior`.
- **Prior Probability**: A scalar weight derived from k-NN retrieval in the HPM that biases UCB1 selection toward promising DSL Primitives.
- **Exact Match**: A boolean indicating that a candidate output Grid is element-wise identical to the target output Grid.
- **Submission**: The `submission.json` file in Kaggle ARC format containing predicted output grids for all test pairs.
- **Kaggle CLI**: The `kaggle` command-line tool used to download competition datasets.
- **Seed Bank**: A pre-built collection of embedded DSL Programs used to initialize the HPM before inference begins.

---

## Requirements

### Requirement 1 — Data Acquisition

**User Story:** As a solver operator, I want to download ARC-AGI-2 data using my Kaggle credentials, so that the solver always starts from the authoritative dataset.

#### Acceptance Criteria

1. THE Solver SHALL provide a standalone `download_data.py` script that downloads the ARC-AGI-2 dataset using the `kaggle` CLI.
2. WHEN `download_data.py` is executed, THE Solver SHALL place the downloaded dataset files under a configurable local directory (default: `./data/arc-agi-2/`).
3. IF the `kaggle` CLI is not installed or Kaggle credentials are missing, THEN THE Solver SHALL print an actionable error message describing the missing prerequisite and exit with a non-zero return code.
4. IF the target data directory already contains the dataset, THEN THE Solver SHALL skip re-downloading and log a message indicating the dataset is already present.

---

### Requirement 2 — Eidetic Spatial Buffer

**User Story:** As a DSL execution engine, I want a tensor-backed grid buffer with core spatial operations, so that all grid transformations are performed on a uniform, type-safe representation.

#### Acceptance Criteria

1. THE ESB SHALL represent every Grid as a `torch.Tensor` of dtype `torch.int64` and shape `(C, H, W)` where `C` is the number of color channels.
2. THE ESB SHALL implement a `pad(top, bottom, left, right, fill_value)` operation that returns a new ESB with the Grid extended by the specified number of cells on each side, filled with `fill_value`.
3. THE ESB SHALL implement a `crop(row_start, row_end, col_start, col_end)` operation that returns a new ESB containing the specified sub-region.
4. THE ESB SHALL implement a `mask(condition_tensor)` operation that returns a new ESB where cells satisfying `condition_tensor` retain their values and all other cells are set to 0.
5. THE ESB SHALL implement a `shift(delta_row, delta_col, fill_value)` operation that translates the Grid content by `(delta_row, delta_col)` cells and fills vacated cells with `fill_value`.
6. IF an ESB operation receives parameters that would produce a Grid with any dimension less than 1, THEN THE ESB SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 3 — Spatial DSL

**User Story:** As the MCTS Engine, I want a composable set of spatial transformation primitives, so that DSL Programs can be constructed and executed deterministically on any Grid.

#### Acceptance Criteria

1. THE SpatialDSL SHALL be implemented in `hyper_arc/spatial_dsl.py` as a stateless class where every method is a pure, deterministic tensor operation with no learned weights or neural layers.
2. ALL SpatialDSL primitives SHALL accept tensors with a batch dimension `(B, C, H, W)` or the standard `(C, H, W)` ESB shape, applying the transformation uniformly across the batch.
3. THE SpatialDSL SHALL implement a `translate(dx, dy, wrap=False)` primitive that shifts the Grid content by `dx` columns and `dy` rows; WHEN `wrap=False` (default), vacated cells SHALL be filled with 0; WHEN `wrap=True`, content SHALL wrap using `torch.roll`.
4. THE SpatialDSL SHALL implement a `rotate(degrees)` primitive that rotates the Grid clockwise by `degrees` (constrained to {90, 180, 270}).
5. THE SpatialDSL SHALL implement a `reflect(axis)` primitive that mirrors the Grid along the specified `axis` (constrained to `{"horizontal", "vertical"}`).
6. THE SpatialDSL SHALL implement an `extract_object(x, y)` primitive that isolates the connected component at coordinates `(x, y)` using Connected Component Labeling and masks all other cells to 0 (background).
7. THE SpatialDSL SHALL implement an `overlay(foreground_grid, blend_mode)` primitive that composites a `foreground_grid` ESB on top of the current ESB; WHEN `blend_mode="overwrite"` (default), non-zero foreground cells replace background cells; WHEN `blend_mode="or"`, cells are combined using logical OR.
8. THE SpatialDSL SHALL implement a `crop_to_bbox()` primitive that trims all background (0) padding from the edges of the grid and returns the minimal bounding box containing all non-zero cells.
9. THE SpatialDSL SHALL implement a `scale_integer(factor)` primitive that upscales the grid by a positive integer `factor` using `torch.repeat_interleave`; for `factor < 1` (downscale), mode-pooling SHALL be used; `factor` SHALL be constrained to {0.5, 1, 2, 3, 4}.
10. THE SpatialDSL SHALL implement a `tile(repeats_h, repeats_w)` primitive that repeats the grid `repeats_h` times vertically and `repeats_w` times horizontally using `torch.tile`.
11. THE SpatialDSL SHALL implement a `connect_points(color, row1, col1, row2, col2)` primitive that draws a 1-pixel-wide line between coordinates `(row1, col1)` and `(row2, col2)` using Bresenham's line algorithm, assigning `color` to all cells on the line.
12. THE SpatialDSL SHALL implement a `color_remap(mapping)` primitive that replaces each cell color `k` with `mapping[k]` using a 10-element lookup tensor (`lookup[grid]`); colors absent from `mapping` SHALL remain unchanged; `mapping` SHALL accept a `dict[int, int]`.
13. THE SpatialDSL SHALL implement a `flood_fill(row, col, new_color)` primitive that replaces all connected cells of the same color as `(row, col)` with `new_color` using 4-connectivity.
14. THE SpatialDSL SHALL implement a `symmetrize(axis, mode)` primitive that forces the grid to be symmetric about the specified `axis` (constrained to `{"horizontal", "vertical"}`); WHEN `mode="copy"`, one half is copied over the other; WHEN `mode="priority"`, non-zero cells take precedence over zero cells when resolving conflicts.
15. WHEN a DSL Program is applied to an input Grid, THE SpatialDSL SHALL apply each DSL Primitive in sequence via `apply_program` and return the resulting output Grid as an ESB.
16. IF a DSL Primitive receives a parameter value outside its defined domain, THEN THE SpatialDSL SHALL raise a `ValueError` specifying the invalid parameter and its acceptable range.
17. THE SpatialDSL SHALL expose a `list_primitives()` method that returns the names and parameter schemas of all 12 implemented DSL Primitives.
18. ALL `translate` and `rotate` operations SHALL default to padding with 0 (ARC background color) when `wrap=False`, never wrapping by default.

---

### Requirement 4 — Hyperbolic Program Memory

**User Story:** As the MCTS Engine, I want to retrieve previously successful DSL Programs that are semantically similar to the current search state, so that prior knowledge biases search toward high-quality regions of program space.

#### Acceptance Criteria

1. THE HPM SHALL embed DSL Programs as points on a Poincaré Ball manifold using the `geoopt` library.
2. THE HPM SHALL be initialized with a pre-built Seed Bank of embedded DSL Programs before any inference begins.
3. WHEN a DSL Program is successfully applied (Exact Match achieved), THE HPM SHALL add the embedding of that program to its store (online update).
4. WHEN the MCTS Engine queries the HPM with a partial DSL Program sequence, THE HPM SHALL return the `k` nearest neighbor embeddings along with their associated prior probability weights, where `k` is a configurable parameter (default: `k=5`).
5. THE HPM SHALL compute prior probability weights using hyperbolic distance: programs with smaller hyperbolic distance to the query SHALL receive higher prior probability weights.
6. IF the HPM store contains fewer than `k` entries at query time, THEN THE HPM SHALL return all available entries without error.
7. THE HPM SHALL support serialization to and deserialization from a file path so that the memory state can be persisted and reloaded across runs.

---

### Requirement 5 — MCTS Engine

**User Story:** As the solver core, I want a Memory-guided MCTS that searches over DSL Program space, so that the best transformation program for each ARC Task is found efficiently.

#### Acceptance Criteria

1. THE MCTS Engine SHALL represent the search tree using MCTS Nodes, each storing `state` (ESB tensor), `parent`, `children`, `action_taken`, `visits` (N), and `value` (Q).
2. THE MCTS Engine SHALL perform node selection using UCB1 Score with an additive Prior Probability term retrieved from the HPM.
3. WHEN the MCTS Engine expands a node, THE MCTS Engine SHALL generate child nodes by applying each available DSL Primitive as a macro-step action, then use deterministic parameter search (micro-step) to enumerate valid parameter assignments for the chosen primitive.
4. WHEN the MCTS Engine simulates (rollout) from an unexpanded node, THE MCTS Engine SHALL apply memory-weighted random DSL Primitive selections until a terminal condition is reached or a maximum rollout depth of 10 steps is exceeded.
5. WHEN Exact Match is achieved during simulation or expansion, THE MCTS Engine SHALL record the DSL action sequence from root to that node and immediately terminate the search for the current Training Pair.
6. THE MCTS Engine SHALL backpropagate reward signals from terminal nodes to the root, incrementing `visits` by 1 and updating `value` by `(reward - Q) / N` at each ancestor node.
7. THE MCTS Engine SHALL run for a maximum of 2,000 iterations per Training Pair unless early-exit on Exact Match is triggered.
8. WHEN all Training Pairs for a Task have been solved (Exact Match on each), THE MCTS Engine SHALL return the discovered DSL Program sequence.
9. IF no Exact Match is found within the iteration budget, THE MCTS Engine SHALL return the DSL action sequence leading to the MCTS Node with the highest `value` (Q).
10. THE MCTS Engine SHALL accept a configurable exploration constant `C` (default: `C=1.41`) for the UCB1 formula.

---

### Requirement 6 — Exact-Match Cost Function

**User Story:** As the MCTS Engine, I want a precise measure of program quality on a Training Pair, so that the search correctly identifies when a transformation is fully correct.

#### Acceptance Criteria

1. THE Solver SHALL implement an `exact_match(candidate_grid, target_grid)` function that returns `True` if and only if the two Grids have identical shape and all corresponding cell values are equal.
2. WHEN `exact_match` returns `True` for all Training Pairs of a Task, THE MCTS Engine SHALL assign a reward of `1.0` to the terminal node.
3. WHEN `exact_match` returns `False`, THE Solver SHALL compute a partial reward as the fraction of correctly predicted cells divided by the total number of cells in the target Grid.

---

### Requirement 7 — Solver Orchestration

**User Story:** As a competition participant, I want a single entry-point script that processes all tasks and produces a valid submission file, so that the end-to-end pipeline is reproducible with one command.

#### Acceptance Criteria

1. THE Solver SHALL provide a `main.py` entry-point script that reads all ARC-AGI-2 test tasks from the data directory, runs the MCTS Engine on each task, and writes results to `submission.json`.
2. WHEN `main.py` is executed, THE Solver SHALL initialize the HPM with the Seed Bank before processing any Task.
3. WHEN `main.py` finishes processing all Tasks, THE Solver SHALL write `submission.json` to the working directory in the Kaggle ARC submission format: a JSON object mapping task IDs to lists of predicted output grids.
4. WHEN `main.py` completes successfully, THE Solver SHALL exit with return code 0.
5. IF an unhandled exception occurs during task processing, THEN THE Solver SHALL log the task ID and exception details, skip that task, and continue processing remaining tasks.
6. THE Solver SHALL log progress to stdout including task ID, iteration count, and whether Exact Match was achieved, for each processed task.

---

### Requirement 8 — Dependency and Packaging

**User Story:** As a developer or Kaggle notebook operator, I want a complete `requirements.txt` so that the environment can be reproduced exactly.

#### Acceptance Criteria

1. THE Solver SHALL provide a `requirements.txt` that lists all direct Python dependencies with pinned major and minor versions (e.g., `torch==2.3.*`).
2. THE `requirements.txt` SHALL include at minimum: `torch`, `geoopt`, `numpy`, `kaggle`, and `pytest`.
3. THE Solver SHALL be structured as a Python package with all source modules under a top-level `hyper_arc/` directory and `main.py` at the project root; the Spatial DSL SHALL reside in `hyper_arc/spatial_dsl.py`.

---

### Requirement 9 — Unit Tests for DSL Primitives

**User Story:** As a developer, I want automated unit tests for every DSL Primitive, so that regressions in core transformation logic are caught immediately.

#### Acceptance Criteria

1. THE Solver SHALL provide a `tests/` directory containing a `pytest`-compatible test module for the SpatialDSL.
2. WHEN `pytest tests/` is executed, THE Solver SHALL have at least one test case for each of the twelve DSL Primitives: `translate`, `rotate`, `reflect`, `extract_object`, `overlay`, `crop_to_bbox`, `scale_integer`, `tile`, `connect_points`, `color_remap`, `flood_fill`, and `symmetrize`.
3. WHEN `pytest tests/` is executed, THE Solver SHALL have at least one test case for each ESB operation: `pad`, `crop`, `mask`, and `shift`.
4. WHEN `pytest tests/` is executed, THE Solver SHALL have at least one test case for the `exact_match` function covering both matching and non-matching grid pairs.
5. WHEN `pytest tests/` is executed, THE Solver SHALL have at least one test case verifying that invalid DSL Primitive parameters raise `ValueError`.

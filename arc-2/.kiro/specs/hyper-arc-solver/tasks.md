# Implementation Plan: Hyper-ARC Neuro-Symbolic Solver

## Overview

Implement the Hyper-ARC solver in Python 3.10+, building each component incrementally: project scaffold → ESB → DSL → cost functions → HPM → MCTS engine → orchestration. Property-based tests (using `hypothesis`) and unit tests (using `pytest`) are embedded as sub-tasks alongside each component.

## Tasks

- [ ] 1. Scaffold project structure and packaging
  - Create `hyper_arc/` package directory with `__init__.py`
  - Create `tests/` directory with `__init__.py`
  - Write `requirements.txt` with pinned versions: `torch==2.3.*`, `geoopt==0.5.*`, `numpy==1.26.*`, `kaggle==1.6.*`, `pytest==8.2.*`, `hypothesis==6.*`
  - Write `hyper_arc/__init__.py` with all public re-exports (`ESB`, `SpatialDSL`, `DSLProgram`, `HyperbolicProgramMemory`, `GlobalMemoryBank`, `LocalTaskBuffer`, `MCTSEngine`, `exact_match`, `partial_reward`)
  - Note: `SpatialDSL` and `DSLProgram` live in `hyper_arc/spatial_dsl.py`
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 2. Implement data acquisition script
  - [ ] 2.1 Write `download_data.py`
    - Implement `download(dest)` using `subprocess` to invoke `kaggle competitions download -c arc-prize-2025 -p {dest} --unzip`
    - Skip download if `dest` directory already exists and is non-empty (log info message)
    - Catch missing/broken `kaggle` CLI, print actionable error, return exit code 1
    - Wire `__main__` block with `argparse` for `--dest`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 3. Implement Eidetic Spatial Buffer (`hyper_arc/esb.py`)
  - [ ] 3.1 Implement the `ESB` dataclass and all five operations
    - `ESB.from_grid(grid)` → `(1, H, W)` `torch.int64` tensor
    - `pad(top, bottom, left, right, fill_value)` → new ESB with extended shape
    - `crop(row_start, row_end, col_start, col_end)` → new ESB with sub-region
    - `mask(condition)` → new ESB zeroing cells where condition is False
    - `shift(delta_row, delta_col, fill_value)` → new ESB with same shape, translated content
    - `to_grid()` → 2-D list of ints from channel 0
    - Raise `ValueError` when any resulting dimension < 1
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.2 Write property tests for ESB (Property 1: ESB Shape Invariant)
    - **Property 1: ESB Shape Invariant**
    - Use `hypothesis` to generate arbitrary valid grids and pad/shift parameters
    - Assert `from_grid` produces `(1, H, W)` int64 tensor; pad produces `(C, H+top+bottom, W+left+right)`; shift preserves shape
    - **Validates: Requirements 2.1, 2.2, 2.5**

  - [ ]* 3.3 Write property tests for ESB (Property 2: Crop Yields Correct Sub-region Shape)
    - **Property 2: Crop Yields Correct Sub-region Shape**
    - Generate arbitrary ESBs and valid crop slices; assert output shape equals `(C, row_end-row_start, col_end-col_start)`
    - **Validates: Requirements 2.3**

  - [ ]* 3.4 Write property tests for ESB (Property 3: Mask Zeros Out Non-Matching Cells)
    - **Property 3: Mask Zeros Out Non-Matching Cells**
    - Generate arbitrary ESBs and boolean condition tensors; assert masked cells are 0 and unmasked cells retain original values
    - **Validates: Requirements 2.4**

  - [ ]* 3.5 Write unit tests for ESB (`tests/test_esb.py`)
    - One test case each for `pad`, `crop`, `mask`, `shift`, `from_grid`, `to_grid`
    - One test case verifying `ValueError` is raised for out-of-bounds crop and negative-dimension pad
    - _Requirements: 9.3, 9.5_

- [ ] 4. Implement Spatial DSL (`hyper_arc/spatial_dsl.py`)
  - [ ] 4.1 Implement `SpatialDSL` in `hyper_arc/spatial_dsl.py` with all 12 primitives
    - **CRITICAL CONSTRAINTS**: zero neural approximations; all ops are pure tensor math; support `(B,C,H,W)` batch dimension; default out-of-bounds padding is 0 (not wrap)
    - **Group 1 — Rigid Spatial:**
      - `translate(esb, dx, dy, wrap=False)` — `torch.roll` when `wrap=True`, `esb.shift` when `False`; validate `dx,dy ∈ [-2,2]`
      - `rotate(esb, degrees)` — clockwise via `torch.rot90(..., k=-degrees//90, dims=[1,2])`; validate `degrees ∈ {90,180,270}`
      - `reflect(esb, axis)` — flip dim 1 for "horizontal", dim 2 for "vertical"; validate `axis ∈ {"horizontal","vertical"}`
    - **Group 2 — Object Manipulation:**
      - `extract_object(esb, x, y)` — BFS CCL from `(row=y, col=x)` to identify connected component; mask all other cells to 0
      - `overlay(esb, foreground, blend_mode="overwrite")` — `blend_mode="overwrite"`: non-zero fg replaces bg; `blend_mode="or"`: logical OR; validate `blend_mode ∈ {"overwrite","or"}`
      - `crop_to_bbox(esb)` — `torch.nonzero` to find extents; slice to min bounding box; return input unchanged if all-zero
    - **Group 3 — Scaling & Patterns:**
      - `scale_integer(esb, factor)` — upscale via `repeat_interleave(factor, dim=1/2)`; downscale (factor=0.5) via `F.max_pool2d`; validate `factor ∈ {0.5,1,2,3,4}`
      - `tile(esb, repeats_h, repeats_w)` — `torch.tile(esb.data, (1, repeats_h, repeats_w))`
      - `connect_points(esb, color, row1, col1, row2, col2)` — Bresenham's line algorithm; assign `color` to all cells on the line; skip out-of-bounds cells silently
    - **Group 4 — Logic & Color:**
      - `color_remap(esb, mapping)` — build a 10-element `lookup` tensor (indices 0–9); assign `lookup[src]=dst` for each entry in `mapping`; apply via `lookup[esb.data.clamp(0,9)]`
      - `flood_fill(esb, row, col, new_color)` — BFS 4-connected flood fill; no-op if `new_color == current color`
      - `symmetrize(esb, axis, mode="copy")` — `mode="copy"`: flip and replace; `mode="priority"`: `torch.where(data != 0, data, flipped)`; validate `axis` and `mode`
    - `apply_program(esb, program)` — sequential dispatch by primitive name; raise `ValueError` for unknown names
    - `list_primitives()` — return static list of name+parameter-domain dicts for all 12 primitives
    - Define `PrimitiveCall = tuple[str, dict[str, Any]]` and `DSLProgram = list[PrimitiveCall]`
    - Raise `ValueError` for all out-of-domain parameters
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18_

  - [ ] 4.2 Write property tests for DSL rotations (Property 4: Rotation Round-Trip)
    - **Property 4: Rotation Round-Trip (4 × 90°)**
    - Generate arbitrary ESBs; apply `rotate(90)` four times; assert result is element-wise identical to original
    - **Validates: Requirements 3.2**

  - [ ] 4.3 Write property tests for DSL reflections (Property 5: Reflection Involution)
    - **Property 5: Reflection Involution**
    - Generate arbitrary ESBs and axis choices; apply `reflect(axis)` twice; assert result is element-wise identical to original
    - **Validates: Requirements 3.3**

  - [ ] 4.4 Write property tests for flood fill (Property 6: Flood Fill Connectivity)
    - **Property 6: Flood Fill Connectivity**
    - Construct ESBs with a known contiguous color region; apply `flood_fill`; assert entire 4-connected region is recolored and all other cells unchanged
    - **Validates: Requirements 3.4**

  - [ ] 4.5 Write property tests for color_map (Property 7: Color Map Bijective Round-Trip)
    - **Property 7: Color Map Bijective Round-Trip**
    - Generate arbitrary ESBs and bijective permutations of {0..9}; apply `color_map(M)` then `color_map(M⁻¹)`; assert result is element-wise identical to original
    - **Validates: Requirements 3.5**

  - [ ] 4.6 Write property tests for overlay (Property 8: Overlay Compositing Correctness)
    - **Property 8: Overlay Compositing Correctness**
    - Generate arbitrary same-shape background/foreground ESB pairs and mask_value; assert every cell where `fg != mask_value` equals fg value, all other cells equal background value
    - **Validates: Requirements 3.6**

  - [ ] 4.7 Write property tests for apply_program (Property 9: DSL Program Sequential Composition)
    - **Property 9: DSL Program Sequential Composition**
    - Generate short programs (1–4 primitives); apply via `apply_program`; assert result equals manually chaining the same calls in sequence
    - **Validates: Requirements 3.7**

  - [ ] 4.8 Write unit tests for all 12 DSL primitives (`tests/test_spatial_dsl.py`)
    - At least one test per primitive: `translate` (clip and wrap modes), `rotate`, `reflect`, `extract_object`, `overlay` (both blend modes), `crop_to_bbox`, `scale_integer`, `tile`, `connect_points`, `color_remap`, `flood_fill`, `symmetrize` (both modes)
    - One test verifying `ValueError` for: invalid `degrees`, invalid `axis`, invalid `blend_mode`, invalid `scale_integer factor`, invalid `symmetrize mode`
    - One test verifying batch dimension `(B, C, H, W)` passes through at least `rotate` and `reflect` without error
    - _Requirements: 9.2, 9.5_

  - [ ] 4.9 Write property tests for extract_object (Property 18: Extract Object Isolation)
    - **Property 18: Extract Object Isolation**
    - Construct ESBs with a known connected region at a known seed coordinate; apply `extract_object(x, y)`; assert all cells in the 4-connected region retain original values and all other cells equal 0
    - **Validates: Requirements 3.6**

  - [ ] 4.10 Write property tests for crop_to_bbox (Property 19: Crop-to-BBox Minimality)
    - **Property 19: Crop-to-BBox Minimality**
    - Generate arbitrary ESBs with at least one non-zero cell; apply `crop_to_bbox`; assert no all-zero border rows or columns remain in the output
    - **Validates: Requirements 3.8**

  - [ ] 4.11 Write property tests for scale_integer (Property 20: Scale Integer Upscale Shape)
    - **Property 20: Scale Integer Upscale Shape**
    - Generate arbitrary ESBs of shape `(C, H, W)` and `factor ∈ {2, 3, 4}`; assert output shape equals `(C, H*factor, W*factor)` and every source cell value appears `factor×factor` times in the corresponding block
    - **Validates: Requirements 3.9**

  - [ ] 4.12 Write property tests for tile (Property 21: Tile Shape)
    - **Property 21: Tile Shape**
    - Generate arbitrary ESBs and `(repeats_h, repeats_w)` pairs; assert output shape equals `(C, H*repeats_h, W*repeats_w)`
    - **Validates: Requirements 3.10**

  - [ ] 4.13 Write property tests for symmetrize (Property 22: Symmetrize Idempotency)
    - **Property 22: Symmetrize Idempotency**
    - Generate arbitrary ESBs and `(axis, mode)` combinations; apply `symmetrize` twice; assert result is element-wise identical to applying it once
    - **Validates: Requirements 3.14**

  - [ ] 4.14 Write property tests for color_remap (Property 23: Color Remap Lookup Correctness)
    - **Property 23: Color Remap Lookup Correctness**
    - Generate arbitrary ESBs and color mappings `M`; apply `color_remap(M)`; assert every cell with color `src ∈ M` now equals `M[src]`, and all other cells are unchanged
    - **Validates: Requirements 3.12**

- [ ] 5. Implement cost functions (`hyper_arc/cost.py`)
  - [ ] 5.1 Implement `exact_match` and `partial_reward`
    - `exact_match(candidate, target)` → `True` iff same shape and all values equal
    - `partial_reward(candidate, target)` → fraction of matching cells (0.0 if shapes differ)
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 5.2 Write property tests for cost functions (Property 16: Exact Match Identity)
    - **Property 16: Exact Match Identity**
    - For any ESB `g`: `exact_match(g, g)` is `True`; for different shapes: `False`; for same shape with ≥1 differing cell: `False`
    - **Validates: Requirements 6.1**

  - [ ]* 5.3 Write property tests for cost functions (Property 17: Partial Reward Range and Correctness)
    - **Property 17: Partial Reward Range and Correctness**
    - Generate arbitrary same-shape candidate/target ESB pairs; assert `partial_reward` ∈ `[0.0, 1.0]` and equals exact cell-match fraction
    - **Validates: Requirements 6.3**

  - [ ]* 5.4 Write unit tests for cost functions (`tests/test_cost.py`)
    - Test `exact_match` on identical grids, mismatched shapes, and same-shape with one differing cell
    - Test `partial_reward` on full match (1.0), no match (0.0), and partial match
    - _Requirements: 9.4_

- [ ] 6. Checkpoint — Ensure all tests pass
  - Run `pytest tests/` to verify ESB, DSL, and cost function tests all pass. Ask the user if any questions arise before proceeding to HPM.

- [ ] 7. Implement Hyperbolic Program Memory (`hyper_arc/hpm.py`)
  - [ ] 7.1 Implement `_encode_program`, `HPMEntry`, `GlobalMemoryBank`, and `LocalTaskBuffer`
    - `_encode_program(program)` → mean-pooled one-hot vector in R^16
    - `_project(raw)` → expmap0 projection onto Poincaré Ball (norm < 1)
    - Implement `GlobalMemoryBank` class — read-only during inference, loaded from `seed_bank.json` at startup; never updated during inference
      - `add(program)` → encode, project, append `HPMEntry` (used during seed bank construction only)
      - `query(partial_program)` → hyperbolic k-NN, return `(program, weight)` pairs from the seed bank with weights summing to ~1.0
      - `load_seed_bank(path)` → parse JSON and call `add` for each entry
      - `save(path)` / `load(path)` → JSON serialization round-trip
      - Return empty list (no error) when store has fewer than `k` entries
    - Implement `LocalTaskBuffer` class — read/write, instantiated fresh for every new test task; updated when programs succeed
      - `add(program)` → encode, project, append `HPMEntry`
      - `query(partial_program)` → hyperbolic k-NN, return `(program, weight)` pairs from current task's discovered programs with weights summing to ~1.0
      - Return empty list (no error) when store has fewer than `k` entries
    - Both classes share the same `add(program)` / `query(partial_program)` interface
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 7.2 Write property tests for HPM embeddings (Property 10: HPM Embeddings Lie on the Poincaré Ball)
    - **Property 10: HPM Embeddings Lie on the Poincaré Ball**
    - Generate arbitrary DSL programs; add to both `GlobalMemoryBank` and `LocalTaskBuffer`; assert every stored embedding has Euclidean norm < 1
    - **Validates: Requirements 4.1**

  - [ ]* 7.3 Write property tests for HPM k-NN (Property 11: HPM k-NN Ordering and Prior Validity)
    - **Property 11: HPM k-NN Ordering and Prior Validity**
    - Populate both memory classes with known programs at different distances; query each; assert closer entry has strictly higher weight; assert all weights positive and sum ≈ 1.0
    - **Validates: Requirements 4.4, 4.5**

  - [ ]* 7.4 Write property tests for HPM serialization (Property 12: HPM Serialization Round-Trip)
    - **Property 12: HPM Serialization Round-Trip**
    - Generate arbitrary program sets; save `GlobalMemoryBank` to a temp file and reload; assert same count, same programs in order, embeddings element-wise identical within float tolerance
    - **Validates: Requirements 4.7**

  - [ ]* 7.5 Write unit tests for HPM (`tests/test_hpm.py`)
    - Test `GlobalMemoryBank.add` + `query` returns up to `k` results
    - Test `LocalTaskBuffer.add` + `query` returns up to `k` results
    - Test `query` on empty store returns `[]` for both classes
    - Test `GlobalMemoryBank.load_seed_bank` populates from JSON file
    - Test `GlobalMemoryBank.save`/`load` round-trip preserves program content
    - _Requirements: 4.2, 4.3, 4.6, 4.7_

- [ ] 8. Implement MCTS Engine (`hyper_arc/mcts.py`)
  - [ ] 8.1 Implement `MCTSNode` and `UCB1` scoring
    - Dataclass fields: `state`, `parent`, `action_taken`, `children`, `visits`, `value`
    - `ucb1(C, prior)` formula: `value + C * sqrt(ln(parent.visits) / visits) + prior`; return `inf` when `visits == 0`
    - _Requirements: 5.1, 5.2_

  - [ ]* 8.2 Write property tests for UCB1 selection (Property 13: UCB1 Selection Picks Maximum Score)
    - **Property 13: UCB1 Selection Picks Maximum Score**
    - Construct MCTS trees with known visits/values/priors; assert `_select` returns child with maximum UCB1 score
    - **Validates: Requirements 5.1, 5.2**

  - [ ] 8.3 Implement `enumerate_actions` action space for all 12 DSL primitives
    - `translate`: `dx, dy ∈ [-2,2] × [-2,2]` excluding (0,0); `wrap ∈ {False, True}`
    - `rotate`: `degrees ∈ {90, 180, 270}`
    - `reflect`: `axis ∈ {"horizontal", "vertical"}`
    - `extract_object`: seed points (corners + center of grid)
    - `overlay`: `blend_mode ∈ {"overwrite", "or"}` (foreground is the current best partial state)
    - `crop_to_bbox`: no parameters — always emit one action
    - `scale_integer`: `factor ∈ {2, 3, 4}`
    - `tile`: `(repeats_h, repeats_w) ∈ {2,3}×{2,3}`
    - `connect_points`: corner-pair combinations; `color ∈ colors_present`
    - `color_remap`: single-color swaps `{src: dst}` for `src ∈ colors_present`, `dst ∈ 0–9`, `src≠dst`
    - `flood_fill`: seed points (corners + center) × colors 0–9 (skip same-color)
    - `symmetrize`: `axis ∈ {"horizontal","vertical"}` × `mode ∈ {"copy","priority"}`
    - _Requirements: 5.3_

  - [ ] 8.4 Implement `MCTSEngine` with all four MCTS phases and dual-memory prior
    - `MCTSEngine.__init__` accepts both `global_memory: GlobalMemoryBank` and `local_memory: LocalTaskBuffer`
    - `_select(root)` — traverse using `ucb1` until a childless leaf
    - `_expand(node)` — call `enumerate_actions`, apply each via DSL, create child nodes (skip on `ValueError`)
    - `_rollout(node, target)` — memory-weighted random walk up to `MAX_ROLLOUT_DEPTH=10`; return `(reward, program|None)`
    - `_backprop(node, reward)` — walk to root incrementing visits and updating value
    - `solve(training_pairs)` — run up to `MAX_ITERATIONS=2000`; early-exit on exact match; return best program
    - On exact match: call `local_memory.add(program)` before returning
    - **Dual-memory selection policy** when computing priors:
      - Query `local_memory` (LocalTaskBuffer) first — assign weight multiplier 0.8
      - Query `global_memory` (GlobalMemoryBank) as fallback — assign weight multiplier 0.2
      - Combine and normalize the results before passing as prior to UCB1
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

  - [ ]* 8.5 Write property tests for rollout depth (Property 14: Rollout Depth Bounded by 10)
    - **Property 14: Rollout Depth Bounded by 10**
    - Instrument rollout to count primitive applications; run over arbitrary starting nodes; assert count ≤ 10
    - **Validates: Requirements 5.4**

  - [ ]* 8.6 Write property tests for backpropagation (Property 15: Backpropagation Formula Correctness)
    - **Property 15: Backpropagation Formula Correctness**
    - Build small trees with known initial Q/N; call `_backprop` with a reward; assert each ancestor's visits incremented by 1 and value updated by `Q_old + (r - Q_old) / N_new`
    - **Validates: Requirements 5.6**

- [ ] 9. Checkpoint — Ensure all tests pass
  - Run `pytest tests/` to verify HPM and MCTS tests all pass. Ask the user if any questions arise before proceeding to orchestration.

- [ ] 10. Implement solver orchestration (`main.py`)
  - [ ] 10.1 Write `main.py` entry point with Kaggle compatibility and checkpointing
    - **CLI arguments** (all optional, with auto-detected Kaggle defaults):
      - `--data-dir`: ARC task directory (default: `/kaggle/input/arc-prize-2025` if running on Kaggle, else `./data/arc-agi-2`)
      - `--output`: submission file path (default: `/kaggle/working/submission.json` on Kaggle, else `./submission.json`)
      - `--checkpoint`: checkpoint file path (default: `/kaggle/working/checkpoint.pt` on Kaggle, else `./checkpoint.pt`)
      - `--seed-bank`: seed bank path (default: `hyper_arc/seed_bank.json`)
    - **Kaggle detection**: check `Path("/kaggle/working/").exists()` at startup; if true, use Kaggle default paths unless overridden via CLI
    - **Seed bank loading — MVP graceful fallback (REQUIRED)**:
      - Wrap seed bank loading in a `try/except` block
      - If `seed_bank.json` exists and loads cleanly: populate `GlobalMemoryBank` normally, use `global_prior_weight = 0.2`
      - If `seed_bank.json` is missing OR loading raises any exception: initialize an empty `GlobalMemoryBank`, set `global_prior_weight = 0.0`, log `[WARN] seed_bank.json not found — GlobalMemoryBank empty, using uniform prior`
      - `build_global_memory.py` is a separate offline utility; its absence must NEVER block `main.py` from running
      - With `global_prior_weight = 0.0`: MCTS relies 100% on `LocalTaskBuffer` priors and uniform random fallback
    - **Checkpoint resume logic** (run before processing any tasks):
      - If checkpoint file exists: load with `torch.load`; restore `submission` dict, `completed_task_ids` set, and `GlobalMemoryBank` state; log how many tasks are being skipped
      - If checkpoint load fails for any reason: log warning, start fresh (do not raise)
    - **Task loop**:
      - Skip any `task_id` already in `completed_task_ids`
      - For each task: instantiate a fresh `LocalTaskBuffer`; build `train_pairs`; call `engine.solve`; apply program to test inputs; collect predictions
      - After each task completes (success or skip): save checkpoint with `torch.save({"submission": submission, "completed_ids": completed_task_ids, "global_memory_state": global_memory.state_dict_entries()}, checkpoint_path)`
    - **Logging**: all `print()` calls use `flush=True` for live Kaggle Notebook cell output; log format: `[TASK] {task_id} exact_match={bool} program_len={int} ({n}/{total})`
    - **Shutdown**: after writing final `submission.json`, delete checkpoint file and log clean completion; exit code 0
    - Catch per-task exceptions: log task ID + traceback, set `submission[task_id] = []`, save checkpoint, continue
    - `load_tasks(data_dir)` — glob `**/*.json`, skip `seed_bank.json`, parse each, return `{task_id: task_dict}`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12_

- [ ] 11. Build Global Memory Seed Bank (`build_global_memory.py`)
  - **This is a standalone offline utility. It is NOT a dependency of `main.py`. Run it separately on Kaggle Notebooks after the MVP is working.**
  - [ ] 11.1 Write `build_global_memory.py` offline generation script
    - Accept `--data-dir` (path to ARC training data), `--n-tasks` (default: 50), and `--output` (default: `hyper_arc/seed_bank.json`) command-line arguments via `argparse`
    - Load a random subset of `--n-tasks` ARC training tasks from `--data-dir`
    - For each task, run a shallow MCTS (budget: 500 iterations) using a temporary `LocalTaskBuffer` to find winning DSL programs
    - For each successfully solved task, embed the winning program into hyperbolic space using `GlobalMemoryBank.add(program)` (which uses geoopt internally)
    - Export all embeddings and program ASTs to `--output` in the format expected by `GlobalMemoryBank.load_seed_bank`: `[{"program": [{"name": ..., "kwargs": {...}}]}, ...]`
    - Log to stdout: tasks attempted, tasks solved, programs embedded
    - Catch per-task exceptions: log task ID + error, continue to next task
    - Exit with code 0 on success
    - _Requirements: 4.2_

- [ ] 12. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ -v` to confirm the full test suite passes end-to-end. Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; **DSL property tests (4.2–4.14) are MANDATORY**
- All property tests use the `hypothesis` library; add it to `requirements.txt` alongside `pytest`
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests (Properties 1–23) validate universal correctness guarantees from the design
- Unit tests validate specific examples and edge cases

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1"] },
    { "id": 1, "tasks": ["3.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "3.4", "3.5"] },
    { "id": 3, "tasks": ["4.1", "5.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "4.11", "4.12", "4.13", "4.14", "5.2", "5.3", "5.4"] },
    { "id": 5, "tasks": ["7.1", "8.1", "8.3"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4", "7.5", "8.2", "8.4"] },
    { "id": 7, "tasks": ["8.5", "8.6", "11.1"] },
    { "id": 8, "tasks": ["10.1"] }
  ]
}
```

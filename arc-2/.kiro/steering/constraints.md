# Project Constraints & Strict Rules

## Strict Mathematical Constraints

### 1. Zero Neural Approximations in DSL
All `SpatialDSL` primitives MUST be pure, deterministic PyTorch tensor operations.
- NEVER use `nn.Module`, learned weights, or continuous float approximations for grid transformations
- No gradients, no autograd, no `requires_grad=True` on grid data

### 2. Out-of-Bounds Handling
For `translate` and `rotate`, default padding is `0` (ARC background color).
- Wrapping is only allowed when `wrap=True` is explicitly passed
- This applies to `ESB.shift` as well

### 3. Grid Tensor Data Type
> ⚠️ **Conflict with spec:** The design doc and requirements specify `torch.int64` for all ESB tensors.
> The rule below (preferring `int8`/`int16`) contradicts that. Resolve before implementing `esb.py`.

Prefer `torch.int8` or `torch.int16` for grid tensors to minimize VRAM on free-tier GPUs.
ARC colors are integers 0–9, so `int8` is sufficient for raw grid values.

### 4. Discrete DSL Parameters
MCTS operates over discrete spaces only. All DSL primitive parameters must be discrete and enumerable.
- Example: `rotate(degrees)` where `degrees ∈ {90, 180, 270}`, not a continuous angle
- See `enumerate_actions` in `mcts.py` for the full per-primitive parameter domains

---

## MCTS & Memory Rules

### 5. Missing Seed Bank — Graceful Fallback (MVP Default)
`build_global_memory.py` is a standalone offline utility. `main.py` MUST run correctly without `seed_bank.json` present.

In `main.py`, seed bank loading MUST be wrapped in a `try/except`:
- **Seed bank found and loads cleanly** → populate `GlobalMemoryBank` normally; use `global_prior_weight = 0.2`
- **Seed bank missing or load fails** → initialize empty `GlobalMemoryBank`; set `global_prior_weight = 0.0`; log `[WARN] seed_bank.json not found — GlobalMemoryBank empty, using uniform prior`; do not raise

With `global_prior_weight = 0.0`: MCTS relies 100% on `LocalTaskBuffer` priors (weight `1.0`) and uniform random fallback. The dual-memory prior formula in `MCTSEngine` must respect this weight at runtime.

### 6. Batch Dimension Support
All DSL operations must handle PyTorch batch dimensions `(B, C, H, W)` so MCTS can evaluate multiple hypotheses in parallel on GPU.
- Single-grid calls use `(C, H, W)`; batched calls use `(B, C, H, W)`
- Transformations must apply uniformly across the batch dimension

### 7. Reward Function Priority
The MCTS reward function must strictly prioritize exact pixel matches.
- Exact match → reward `1.0`, triggers immediate early exit
- Partial match (`partial_reward`) is used only as a rollout heuristic or tie-breaker
- Never use partial reward as the primary optimization target

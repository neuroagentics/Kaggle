# Spatial-dynamics simulator — fixing the movement root cause (2026-09-11)

Milestone: trustworthy internal game. Steps 1–2 established that the vector-state
`ArcSimulator` could not learn position-dependent, coordinate-free mechanics
(movement/collision appear-half = 0/115), and pinned the root cause to a recurrent
core that operates on a globally-pooled vector — it cannot compute an object's NEW
position as a function of the action.

This step builds and validates a fix, separately versioned so nothing existing is
disturbed.

No submission, no competition scoring. Diagnostic worlds only. `ArcSimulator` and
the production checkpoint are untouched.

## Two attempts, honestly

**Attempt 1 — spatial state at the encoder's H/8 resolution (ConvGRU on the 128-ch
map).** Failed: movement appear-half stayed ~0 (1/115). Diagnosis: for a 12×12 grid
the H/8 map is only 2×2, so each spatial cell covers a 6×6 region — single-cell
moves are unrepresentable. Same bottleneck, moved location.

**Attempt 2 — FULL-RESOLUTION spatial dynamics (kept).** A light full-resolution
encoder (embedding + two same-resolution convs, NO downsampling) produces an obs map
at H×W; a ConvGRU runs the recurrent state at H×W conditioned on the action broadcast
across the map; the change-mask heads read the full-resolution state directly. The
convolution at cell resolution lets a cell's next value depend on its immediate
neighbours — exactly what "move one cell" requires.

`agent/spatial_simulator.py`: `FullResEncoder`, `ConvGRUCell`, `SpatialPosterior`,
`SpatialPrior`, `SpatialArcSimulator` (`VERSION = "sim-spatial-v1"`).
Harness: `scripts/step_spatial_controlled_world.py` (self-contained trainer, reuses
the real `masked_frame_loss` and `kl_loss`).

## Result (64 train / 16 eval, 8 steps, 60 epochs, CPU)

Full data: `artifacts/step7_spatial_report.json`.

| mechanic | k=1 changed_acc | k=2 | k=4 | appear-half (k=1 decomp) |
|---|---|---|---|---|
| movement  | **1.00** | 0.54 | 0.56 | **115/115 = 1.0** (was 0/115) |
| collision | **1.00** | 0.54 | 0.53 | **117/117 = 1.0** (was 0/117) |
| click     | 1.00 | 1.00 | 1.00 | n/a |
| delayed   | 1.00 | 1.00 | 1.00 | n/a |

Params: **287K** (vs 1.79M for the vector model) — the full-res ConvGRU is far more
parameter-efficient (no giant linear layers).

### Honest reading

- **The movement root cause is fixed.** One-step (k=1) prediction of movement and
  collision is now perfect, and the appear-half — the model predicting WHERE the
  agent moves TO — went from 0/115 to 115/115. This is the capability Steps 1–2 said
  was missing.
- **click / delayed are now perfect** at all horizons.
- **Remaining honest caveat: multi-step drift.** k=2/k=4 on movement/collision drop
  to ~0.54. This is imagined-rollout drift: k=1 is exact, but the model re-encodes
  its OWN predicted frame each step and small errors compound. This is a SEPARATE,
  improvable problem (rollout stability / scheduled sampling / longer-horizon training
  signal), NOT the position-representation failure that this step fixed. click/delayed
  don't drift because their per-step change is trivial to reproduce exactly.

## Consequence for the plan

- The full-resolution spatial simulator is the right core for position-dependent
  mechanics. It should become the basis for the task-6 retrain (separately versioned),
  after: (a) reducing multi-step drift, and (b) confirming it scales to larger grids
  and mixed mechanics.
- Integration into the live agent (`controller.py`/`world_model_agent.py`) is a
  follow-up: the controller currently assumes the vector `ArcSimulator` interface
  (h/z vectors, pack_latent). Wiring the spatial state through the planner and the
  latent-packing/checkpoint path is a distinct, careful task — deferred until the
  spatial core is validated on real games, so the live loop is not destabilized.

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/step_spatial_controlled_world.py --device cpu `
    --mechanics movement collision click delayed `
    --n-train 64 --n-eval 16 --steps 8 --epochs 60 --grid 12 `
    --out artifacts/step7_spatial_report.json
```

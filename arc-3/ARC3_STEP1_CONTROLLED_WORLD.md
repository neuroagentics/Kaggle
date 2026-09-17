# Step 1 — Can the model learn a small controlled world? (2026-09-11)

Milestone: **trustworthy internal game**. Before spending GPU-hours on the real
ARC-3 games, this step tests whether the simulator can learn a world *at all* on
tiny environments whose transition function is exactly known — so a failure is
diagnosable (representation / loss / transition alignment), not blamed on "the game
is hard".

No submission, no scoring on the competition. Diagnostic worlds only. The trained
production checkpoint was not touched.

## What was built

`training/controlled_worlds.py` — four 12×12 worlds, one isolated mechanic each,
each with an EXACT known transition:

- **movement**: an agent cell moves one step in the commanded direction.
- **collision**: like movement, but a wall blocks the move (state unchanged).
- **click**: clicking the target cell (ACTION6 x,y) activates it (8→9); off-target
  clicks change nothing (a built-in false-change control).
- **delayed**: clicking a button opens a distant door exactly N steps LATER
  (temporal credit — the effect is not co-located in time with the trigger).

Trajectories are emitted in the SAME per-step schema the JSONL collector produces,
so the REAL trainer (`training/train_simulator.py`, with fix #7 imagined-rollout
loss and 5× changed-cell weighting) consumes them via
`training/dataset.InMemoryTrajectoryDataset` with zero special-casing.

Harness: `scripts/step1_controlled_world.py`. Metrics per horizon k=1/2/4, on
held-out trajectories (disjoint seed space), rolling the model forward on its OWN
prior predictions (genuine imagined futures, no teacher forcing):

- `frame_acc` — accuracy over all valid cells.
- `changed_acc` — accuracy on cells that ACTUALLY changed vs the anchor (the mechanic).
- `false_change_rate` — truly-unchanged cells the model WRONGLY changed (damage).
- `baseline_persist_acc` — a do-nothing "predict the anchor" baseline.

## Result (64 train / 16 eval trajectories, 8 steps, 60 epochs, CPU)

Full data: `artifacts/step1_report.json`.

| mechanic | frame_acc | changed_acc (k1/k2/k4) | false_change | persist_acc |
|---|---|---|---|---|
| movement | 0.993 | 0.50 / 0.50 / 0.50 | ~0.001–0.002 | 0.988–0.991 |
| collision | 0.986 | 0.50 / 0.50 / 0.50 | ~0.008–0.009 | 0.987–0.990 |
| click | 0.993 | **0.00 / 0.00 / 0.00** | ~0.005 | 0.998 |
| delayed | 0.986 | **0.00 / 0.00 / 0.00** | ~0.010–0.013 | 0.997 |

### Honest reading

- **High frame accuracy is a mirage.** In every world the changed cells are a tiny
  minority (16–150 out of ~11,500 valid cells). Predicting "nothing changes" already
  scores ~0.99 frame_acc and roughly ties the persistence baseline — while learning
  none of the mechanic.
- **Movement/collision = 0.5 changed_acc is not "half learned".** A focused probe
  decomposed the changed cells into `vanish` (agent's old cell → background) and
  `appear` (background → agent's new cell):
  - vanish: **115/115 = 100%** correct.
  - appear: **0/115 = 0%** correct.
  The model learns the trivial, direction-independent half ("the agent leaves its old
  spot") and NONE of the actual dynamics ("the agent moves one step in the commanded
  direction"). It cannot place the action-conditioned effect.
- **Click/delayed = 0.0.** The model never predicts the activated target or the opened
  door. It reproduces the static frame.
- **Not an epochs problem.** Collision converged in ~46s; movement plateaued at 0.5
  even with ~200s of training. More epochs will not move these numbers.

## Diagnosis

The simulator learns a frame's static structure and the easy "something disappeared"
signal, but not **action-conditioned localized change** — "given this action, WHICH
cell changes and to WHAT". The full-frame decoder is fed a globally-pooled recurrent
state (h pooled to a vector, small spatial skip); it has no straightforward pathway to
express "flip exactly this cell as a function of the action". The 5× changed-cell
weight is insufficient because the loss is still dominated by, and easily minimized on,
the unchanged majority.

## Consequence for the plan

This is exactly the representation/decoder weakness Step 2 targets. Next:

- **Step 2**: add a learned change-mask / localized color-update decoder head that
  predicts *where* the frame changes and *to what*, preserving unaffected cells,
  alongside the existing full-frame head (keep full-frame change possible; no
  hard-coded stationary-background assumption). Re-run this exact controlled-world
  test and require: movement/collision `changed_acc` (appear half) well above 0, and
  click/delayed `changed_acc` above 0, with `false_change_rate` staying low.

Only after Step 2 improves changed-cell learning on these controlled worlds does a
retrain on the real games (task #6) make sense.

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/step1_controlled_world.py --device cpu `
    --mechanics movement collision click delayed `
    --n-train 64 --n-eval 16 --steps 8 --epochs 60 --grid 12 `
    --out artifacts/step1_report.json
```

# Step 2 — Change-mask decoder vs full-redraw decoder (2026-09-11)

Milestone: trustworthy internal game. Step 1 showed the full-redraw decoder could
not learn action-conditioned localized change (click/delayed changed_acc=0.0;
movement learned only the "agent leaves its old cell" half). Step 2 adds a learned
change-mask / localized color-update decoder and re-runs the exact controlled-world
test.

No submission, no competition scoring. Diagnostic worlds only. Production checkpoint
untouched.

## What changed

`agent/simulator.py SimulatorDecoder` now emits, per cell:
- `update_logits` (B,16,H,W) — the new color if the cell changes,
- `change_gate` (B,1,H,W) — a learned per-cell change probability (sigmoid).

When the previous frame is supplied (`prev_tokens`), the final logits blend:

```
frame_logits = gate * update_logits + (1 - gate) * keep_logits
```

where `keep_logits` is a scaled one-hot of the PREVIOUS frame. A closed gate
reproduces the previous cell exactly; an open gate lets the model choose a new
color. "No change" is the default, so the model must actively decide WHERE and TO
WHAT to change. Full-frame change stays possible (the gate can open anywhere) — no
stationary-background assumption. When `prev_tokens` is omitted the decoder returns
pure `update_logits` (original full-redraw behaviour; kept for back-compat).

`prev_tokens` is threaded through `ArcSimulator.decode`, `predict()`, the training
loop, the imagined-rollout loss, and validation. In imagined rollouts the "previous
frame" is the model's OWN prior prediction (never the real intermediate frame), so
multi-step remains a genuine imagined future.

## Result (64 train / 16 eval, 8 steps, 60 epochs, CPU)

Full data: `artifacts/step2_report.json` (compare `artifacts/step1_report.json`).

| mechanic | changed_acc k1/k2/k4 — Step 1 | changed_acc k1/k2/k4 — Step 2 | false_change (k1) S1 → S2 |
|---|---|---|---|
| movement  | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.0007 → 0.0007 |
| collision | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.0075 → **0.0004** |
| click     | **0.00 / 0.00 / 0.00** | **0.69 / 0.72 / 0.74** | 0.0056 → **0.0003** |
| delayed   | **0.00 / 0.00 / 0.00** | 0.00 / 0.00 / **0.67** | 0.0126 → **0.0000** |

### Honest reading

- **click: 0.00 → ~0.7.** Clear win. The model now activates the clicked target most
  of the time, with near-zero false change.
- **delayed: k=4 0.00 → 0.67.** The door opens 2 steps AFTER the press, so at short
  horizons (k=1/k=2) there is often no change to predict yet; at k=4 the longer
  imagined rollout captures the delayed opening. This is real learning of a delayed
  consequence, not a failure of the short horizons.
- **false-change dropped everywhere** (collision 0.0075→0.0004, delayed 0.0126→0.0).
  The gate learns to leave unchanged cells alone — the decoder no longer damages the
  background.
- **movement/collision: still 0.50, unchanged.** A focused probe (vanish vs appear)
  reconfirmed with the change-mask decoder: `vanish` (agent old cell → bg) = 114/115
  ≈ 0.99; `appear` (bg → agent new cell) = **0/115 = 0.0**. The model still cannot
  predict WHERE the agent moves to.

## Deeper diagnosis (why movement's "appear" half is still 0)

The tell is the contrast between click and movement:
- **click works** because the click action ENCODES the coordinates (x, y). The
  decoder is told where to change; it only has to learn "the clicked target turns
  on".
- **movement fails on the appear half** because the target cell is
  `agent_position + direction`, and the agent's position is random per trajectory.
  To predict it, the decoder must recover the CURRENT agent position spatially and
  shift it by the action. But the recurrent state h/z are GLOBAL pooled vectors, and
  the only spatial signal (`spatial_skip`) is at 1/8 resolution — a 12×12 grid
  collapses to a 2×2 map. Single-cell position is essentially destroyed at that
  bottleneck.

So the change-mask head fixed the "what/preserve" problem (color updates at a
known location, and not damaging the background). The remaining movement gap is a
SPATIAL-RESOLUTION problem: the encoder/decoder bottleneck loses the fine position
needed to move a single cell when the action does not carry coordinates.

## Consequence for the plan

- **Keep the change-mask decoder** — it is a genuine, measured improvement (click,
  delayed) and reduces false change everywhere, with full-frame change still
  possible.
- **Next lever (carried into the retrain / an architecture follow-up):** preserve
  spatial resolution through the bottleneck so position-dependent, coordinate-free
  mechanics (movement/collision) can be learned. Options to try and MEASURE on these
  same controlled worlds before committing:
    - a higher-resolution spatial skip (fewer stride-2 stages, or a skip tapped
      earlier in the encoder),
    - a spatially-structured latent instead of a single pooled vector,
    - letting the dynamics operate on a small spatial feature map rather than a
      global vector.
  This must be validated on the controlled worlds (movement appear-half changed_acc
  well above 0, false_change low) before any real-game retrain (task #6).

## Follow-up: high-resolution encoder skip (2026-09-11)

To test the "spatial resolution" hypothesis, the encoder now also returns its
stage-1 (H/2) feature map, and the decoder fuses it into the change-mask heads
(`agent/simulator.py FrameEncoder` returns `(feature, spatial, hi_res)`; decoder
gains an optional `hi_res_skip`; threaded through decode/predict/training/rollout/
validation; back-compatible — a zero map is used when absent). Confirmatory probe
first showed a coordinate-free "toggle" world (marker flips in place, location must
come from the frame) reached only changed_acc≈0.22 with the pooled state — position
IS partially recoverable but heavily degraded.

Result of the high-res skip (`artifacts/step2b_hires_report.json`):

| mechanic | changed_acc (change-mask only) | changed_acc (+ hi-res skip) |
|---|---|---|
| click    | 0.69 / 0.72 / 0.74 | **0.94 / 0.94 / 0.95** |
| delayed  | 0.00 / 0.00 / 0.67 | 0.00 / 0.00 / **0.79** |
| movement | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 |
| collision| 0.50 / 0.50 / 0.50 | 0.50 / 0.49 / 0.50 |

Honest reading: the hi-res skip clearly helps where localization was already partly
working (click 0.7→0.94, delayed k4 0.67→0.79) but does NOTHING for movement — the
appear-half is STILL 0/115 (vanish 0.96). So decoder spatial resolution was not the
root cause of the movement failure.

## Root cause of the movement failure (precise)

The dynamics `h_new = GRU(h, z, action_enc)` — and the prior/posterior — operate on
a GLOBALLY-POOLED vector with no spatial structure. The decoder can reconstruct what
the hi-res skip shows it (the agent at its OLD position) and correctly clears the old
cell, but NOTHING in the model computes the NEW position as a spatial function of the
action. The hi-res skip actually reinforces "the agent is where it was". To learn
movement, the state the dynamics transform must itself be SPATIAL — a feature map the
action can shift/warp — not a pooled vector.

This is a genuine architectural limit, not a decoder or an epochs problem. Fixing it
means a spatial latent + spatial dynamics (+ spatial prior/posterior, packing and
checkpoint changes) — a substantial change to `agent/simulator.py`. It is the correct
next lever for position-dependent mechanics and must be validated on these controlled
worlds (movement appear-half changed_acc well above 0, false_change low) before any
real-game retrain.

KEPT from this step (measured, honest wins): the change-mask decoder AND the hi-res
skip — together they take click to ~0.94 and delayed k4 to ~0.79, with near-zero
false change, and full-frame change is still possible.

UPDATE: the movement root cause was subsequently FIXED by a full-resolution spatial
recurrent core — see ARC3_STEP7_SPATIAL_DYNAMICS.md (movement/collision appear-half
0/115 → 115/115, k=1 changed_acc 1.0).

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/step1_controlled_world.py --device cpu `
    --mechanics movement collision click delayed `
    --n-train 64 --n-eval 16 --steps 8 --epochs 60 --grid 12 `
    --out artifacts/step2_report.json
# high-res skip variant writes artifacts/step2b_hires_report.json (same command)
```

# ARC-3 corrective implementation and evidence — 2026-09-15

This supersedes claims that slice 9 proved the full framework sound. It does not
replace or erase earlier artifacts. No upload, submission, public release or paid
compute was performed. ARC-2 was not modified. The explorer baseline remains
byte-identical: SHA256 68b86938d3e7dfa2f940a84eaac7b0a87242e69d2e4aa35b92b1b5a11246033e.

## Verified repairs

- Unknown prediction errors use explicit null at the reasoner interface. Probes
  retain their actual pre-action predictions. Current observation is included,
  and recent episodes are ordered chronologically.
- The corrected posterior is consumed by the recurrent action transition before
  predicting the next latent. Real feedback advances live h; duplicate feedback
  is ignored. Imagined branches do not mutate the live state.
- Arbitrary whole-image movement bias was removed. Learned mechanic conditioning
  is explicitly unsupported in this checkpoint architecture. Goal-conditioned
  planning remains, but is not represented as learned mechanic conditioning.
- Multi-step loss rolls through imagined states; runtime and validation use the
  same transition order and re-encode imagined frames. Existing weights were not
  overwritten. New checkpoints record posterior-action-v2 transition semantics.
- Real level progress and final WIN feedback reach memory. A tiny game-local
  decoder color-bias adapter is attached: 16 parameters, at least 16 real samples,
  oldest/recent temporal train/validation separation, at most one attempt per
  eight corrections, validation rollback, exception rollback, RNG/mode restoration.
  It is output calibration, not full dynamics learning. Timing is cooperative;
  a hung GPU kernel still needs outer process control.
- Contextual single-action procedures preserve click coordinates, require diverse
  real successes for promotion, and can be reused only in matching observable
  contexts with current low-risk predictions. Contradiction revokes promotion.
  Multi-action procedural generalization remains future work.
- Evaluator sends each action once, counts attempted dispatch, propagates errors,
  rejects unavailable controls, includes terminal feedback and startup time, and
  penalizes grid-shape mismatch. Framework adapter exposes real predictions and
  increments action counters. Missing model components become recorded failures,
  not a baseline mislabeled as a full-model condition. Seeds are applied.
- No-planning performs no imagination; baselines are distinct and respect legal
  controls. Full ablations now require a disjoint split and record code, split,
  checkpoint, reasoner-file and environment-source hashes. These changes still
  need a fresh paired study; historical ablation results are invalid for this code.
- Reasoner generation and repair check deadlines, including token-boundary stopping.
  Scored notebook requires offline model paths and both real components, rejects
  legacy transition weights, and bounds the agent subprocess. Required wheel/model
  attachments are NOT yet supplied; this remains intentionally unreleasable.
- GPU predicted-grid extraction uses one bulk transfer instead of per-cell .item().
  Planning/probing share root predictions rather than repeating the same inference.

## Reproduced checks

`python -m pytest -q`: **425 passed** (16 new corrective regressions).
`python -m pip check`: no broken requirements. Notebook regenerated from source.
Release gate: **BLOCKED**, not bypassed or populated with inferred evidence.

Local GPU training smoke: 12 trajectories, 228 real transitions, three public
development games; train ar25/bp35, holdout cd82; three epochs, 620,387 parameters.
Training took 5.6 seconds. Artifact: artifacts/corrective-smoke-20260915/simulator.pt.
SHA256: 82a020d70bb6aba757bf920e577d7bc03f2ac0ca2e309c89f2a6ccdab1d4202d.
This is a wiring checkpoint, NOT competitive-quality weights: changed-cell errors
were approximately 0.993/0.997/0.996 at horizons 1/2/4.

### Actual environment diagnostic (no deliberator)

| Diagnostic | Device | Actions / corrections | Elapsed | Crash / timeout |
|---|---|---|---|---|
| Initial | RTX 3060 | 5 / 5 | 60.20 s | no / yes |
| Bulk grid transfer | RTX 3060 | 8 / 8 | 60.08 s | no / yes |
| CPU comparison | CPU, 2 torch threads | 24 / 24 | 41.95 s | no / no |
| Final, shared root predictions | CPU, 2 torch threads | 24 / 24 | 23.09 s | no / no |

Final result: 648 planner-reported imagined steps, 24 prediction/outcome pairs,
nonzero live recurrent h, 24 episodes using 301,344 memory bytes, two real adapter
attempts (one accepted, one rejected). **Zero completed levels**, approximately
96% overall prediction error, all actions remained uncertainty probes. No procedural
reuse occurred in this game because there was no success; reuse has synthetic
regression evidence only. Qwen was deliberately absent from this diagnostic.

Final evidence: artifacts/corrective-runtime-20260915-final.json. Source hash:
6d9433879fbeea975b3946060d1ac7c097270594a5d403558b02c43edd49e0fd.
Earlier failed/timed-out diagnostic reports are retained alongside it.

Reproduce from C:\Kaggle\arc-3:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/verify_corrective_runtime.py --checkpoint artifacts/corrective-smoke-20260915/simulator.pt --device cpu --actions 24 --out artifacts/new-runtime-diagnostic.json
.venv\Scripts\python.exe scripts/release_gate.py
```

## Open, dependency-ordered

1. Grounded hypothesis-to-dynamics conditioning: choose/train an actual supported
   interface, not output manipulation. Preserve the user's recursive-world design.
2. Validate and improve state-transition/frame fidelity, especially unchanged-cell
   damage; calibrate uncertainty. Predictive entropy alone is not reliability.
3. Retrain a separately versioned candidate on approved data with held-out evaluation;
   neither loading old weights nor the tiny smoke run establishes competence.
4. Full dynamics adaptation, causal multi-action procedural reuse, goal revision
   and delayed credit beyond the conservative mechanisms verified here.
5. Real-Qwen/full-system rerun and fair ablations. Determine a separately configurable
   simulator/reasoner device split; CPU was faster for this small serial simulator
   locally, not proof that CPU is faster for all models or on Kaggle.
6. Full track rules/data/license review, actual offline wheels and model attachments,
   official competition-mode rehearsal, per-game reset semantics, fault injection,
   and explicit user approval. Never submit the current partial system as ready.

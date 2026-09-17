# ARC-3 corrective implementation plan
Reviewed 2026-09-14 against source and behavioral probes; replaces optimistic completion claims where the review found disconnected paths.

## Invariants
Official ARC-3 rules govern. Preserve the user's internal-world reasoning, original implementation and lean game-local memory. No competition upload/submission or public release. Retain existing weights and baseline notebook; do not overwrite evidence with incompatible results. New weight architecture requires a new checkpoint version. Passing fixtures is not performance evidence.

## Dependency order and acceptance
| Phase | Repairs | Acceptance |
|---|---|---|
| P0 | Freeze review baseline, retain old reports with invalidation notice; keep rules/disclosure gate blocked | Reproduce baseline suite; no weights/notebook replacement |
| P1 | One initialization, cached observations, exactly one environment action per decision; complete action coordinates; real availability; terminal feedback; accurate counters | Stateful spy tests, click-only game, provider failure, deadline and final-success tests |
| P2 | Correct observed-state/posterior lifecycle, idempotent feedback including failure/reset; retain actual probe predictions or explicitly unknown error; no observation-as-prediction substitution | Consecutive observations update state; saved prediction precedes outcome; terminal events recorded once |
| P3 | Make hypotheses reach the actual dynamics backend and preserve full recurrent branch state | Seed-controlled tests of supported conditioning effects; frozen branch isolation; reject unsupported conditioning rather than fabricate influence |
| P4 | Replace availability-derived uncertainty with calibrated dynamics uncertainty; true multi-step open-loop loss; spatial fidelity and action-conditioned holdouts | Separate changed/unchanged errors; 1/2/4/8-step rollouts; calibration and false-change reports; checkpoint versioning/retraining |
| P5 | Wire bounded online adapters, rollback and contextual procedure creation/reuse; validated goal revision | Actual accepted/rejected updates and procedural reuse traces; temporal split; contradiction revocation; memory-off ablation |
| P6 | Repair ablation interventions, seed control and provenance; distinct baselines; no fake full condition | One mechanism changed per condition, real hashes, no fallback labeled full, cost includes initialization, correct action accounting |
| P7 | Enforce inference/run budgets; offline complete model/dependency bundle | Stopping criteria plus outer cancellation strategy, one shared run deadline, cold-start on actual Kaggle runtime |
| P8 | Repeat whole-game holdout, fault injection and compliance review before release | Test-fix-test report, no benchmark exclusions, full official rules/permissions, explicit user approval |

## Review-driven revisions
- Evaluation correctness comes before more training: current ablation data is not usable for mechanism comparisons.
- Retain legacy fixture callback support only when explicit; real environment transport must carry availability and coordinates. Do not silently invent legal controls.
- Initialization RESET is counted separately from agent-issued actions. Retry/read operations must not advance the game; failed attempted actions remain visible.
- GAME_OVER feedback must be recorded before legal level reset; WIN must not disappear merely because the framework stops requesting actions.
- Learning success means observed level advance or final win, not arbitrary visual change.
- A missing prediction is unknown, not zero error. Capture probe prediction before execution if available.
- Do not make a neural conditioning test pass by shifting pixels with hand-authored mechanics; implementation and training must ground effects.
- The current weights cannot be certified for a changed latent representation or training loss. Keep old artifacts, write candidate artifacts separately, and validate before promotion.
- Fix goal/deliberator context: include current observation, not only older episodes; preserve observations needed for object coordinates.
- Neither an additional model nor more epochs is assumed to cure incorrect transport or missing data flow.
- Free-tier/local resources only; confirm permission for data use or paid compute before affected work.

## Status
Implementation checkpoint, 2026-09-15: P1/P2 transport, feedback, unknown-error handling,
posterior/action lifecycle, and probe-prediction retention repaired and regression-tested.
P3 fake whole-frame movement conditioning removed; unsupported learned conditioning is
explicitly reported, not claimed complete. P4 runtime/training/validation transitions
aligned and a separately named smoke checkpoint trained; calibration and competitive
retraining remain open. P5 live 16-parameter color-bias adaptation with rollback and
conservative contextual procedure reuse wired; full dynamics adaptation and multi-action
procedural generalization remain open. P6 key evaluation controls and provenance repaired;
no new full-model ablation or score evidence. P7 notebook rebuilt, strict model prerequisites
and token-boundary deadlines added; offline dependencies/inputs and actual Kaggle rehearsal
remain open. P8 remains blocked. See ARC3_CORRECTIVE_RESULTS_2026-09-15.md.

Full track-specific rules, data permissions and disclosure review remain unresolved;
no compliance, novelty, competitive improvement or release-readiness claim is made.

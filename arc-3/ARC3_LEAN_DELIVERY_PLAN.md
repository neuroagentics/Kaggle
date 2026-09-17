# ARC-3 lean delivery and independent-agent handoff

Draft 2026-09-10. Implements ARC3_LEAN_BLUEPRINT.md. No publication, model upload,
paid training, or competition submission is authorized merely by this plan.

## Starting truth

The project has baseline competition plumbing and an exploratory agent. The new
internal_world.py supplies simulation/planning contracts, not trained dynamics.
Fixture tests are not unfamiliar-game evidence. Neither required learned role is
integrated by this document. Historical scores belong to their original bundles.
Preserve all existing user changes and official-starter attribution. No ARC-2 edits.

## Dependency-ordered work packages

| Gate | Work and deliverable | Depends on | Exit evidence |
|---|---|---|---|
| G0 | Full ARC-3 rules matrix; public/private boundary; immutable baseline inventory | None | Each rule linked to source, implementation control and test; unresolved constraints block release |
| G1 | Schemas, memory caps, environment adapter, model-interface contract | G0 provisional | Replay, isolation, contradiction, eviction, coordinates, fork and timeout tests pass |
| G2 | Permitted trajectory collector, curriculum, game-family splits and manifest | G0 data clearance; G1 | Dataset audit; no engine introspection, private sources or cross-split trajectories |
| G3 | Train encoder/dynamics/decoder and outcome heads | G2 | Real checkpoint, reproducible config, loss curves, held-out changed-cell and multi-step reports |
| G4 | Select and package one quantized deliberator; validate structured hypotheses | G0 license clearance; G1 | Offline load, device/VRAM report, bounded inference, malformed-output tests |
| G5 | Integrate posterior, simulation, recursive search, action, feedback and adaptation | G3 + G4 | Trace proves real action choice used imagined states and same-game evidence |
| G6 | Contextual procedures and chaining | G5 | Preconditions checked; later behavior improves from earlier experience; contradictions revoke reuse |
| G7 | Whole-game evaluation and fault injection | G5 + G6 | Paired ablations, complete denominators, no hidden exclusions, measured resource envelope |
| G8 | Immutable offline Kaggle package and competition-mode rehearsal | G0 closed + G7 | Same hashes, both real models active, bounded completion, official artifact validated |
| G9 | User review and explicit upload/submission approval | G8 | Release report with strengths, failures and risks; no automatic submission |

G4 can proceed alongside data/model development once G1 exists. G5 cannot be
declared done with a fixture dynamics backend. A model download is not G3.
Do not spend weeks perfecting memory before proving the simulator can predict.

## First implementation slice

1. Inventory existing files and tests, record hashes and protect the baseline.
2. Obtain the full rules text or ask the user for an export if inaccessible.
   Continue safe local contracts meanwhile; do not assume unresolved permissions.
3. Implement the bounded memory schema and versioned model protocol. Extend
   internal_world.py rather than creating an unrelated competing controller.
4. Collect a small permitted training corpus plus untouched whole-game holdout.
5. Train the first real simulator and test 1/2/4-step predictions immediately.
6. Connect one real reasoner and demonstrate observe -> imagine -> act -> correct
   on a permitted environment with exported evidence, then expand the corpus.

If this slice fails, diagnose data coverage, representation, uncertainty or
optimization before adding more frameworks. Never respond by substituting an
unrelated historical solver or an LLM-only button chooser.

## Expected module boundaries (planned, not existing)

- agent/memory.py: bounded real evidence, revisions and contextual procedures.
- agent/model.py: encoder/posterior/dynamics/decoder and versioned checkpoint load.
- agent/deliberator.py: one local model backend, schema validation, time budget.
- agent/internal_world.py: existing fork/step/planner contracts extended carefully.
- agent/controller.py: one MRLA loop and official adapter integration.
- training/: collectors, source manifests, split builder, reproducible training.
- scripts/evaluate.py: full-game metrics, ablations, raw report and confidence intervals.
- scripts/preflight.py: hash/license/device/offline/model-activity checks.
- tests/: contracts, model tests, integration and failure-injection tests.

Do not add a service framework, distributed workers, vector store, PHCG package,
web backend, model registry server or cloud memory dependency.

## Measurement contract

Freeze the evaluation manifest before tuning. Hold out whole games and mechanic
families; do not tune on private scores. Use at least three fixed seeds where
stochastic behavior matters, and equal total wall-clock/action allowances.
Report per-game outcomes as well as aggregate results; count timeouts and failures.

Required comparisons:
- historical explorer as a baseline only, not packaged as the intended solution;
- complete system vs no deliberator;
- complete system vs no imagined planning;
- complete system vs no within-game memory/adaptation;
- complete system vs no procedural chaining;
- prediction vs persistence and simple-motion baselines.

For each report: game IDs/split hash, model/data/code hashes, mode, seeds, completed
levels, official metric where available, actions, latency distribution, elapsed
time, peak RAM/VRAM, model activity, prediction/calibration errors, and crashes.
Label local NORMAL-mode scores explicitly; do not equate them with Kaggle scores.

Promotion requires a positive aggregate held-out improvement with paired
uncertainty reported, no unexplained runtime failures, and compliance closure.
If sample size makes the gain inconclusive, mark it inconclusive and gather more
evidence. Set practical effect-size targets before evaluation, not after seeing
results. A good pixel score cannot substitute for more solved levels/actions saved.

## Reliability tests required before packaging

Missing/corrupt weights; missing wheel; wrong CUDA/kernel; CPU fallback; OOM;
non-finite dynamics; hallucinated action; invalid coordinates; incorrect grid
shape; stale model version; branch contamination; memory cap/eviction; new-game
isolation; bad hypothesis JSON; unsupported evidence; deliberator timeout; adapter
regression/rollback; terminal/reset handling; deadline reserve; commit versus
scored mode; artifact schema and offline cold start. No promise of zero risk.

Each controllable warning gets a source and disposition. Fix project warnings;
isolate third-party dependency conflicts. Do not globally suppress warnings or
modify unrelated Kaggle libraries to produce cosmetically clean logs.

## Equal comparison between coding agents

Give each implementation this same blueprint, rules matrix, fixed data split,
hardware/time budget and acceptance tests. Keep evaluation answers inaccessible
to training/tuning. Preserve each artifact and code hash before evaluation.
Judge by held-out completion/action efficiency, reliability, same-run learning,
resource use and reproducibility, not eloquent logs or number of passing mocks.
An independent implementation may change engineering details with a written
decision record, but must not change the user's reasoning objective unnoticed.

## Change control and stopping criteria

At every gate ask: Does this help reconstruct the game, simulate futures,
choose better actions or learn from outcomes? Is it legal, measurable and lean?
Reject work that serves none of these purposes. Record decision, evidence,
consequence, next falsification test and any departure from the blueprint.

A single bad run does not disprove the user's approach. Repeated controlled
failures across held-out mechanics, with functioning training and runtime plus
documented targeted repairs, justify proposing a change. They do not authorize
silently abandoning the concept. Escalate unresolved rule conflicts or resource
requirements to the user, with evidence and the smallest viable alternatives.

The current task is drafting. Subsequent progress reports must state which gates
passed, link actual evidence, enumerate blockers and identify the next executable
step. No percent-complete claims based on file count. No AI, no release candidate.

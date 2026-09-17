# Phase B Structural World-Model Gate

Date: 2026-09-09

## Outcome

The first capability-bearing Phase B slice is enabled for the Kaggle runtime.
It adds task-state-derived component recoloring rules to the owned recursive
world model while leaving unvalidated persistent memory channels disabled.

The change passed the development and independent holdout gates:

| Partition | Previous world pass@2 | Phase B world pass@2 | Previous combined pass@2 | Phase B combined pass@2 |
|---|---:|---:|---:|---:|
| Development validation (160 tasks) | 11 | 12 | 12 | 12 |
| Untouched holdout (200 tasks) | 7 | 10 | 8 | 11 |
| Labeled evaluation (120 tasks) | 0 | 0 | 0 | 0 |

All figures are whole-task exact matches. Demonstration fit and cell accuracy
are not counted as solutions.

## Implemented slice

- `TaskState` perception is the source of object attributes.
- `component_recolor` is a typed executable primitive.
- Candidate rules bind output colors to component area, shape, dimensions,
  holes, border contact, and conservative combinations of those attributes.
- Every hypothesis retains its source, parent, repair depth, executable AST,
  traces, and residuals.
- Rules must replay every training pair exactly before they may predict a test
  output.
- The new family cannot displace two established exact-replay hypotheses from
  both pass@2 slots. It fills capacity only after promoted families.
- The Kaggle runtime enables the recursive world-model algorithm without
  enabling persistent world-memory retrieval.

## Evidence

- Development: `artifacts/phase_b_structural_validation_v2.json`
- Independent holdout: `artifacts/phase_b_structural_holdout_v1.json`
- Labeled evaluation: `artifacts/phase_b_structural_public_evaluation_v1.json`
- Production submission score: `artifacts/phase_b_production_public_score.json`
- Production run manifest: `artifacts/phase_b_production_public_run_manifest.json`

The independent holdout gain is three exact tasks for both the world-model and
combined policies. The labeled evaluation remains zero, so this release is a
measured capability improvement and a valid unseen-task experiment, not
evidence of leaderboard points.

## Runtime and ownership boundary

- Offline execution; no runtime package installation or network access.
- External verified-symbolic/NVARC code remains excluded.
- Legacy seed memory, experience memory, and world memory remain disabled
  because they have not added exact evaluation solves.
- The bundle remains marked non-public until the owner chooses a license.

## Verification

- Full suite: 211 passed.
- The bundle sentinel executes both deterministic exact replay and the promoted
  structural world-model path.
- The production-format 120-task submission passed schema validation and
  scored 0 exact tasks / 0 exact outputs at pass@2.

## Kaggle artifact

Use only this archive for the next Kaggle dataset version:

`artifacts/bundle_phase_b_final/hyper_arc_solver_bundle.1282f946bad31763691f0010dc44c4981fb3a2149ea21161784089b321cdccae.zip`

SHA-256:
`1282f946bad31763691f0010dc44c4981fb3a2149ea21161784089b321cdccae`

The notebook source is `notebook6a96ea823f.ipynb`. It verifies the archive and
internal manifest, runs the bundle sentinel, enables `--enable-world-model`,
keeps all memory channels disabled, and validates the final submission schema
and hash.

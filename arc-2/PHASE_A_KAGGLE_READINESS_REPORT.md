# Hyper-ARC Phase A Kaggle Readiness Report

**Date:** 2026-09-09  
**Decision:** deployment integrity gate passed; capability gate failed  
**Submission decision:** do not submit this baseline

## Outcome

Phase A converted the existing solver from a permissive notebook experiment
into a hermetic, fail-closed Kaggle package. It did not improve ARC intelligence,
and no such claim is made.

The exact default-package evaluation remains:

- 120 public evaluation tasks and 172 test outputs;
- pass@1: `0/120` tasks and `0/172` outputs;
- pass@2: `0/120` tasks and `0/172` outputs;
- 0 task failures and 0 deadline fallbacks;
- 134.72 seconds on the local CPU at 25 MCTS iterations per task;
- 216 identity attempts and 0 distinct attempt pairs.

The result proves that the package executes consistently and that the legacy
MCTS path is not a contender. It is the frozen floor for Phase B.

## Completed deployment work

- Replaced global/runtime `geoopt` installation with portable Poincare-ball
  projection and distance math.
- Removed `geoopt` from project dependencies and eliminated the associated
  runtime warning surface.
- Replaced loose-source notebook staging with exactly one content-addressed
  bundle.
- Added archive-member size and SHA-256 verification plus path-traversal checks.
- Added a deterministic internal manifest, declared channel registry, source
  digest, dependency inventory, and explicit pending-license state.
- Added an extracted-bundle sentinel that loads all frozen assets and executes
  an exact-replay fixture.
- Added a 10-hour global deadline, atomic output, run manifest, environment
  capture, failure counts, timeout counts, identity counts, and attempt-diversity
  counts.
- Bound checkpoint resume to a challenge/configuration signature and prevented
  disabled memory state from being restored through an older checkpoint.
- Stopped silently translating an MCTS test-execution exception into an input
  grid inside the solver.
- Made task-level fallback a named `record-input` policy; required asset errors
  remain fatal before inference.
- Disabled the existing experience memory, legacy seed memory, and recursive
  world model by default because none passed an exact-score promotion gate.
- Continued to exclude the external verified-symbolic adapter from Hyper-ARC.

## Verification evidence

| Gate | Result |
|---|---|
| Compile | PASS |
| Full tests | `206 passed` |
| Deprecated/runtime warnings | none reported by the full suite |
| Deterministic bundle build | PASS; repeated builds are byte-identical |
| Bundle digest | `d189a50be51c88bd3d3ae19d01286ccfaccbe75c1eead38ac6a0ea4142bc0e39` |
| Bundle size | 334,084 bytes |
| Extracted sentinel | PASS: 5 seed, 1,920 experience, 47 world-memory records load |
| Packaged smoke task | PASS; valid output and run manifest, no failure or timeout |
| Default public exact evaluation | FAIL capability gate: `0/120` pass@2 |

Artifacts:

- `artifacts/bundle_phase_a_final/hyper_arc_solver_bundle.d189a50be51c88bd3d3ae19d01286ccfaccbe75c1eead38ac6a0ea4142bc0e39.zip`
- `artifacts/bundle_phase_a_final/run_manifest.json`
- `artifacts/bundle_phase_a/public_eval_run_manifest.json`
- `artifacts/bundle_phase_a/public_eval_exact_report.json`
- `artifacts/phase_a_seed_exact_report.json`
- `artifacts/phase_a_no_seed_exact_report.json`

## Memory promotion decision

A deterministic miner found five unique exact-replay seed programs among 200
permitted training tasks. The asset is now real, reproducible, packaged, and
loadable. It is not useful yet:

| Configuration | Public pass@2 | Runtime | Promotion |
|---|---:|---:|---|
| five-record legacy seed enabled | 0/120 | 159.56 s | rejected |
| no legacy seed | 0/120 | 155.56 s | baseline |

The seed bank produced zero unique exact solves and added about four seconds.
It stays in the research bundle as evidence and a sentinel fixture, but the
runtime does not activate it. Phase C must replace primitive-similarity priors
with task-conditioned structural retrieval and executable procedural traces.

## Remaining release blockers

- The public permissive license is deliberately still `PENDING`; selecting it
  changes the legal distribution terms and remains an owner decision.
- `THIRD_PARTY_NOTICES`, SBOM output, model card, data card, and contamination
  registry are not complete.
- A clean Kaggle `Save & Run All` on the intended competition image remains
  required; local packaging evidence is not a Kaggle runtime result.
- Most importantly, the exact capability gate is zero. This bundle must not be
  uploaded as a score-seeking submission.

## Authorized next boundary

Proceed to Phase B with the current package as a frozen deployment harness:

1. introduce one shared `TaskState` and canonical deterministic views;
2. route typed executor, residuals, and candidate lineage through that state;
3. replace broad MCTS fallback with residual-guided typed hypothesis expansion;
4. require a non-zero exact solve on the sealed development/validation protocol
   before re-enabling any learned or memory channel;
5. rebuild the same content-addressed bundle only after that gate passes.

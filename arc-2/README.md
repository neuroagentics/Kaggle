# Hyper-ARC: active ARC-AGI-2 release contract

Alignment date: 2026-09-20. Release identity: `hyper-arc-rc-20260920`.
Status: **UNQUALIFIED_CANDIDATE — no competition submission authorized**.

This document supersedes deployment instructions in older ARC-2 plans, Phase
reports, SYSTEM_PROOF.md, and the parent August operating/status documents.
Those files and everything under `artifacts/` are historical research/evidence,
not active deployable versions. ARC-3 is a separate project and is unchanged.

## One source of truth

- Source: `C:\Kaggle\arc-2`, mirrored in private `neuroagentics/Kaggle`, `arc-2/`.
- Configuration and runtime qualification: `release_policy.py`.
- Notebook source: `kaggle_launcher.py`; do not hand-edit generated notebooks.
- Build: `python build_release.py` generates `release/current/`.
- Active package: only the bundle named by `release/current/release.json`, its
  generated notebook and kernel metadata. The notebook checks release ID and
  exact runtime source-tree digest. Prior generated bundles move to
  `release/history/`, outside the upload directory. Do not upload that history.
- Model profile: attached `google/gemma-4/Transformers/gemma-4-e4b-it/1`, through
  the offline Transformers transport. This is the existing attachment selected
  for qualification, NOT a claim of model quality. Qwen/Ollama bakeoffs remain
  research only; no substitution is implied or authorized by their results.
- Linear: ARC-AGI-2 2026 Solver; NEU-214 owns release qualification, NEU-211
  model evidence, NEU-219 scoped memory, NEU-213 attempt diversity.

The bundle's source revision identifies the build base; the content digest is
authoritative for runtime equality. A later documentation/release-index commit
does not create a different runtime. Exact package hashes are in release.json.

## Intended reasoning architecture

Demonstrations -> object/relational analysis -> executable hypotheses -> exact
demonstration execution -> typed residual diagnosis -> bounded repair or new
hypothesis -> behaviorally distinct predictions -> output validation.

The offline AI proposes executable restricted Python and receives execution
feedback. It augments the owned symbolic/world-model search; it is not replaced
by scripts. The symbolic search is an internal executable model, not a claim
that a learned latent dynamics model has been trained. ARC-2 infers static
transformations, not ARC-3 interactive game dynamics. Full PHCG/collective IP,
external solver implementations, and unpromoted neural specialists are excluded.

## Memory and data boundaries

- Preserve the 25 builder-derived procedural chains; they rank demonstration-
  verified candidates and provide cues. Do not fabricate new solved episodes.
- `agentic_memory_builder_v2.json` is currently empty. The older validation-
  derived e9afcf9a record is quarantined as historical evidence, not packaged.
- Persistent neural success records require builder scope and independently
  replayed source demonstrations AND all supplied builder test labels. An
  ensemble success does not certify every member of the ensemble.
- During a task, hypotheses, failed approaches, residuals and feedback inform
  subsequent rounds immediately.
- `--session-transfer --no-resume` enables bounded in-memory transfer between
  development tasks. Source procedures are only demo-supported, never labeled
  as hidden-test successes, and must replay every new task's demonstrations.
  Failure cues merge after each task. Static input banks are never overwritten.
- Competition mode forbids cross-task session transfer until binding rules
  have been reviewed for that scope. Do not silently broaden a privacy/data rule.
- Validation memory writes are rejected. Previously inspected validation,
  holdout and public-evaluation sets are regression sets, not pristine evidence.
- Color/rotation consistency is advisory research, not a universal correctness
  gate. Replaying an already fitted program is not leave-one-out induction.

## Runtime and release gates

Task budget: 120 seconds shared across channels; full solver budget: 34,200
seconds including initialization. Generation has cooperative time limits and
the notebook provides an outer process watchdog. These limits are not a promise
that a wedged GPU kernel can be interrupted between tokens. Linux code workers
also have a 256 MiB address-space limit and subprocess timeouts.

No silent prompt truncation, model download, CPU fallback, or model-free
competition run. A live-model synthetic preflight must pass separately from
the stub-backed bundle sentinel. Every invoked neural task must execute at
least one candidate. Missing usable AI, task failures/timeouts or budget
overruns prevent runtime qualification. Runtime qualification does not prove
hidden correctness or competition readiness.

Before release, all of the following remain required:

1. Binding Kaggle rules retrieved and checked, including memory/data scope.
2. Owner-approved public license and reviewed model/source license inventory.
3. Actual attached weights/tokenizer load and pass the real offline transport.
4. Independently evaluated exact per-test-output pass@2 improvement with a
   matched baseline, multiple runs/seeds, and no validation-memory leakage.
5. Two clean full offline GPU rehearsals, complete output coverage, acceptable
   runtime, and no non-executing neural tasks.
6. User authorization for competition submission/publication.

The official ARC Prize page was rechecked on September 20; the complete Kaggle
rules body was not accessible through web retrieval. Compliance is not certified.

Sources: https://arcprize.org/competitions/2026 and
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules .

## Verified baseline and current blockers

September 20 baseline: clean GitHub/local b965747; 264 tests passed. Latest
retained Qwen small-set experiments ranged 0/6 to 1/6, same solved task; the
12-task experiment solved zero. No broad neural transfer improvement is proven.
The old generalization filter rejected three correct programs without adding a
task solve. These results do not establish an inherent small-model ceiling.

Current Kaggle kernel status, checked September 20: ERROR. Downloaded status
reports no GPU allocation; it did not reach neural solving. Accelerator selection
must be explicit and confirmed in logs; enable_gpu metadata alone is insufficient.
The older v10 run's 196 invocations/zero executable candidates is not qualified
under the corrected policy. Its historical SOLVER_PROMOTED label is superseded.

Local unit tests do not load deployment weights. No local copy of the selected
weights was found in the inspected model cache, and Ollama was not running.
Private Kaggle qualification requires free-GPU quota and explicit run approval.

Correction evidence: 284 local tests passed after alignment, including rejection
of the historical false promotion, release/source identity, deadline handling,
same-run development-memory transfer, validation-write rejection, per-program
success verification, and distinct neural behaviors. Dependency check passed.
The corrected augmentation diagnostic retained all seven demo-fit programs and
lost zero correct programs; task solves stayed 3/160 validation and 0/120 public
evaluation. This is a correctness repair, not a newly demonstrated score gain.

## Reproduce and operate

```powershell
cd C:\Kaggle\arc-2
python build_release.py
python -m pytest -q -p no:cacheprovider
python -m pip check
```

Development dependency ranges are not a deployment lock. The generated notebook
retains hash-verified offline wheel inputs in an isolated target directory; the
Kaggle GPU/Python/base-image combination still requires rehearsal.

Never push a notebook against an older attached dataset: update the private
code dataset to the exact newly built bundle first, then verify the notebook's
expected source-tree digest. A private qualification run is distinct from
Submit to Competition. No script here performs either upload or submission.

> Historical document, superseded for ARC-2 deployment on 2026-09-20. See [README.md](README.md) and release/current/release.json. Retained results do not certify the current candidate.

# Hyper-ARC: ARC-AGI-2 Placement Recovery Plan

> The refined runtime, compatibility, evaluation, and open-source contract is
> defined in `HYPER_ARC_KAGGLE_REFERENCE_ARCHITECTURE.md`. This recovery plan
> supplies the schedule and gates; the reference architecture is authoritative
> for component boundaries and packaging.
>
> Phase A is complete as a deployment-integrity milestone. See
> `PHASE_A_KAGGLE_READINESS_REPORT.md`. Its exact capability result is still
> `0/120`, so it is explicitly not a score-seeking submission candidate.

**Plan date:** 2026-09-07  
**Competition deadline:** 2026-11-02, 23:59 UTC  
**Current external result:** Version 7, `0.00` public score  
**Current owned-system result:** `0/120` exact pass@2 on the clean public
evaluation run

## 1. Decision

Do not repair and resubmit Version 7 as the contender. Repair its deployment
faults immediately, but replace its shallow MCTS-centered capability path with
an owned hybrid refinement system:

1. exact deterministic and typed-program solvers;
2. a compact recursive neural candidate generator trained on permitted public
   and synthetic data;
3. per-task test-time adaptation and augmentation;
4. residual-guided program and grid refinement;
5. a frozen, task-conditioned hyperbolic experience memory; and
6. a verifier/ranker that emits two behaviorally distinct candidates.

This is the leanest direction consistent with both the project concept and
competition evidence. The 2025 leaders used test-time training, augmentation,
recursive prediction, and refinement. A larger unguided DSL search is not a
credible placement strategy.

Placement cannot be guaranteed. The operational target is to earn it through
clean exact-match evidence. A release candidate is not called competitive until
it reaches at least 10% pass@2 on untouched ARC-AGI-2 evaluation tasks, with a
stretch target of 15% or more. For context, the 2025 private leaderboard's fifth
place was 6.53%, third was 12.64%, second was 16.53%, and first was 24.03%.

## 2. What must never recur

Version 7 was structurally valid but did not run the intended system:

- the notebook staged `main.py` and `hyper_arc/` but not the symbolic directory
  expected by its adapter;
- the adapter silently converted a missing subsystem into zero candidates;
- the seed bank was absent and MCTS used uniform priors;
- every `attempt_2` was the input grid and 122 `attempt_1` values were also the
  input;
- high partial-cell training scores were reported like progress even though
  Kaggle awards only exact output-grid matches.

The next deployment must fail before inference when any required component is
missing, mismatched, unlicensed, untested, or inactive.

## 3. Target system

```text
demonstrations + test input
        |
        v
canonical pixel/object/relationship representation
        |
        +-------------------+--------------------+------------------+
        |                   |                    |                  |
deterministic rules   memory-retrieved     recursive neural    typed-program
and invariants        schemas/fragments     candidates + TTA    synthesis/repair
        |                   |                    |                  |
        +-------------------+--------------------+------------------+
                                    |
                          exact demonstration replay
                                    |
                  leave-one-demo-out + augmentation consistency
                                    |
                     residual refinement and candidate voting
                                    |
                 two distinct ranked test-output hypotheses
```

### 3.1 Canonical world representation

Maintain both pixel and object views. Extract dimensions, palette, background
hypotheses, connected components, bounding boxes, containment, adjacency,
symmetry, separators, panels, repetition, correspondences, and input/output
deltas. Represent uncertainty explicitly; do not commit prematurely to one
background, connectivity rule, or object decomposition.

### 3.2 Typed executable hypotheses

Use a typed AST with grid, object, object-set, mask, color, coordinate, and
integer values. It must support selection, iteration, conditional operations,
multiple working registers, canvas construction, composition, counting,
topology, repetition, and object correspondence. Every prediction must retain
its executable program or neural provenance and complete verification trace.

### 3.3 Recursive refinement loop

For each task:

1. propose several structurally different hypotheses;
2. execute them on every demonstration;
3. classify residuals by shape, object, relation, color, and location;
4. repair the smallest failed assumption or program fragment;
5. repeat until exact replay, repeated state, or budget exhaustion;
6. use minimum-description length only to rank exact candidates, never to make
   an inexact candidate appear solved.

### 3.4 Neural candidate generator and test-time adaptation

Replace the current 135,750-parameter specialist, which contributed zero exact
validation solves, with a deliberately small but adequately trained recursive
or masked-refinement model. Train it offline on:

- the 1,000 permitted ARC-AGI-2 training tasks;
- ARC-AGI-1 public data where license and provenance permit;
- programmatically generated tasks from the owned typed grammar;
- dihedral, color-permutation, translation, padding, and object-order
  augmentations that preserve the rule.

At test time, adapt independently to each puzzle's demonstrations. Candidate
generation should aggregate multiple augmentations and inverse-transform the
outputs. The model is a proposer and refiner; exact execution and consistency
checks remain authoritative.

### 3.5 Hyperbolic memory with a narrow job

The competition memory is a compact, frozen index—not the full personal PHCG.
Each record stores:

- a structural task signature;
- a typed program or reusable program fragment;
- transformation family and invariants;
- successful and failed repair traces;
- exact held-out outcome, provenance, data split, and version;
- task and program embeddings used only for retrieval.

Retrieve top-k analogous schemas to seed synthesis and neural refinement. Every
retrieved hypothesis must still replay all demonstrations exactly. Keep hidden
task working memory ephemeral and do not promote hidden-task guesses into the
frozen competition bank.

Memory remains in production only if a paired no-memory ablation shows unique
exact held-out solves with no unacceptable regression.

### 3.6 Two-attempt policy

The two attempts are treated as a portfolio:

- both must be valid and generated intentionally;
- candidates are deduplicated by their behavior on demonstrations and
  augmented counterfactuals;
- `attempt_1` is the best calibrated exact-demonstration hypothesis;
- `attempt_2` is the best meaningfully different hypothesis;
- identity is allowed only when it is itself a ranked, evidence-supported
  rule—not as a universal fallback;
- if only one credible hypothesis exists, derive a controlled alternative from
  the next uncertainty (object choice, symmetry direction, color binding, or
  output construction), not a blind input copy.

## 4. Deployment contract

### 4.1 One hermetic release artifact

Build exactly one `solver_bundle.zip`. Remove the notebook's loose-source
fallback. The bundle contains all and only:

- owned runtime source;
- model weights and tokenizer/config if used;
- frozen experience/world memory;
- offline wheels that are not already guaranteed by Kaggle;
- license inventory;
- `manifest.json` containing schema version, Git commit, file sizes, SHA-256
  hashes, model/memory IDs, channel list, and expected sentinel results.

The notebook must locate exactly one archive and verify every manifest entry
before extraction. Zero archives, multiple archives, missing files, hash
mismatches, unexpected files, or an unsupported schema are fatal errors.

### 4.2 Fail-closed startup

Replace warnings and silent empty returns with explicit health checks:

- `--require-memory` verifies bank ID, record count, dimensions, and digest;
- `--require-model` loads the exact checkpoint and runs a deterministic forward
  pass;
- `--require-channels` initializes every declared candidate channel;
- each channel solves or recognizes its packaged sentinel task;
- competition data identity and task/output counts are logged;
- network is confirmed disabled;
- GPU/CPU device, available memory, software versions, and deterministic seeds
  are recorded.

The run stops before task 1 if any required check fails. Optional channels must
be explicitly declared optional in the manifest and cannot disappear silently.

### 4.3 Runtime telemetry

For every task, record without exposing hidden answers:

- candidates proposed, executed, rejected, and retained by channel;
- demonstration-exact hypothesis count;
- leave-one-out and augmentation-consistency scores;
- two-attempt behavioral diversity;
- fallback reason, timeout, elapsed time, and peak memory;
- retrieved memory IDs and distances;
- selected candidate provenance.

Post-flight assertions require all 240 task IDs, all 259 ordered test outputs,
two valid grids each, zero unreported exceptions, expected channel activity,
and runtime below the release budget.

### 4.4 Fresh-environment test

Every release must be extracted into a new temporary directory with repository
paths removed from `PYTHONPATH`. Run:

1. manifest and license verification;
2. unit tests and compilation;
3. channel sentinel tests;
4. multi-output submission smoke test;
5. a representative CPU/GPU runtime sample;
6. a full simulated Kaggle notebook run;
7. downloaded-output hash and telemetry audit.

Passing tests in the development checkout is insufficient.

## 5. Evaluation discipline

### 5.1 Data partitions

Create an immutable split manifest before further modeling:

- **builder:** 640 of the 1,000 training tasks;
- **development:** 160 training tasks for fast iteration;
- **shadow holdout:** 200 training tasks, revealed only by the evaluator;
- **public evaluation:** 120 tasks, used only at named milestone gates—not for
  continuous tuning.

Track every dataset, synthetic generator, checkpoint, and external artifact in
a contamination and license registry. Public-evaluation feedback may select a
release, but it must not be written into memory or training examples.

### 5.2 Primary score

The only capability metric is exact pass@1/pass@2 on unseen test outputs and
whole tasks. Partial-cell agreement, demonstration fit, training loss, and
candidate count are diagnostics.

Report 95% bootstrap confidence intervals, unique solves by channel, and paired
differences for every ablation. Do not promote a feature that merely moves soft
scores.

### 5.3 Required gates

| Gate | Required evidence |
|---|---|
| Deployment | Hermetic bundle passes a fresh-directory and Kaggle commit run; all required channels prove active |
| Baseline | Reproduce current results and at least one known open reference result without incorporating its identity or code into Hyper-ARC |
| Representation | Object/relation features survive rule-preserving mutations |
| Program engine | Accepted programs replay every demonstration exactly and add shadow-holdout solves |
| Neural branch | Recursive/TTA branch adds at least 3 unique shadow-holdout solves over symbolic-only |
| Memory | Memory-on beats memory-off by at least 2 unique exact shadow-holdout tasks with no more than 1 regression |
| Pass@2 | Second attempt adds at least 15% relative exact coverage over pass@1 and duplicate attempts remain below 5% |
| Runtime | Two full runs complete below 10.5 hours on the selected Kaggle accelerator |
| Public milestone | At least 2/120, then 6/120, then 12/120 exact pass@2; no placement claim before 12/120 |
| Release | Two clean builds produce identical manifests and deterministic-channel outputs |

## 6. Eight-week execution plan

### September 7-11: stop deployment drift

- add the release manifest, strict archive loader, and required-component flags;
- delete the loose-source notebook fallback;
- add channel sentinels and an end-to-end clean extraction test;
- make missing memory/model/channel fatal;
- add attempt-diversity and exact-metric reports;
- reproduce Version 7 from its artifact and preserve it as the failure baseline.

**Exit:** a deliberately damaged bundle fails before inference; an intact bundle
runs from a clean directory and proves every declared channel active.

### September 12-19: establish credible baselines

- freeze split, contamination, and license manifests;
- benchmark owned deterministic, typed, world-model, memory, and current neural
  channels separately;
- reproduce one small open recursive/TTA baseline as a research control;
- profile full-run compute and choose CPU, T4/P100, or L4x4 deliberately.

**Exit:** one scorecard reports exact pass@2, unique solves, runtime, and peak
memory for every channel. Non-contributing channels are removed from runtime.

### September 20-October 3: build the neural refinement floor

- generate typed-program synthetic tasks with provenance and held-out generator
  families;
- train the compact recursive/masked-refinement model;
- implement per-task adaptation, augment/invert voting, and checkpointed batch
  inference;
- calibrate output shape and candidate confidence separately from cell color.

**Exit:** at least 3 unique shadow-holdout solves beyond symbolic-only and a
reproducible improvement over the current zero-contribution specialist.

### October 4-13: make memory and programs useful

- expand the typed grammar only from observed residual families;
- add residual-guided mutation/crossover of executable programs;
- rebuild the frozen hyperbolic bank from successful builder traces;
- run memory, recursion, TTA, augmentation, and channel ablations.

**Exit:** memory and refinement each contribute independently measured exact
solves; otherwise simplify or remove them.

### October 14-21: ensemble and pass@2 optimization

- train/calibrate the verifier using leave-one-demonstration-out predictions;
- enforce equivariance and counterfactual consistency;
- select two candidates for complementary error profiles;
- optimize scheduling so cheap exact rules run first and expensive TTA receives
  the remaining budget.

**Exit:** pass@2 improves pass@1 by at least 15% relative and the full run stays
below 10.5 hours.

### October 22-26: release candidate

- freeze code, weights, memory, dependencies, and licenses;
- run two clean full simulations and one Kaggle commit-mode run;
- download and audit the produced artifact;
- accept competition rules before the October 26 entry deadline;
- prepare the open-source method and reproducibility write-up.

**Exit:** no mutable dependency, fallback import, missing telemetry, or
unreproducible asset remains.

### October 27-November 2: final submissions

- use the one-submission-per-day allowance only for predeclared experiments;
- retain a stable candidate and a higher-upside diverse candidate;
- select up to two final submissions based on exact evidence, not public-score
  chasing;
- submit before November 2, 23:59 UTC with recovery margin.

## 7. Compute strategy

Start lean, then scale only after exact-solve evidence:

| Tier | Purpose | Hardware | Promotion rule |
|---|---|---|---|
| CPU | deterministic, typed search, memory, unit/smoke tests | local CPU or Kaggle CPU | default development lane |
| Small GPU | recursive model training and fast ablations | one 16-24 GB GPU | only if it adds held-out exact solves |
| Kaggle T4x2/P100 | integrated TTA and submission rehearsal | competition notebook | default contender if under 10.5 h |
| Kaggle L4x4 | larger batch/ensemble or model bakeoff | 96 GB pool; double quota cost | use only when measured score/runtime gain justifies it |

Do not make a 20B-30B language model the first dependency. A language model may
later propose typed programs or repairs, offline or in a bounded Kaggle branch,
but it remains only if it contributes unique exact solves per unit of runtime.

## 8. Stop conditions and honest confidence

The project is technically deployable now but is not currently competitive.
Confidence should rise only at these checkpoints:

- **deployment reliability:** high after the hermetic two-run gate;
- **nonzero score:** moderate after 2/120 clean exact pass@2;
- **competitive entry:** moderate after 6/120 with stable ablations;
- **placement contender:** credible only after at least 12/120 on untouched
  evaluation and a similar result on an independent shadow distribution;
- **likely placement:** cannot be claimed without leaderboard evidence near the
  final competition distribution.

If the neural/TTA branch does not add exact solves by October 3, stop expanding
the symbolic architecture and reproduce a stronger permitted open baseline as
a control. If the hybrid remains below 6/120 by October 14, prioritize a valid,
well-documented research entry over an unsupported placement claim.

## 9. Research basis

- [ARC Prize 2026 ARC-AGI-2 code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)
- [ARC Prize 2026 schedule and goals](https://arcprize.org/competitions/2026)
- [Official ARC-AGI-2 repository and data discipline](https://github.com/arcprize/ARC-AGI-2)
- [ARC Prize 2025 results and refinement-loop analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [SOAR: self-improving evolutionary program synthesis](https://arxiv.org/abs/2507.14172)
- [MindsAI ARC Prize 2025 implementation](https://github.com/jcole75/arc_2025_mindsai)

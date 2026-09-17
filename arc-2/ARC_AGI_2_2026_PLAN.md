# ARC-AGI-2 2026 Implementation Plan

Status: Experience-bank Version 2 implemented and packaged; external solvers are reference-only

## Decision

Build a neuro-symbolic recursive world-model solver. Keep the current DSL/MCTS
implementation as a regression baseline and submission-format harness, but do
not continue treating it as the product architecture.

The target loop is:

```text
demonstrations
  -> perceptual/object state
  -> transformation hypotheses
  -> executable program/world model
  -> exact demonstration replay
  -> residual diagnosis
  -> revised hypotheses
  -> two diverse, verified test predictions
```

The language or recursive model is a hypothesis generator. The typed executor,
demonstration replay, and submission validator are the authority.

## Capability-amplification contract

The contender is not an attempt to ask a stronger language model for the final
grid. Its purpose is to carry a fixed, bounded model across the finish line by
supplying capabilities that the model does not reliably possess on its own:

- canonical pixel, object, relation, role, and invariant perception;
- a closed relational plan language and deterministic parameter binding;
- typed execution, bounded search, simulation, and exact replay;
- residual localization, recursive repair, and behavioral deduplication;
- versioned schema/trace/failure memory and task-conditioned retrieval;
- two verified, behaviorally distinct predictions and fail-closed submission.

The model may rank roles, propose plans, select subgoals, or allocate search
budget. It may not directly authorize an answer, execute generated code, inspect
test solutions, bypass replay, or write persistent competition memory from a
validation or hidden task.

Progress is measured as **system lift** with model weights and decoding held
fixed:

| Ablation | Purpose |
|---|---|
| Typed system without a model | Establish the deterministic/world-model floor |
| Fixed model with the v1 wrapper contract | Establish the current proposer floor |
| Same model with relational-plan v2 | Measure representation and binding lift |
| Relational system without retrieval/repair | Isolate planning and execution |
| Relational system plus retrieval | Measure memory lift |
| Full system plus residual repair | Measure recursive-control lift |

For every ablation, report exact pass@1/pass@2, unique exact solves, regressions,
proposal parse/replay rate, timeout/fallback rate, wall/CPU/RAM/GPU use, and
attempt diversity. A larger model is justified only after the system produces
measurable lift with the fixed small model or exposes a specific capacity limit
that model scale can address.

## Competition target

- Produce `submission.json` with ordered `attempt_1` and `attempt_2` grids for
  every test input of every hidden task.
- Score only exact full-grid matches.
- Complete the hidden rerun within 12 hours on 4 x L4 GPUs, without internet.
- Use only rule-compliant public data, weights, and dependencies.
- Keep enough architectural originality and evidence to support the required
  open-source technical report if the approach becomes competitive.

Official references:

- [ARC-AGI-2 competition](https://arcprize.org/competitions/2026/arc-agi-2)
- [Kaggle overview and data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
- [Kaggle rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)
- [ARC-AGI-2 data](https://github.com/arcprize/ARC-AGI-2)

## Current project assessment

The existing project has useful production mechanics:

- challenge discovery and parsing;
- grid and submission validation;
- checkpoint/resume behavior;
- deterministic task seeds;
- a typed spatial DSL and exact executor;
- 126 passing unit/integration tests as of 2026-08-31.

It does not yet implement the intended solver:

- search is a flat primitive/action enumeration with a short linear program;
- program memory mean-pools primitive names, discarding order and parameters;
- the Poincare projection does not learn semantic or hierarchical structure;
- no object/relationship graph drives hypotheses;
- no recursive residual diagnosis or explicit world-model revision exists;
- `attempt_2` is normally the unchanged input, not an independent hypothesis;
- prior smoke evidence found only two exact training fits while many tasks
  collapsed to identity programs.

These are architectural limits, not test-suite failures. Preserve the tests and
replace the center of the solver behind stable interfaces.

## Contender architecture freeze

The contender will be built beside the legacy solver under
`hyper_arc/contender/`. The existing runner, submission validator, deterministic
rules, verified-symbolic adapter, and MCTS implementation remain available as
independent channels and regression baselines. They do not define the new
architecture.

Five decisions are fixed for the first contender release:

1. **Exact replay is the authority.** A hypothesis cannot reach a test input
   unless it reproduces every visible training output exactly.
2. **Memory stores structured experience.** Programs, parameters, task
   signatures, traces, residuals, and outcomes are canonical. Embeddings and
   Poincare coordinates are derived indexes that can be rebuilt.
3. **No answer leakage.** Reusable memory is built only from the designated
   development split. Validation and public-evaluation answers never enter the
   bank or proposer context.
4. **Retrieval proposes; it never decides.** Retrieved schemas must be
   instantiated and reverified against the current task's demonstrations.
5. **Pass@2 must be genuinely diverse.** The second attempt is a distinct
   demonstration-consistent behavior, not identity or a duplicate unless no
   second valid hypothesis exists.

### Runtime dataflow

```text
current demonstrations
  -> canonical pixel/object/relation state
  -> task signature
  -> retrieve analogous transformation schemas
  -> instantiate + deterministic generation + typed search
  -> execute on every demonstration
  -> exact? retain candidate
  -> not exact? classify residual and recursively repair the smallest cause
  -> behavioral deduplication and stability ranking
  -> two distinct verified predictions
  -> submission validator
```

The offline path is separate:

```text
permitted public development tasks
  -> exact solver traces and verified programs
  -> normalize variables, colors, positions, and object identities
  -> extract reusable transformation schemas
  -> validate on source demonstrations and mutations
  -> versioned competition-static seed bank
```

## Data and contamination protocol

Create a versioned split manifest before contender development:

- **memory-development:** 800 of the 1,000 public training tasks; may be used
  to mine schemas, train a specialist, and tune retrieval;
- **internal holdout:** 200 public training tasks; answers available only to
  the evaluator and never written to memory;
- **official public evaluation:** all 120 public evaluation tasks; sealed local
  benchmark used only at release gates;
- **competition hidden tasks:** inference only; no reusable writes during the
  Kaggle scoring run.

The 800/200 split must be deterministic, hash-addressed, and stratified using
answer-independent structural descriptors where possible. The manifest records
dataset revision, task IDs, purpose, and whether labels may be consumed by each
component.

## Seed bank v1

The existing `build_global_memory.py` is not sufficient: it samples 50 tasks,
uses the weak flat DSL, stores only exact programs that DSL happens to find,
and embeds them with an order- and parameter-destroying representation. Replace
it after the typed AST exists.

Each v1 memory record contains:

```text
record_id, schema_version, source_task_id, source_split, dataset_revision
task_signature, object_roles, relations, invariants
program_ast, bound_parameters, preconditions, postconditions
source_trace, mutation_results, exact_replay=true
complexity, runtime_ms, generator_channel, license, provenance
```

The initial bank is mined from exact solutions produced on the 800-task
development split by:

1. existing deterministic rules;
2. the pinned verified-symbolic ensemble;
3. the new typed enumerator;
4. later, a model proposer whose output passes the same executor and verifier.

Normalize task-specific colors, coordinates, object IDs, and sizes into typed
variables before storing a schema. Reject a record when its program does not
survive source replay or rule-preserving mutations. Start with an ordinary
inverted feature index plus exact structural filters. Add hyperbolic retrieval
only after an ablation shows unique held-out solves beyond the structured
index.

## Typed contender contracts

The first implementation introduces these stable records:

- `TaskState`: demonstrations, pixel features, object graph, correspondences;
- `ObjectState`: mask, color role, geometry, topology, parent/neighbor links;
- `Relation`: typed spatial, topological, count, symmetry, and mapping facts;
- `ProgramAST`: typed executable operations, variables, scopes, and registers;
- `Hypothesis`: program, bindings, provenance, predictions, confidence;
- `Trace`: every executed step and intermediate state;
- `Residual`: unmatched cells, objects, relations, and suspected failed cause;
- `MemoryRecord`: generalized verified experience with versioned provenance.

The narrow service boundaries are:

```text
Perception.parse(task) -> TaskState
Memory.retrieve(task_signature, k) -> list[MemoryRecord]
Generator.propose(task_state, retrieved) -> iterator[Hypothesis]
Executor.run(program, grid, bindings) -> Trace
Verifier.replay(hypothesis, demonstrations) -> VerificationResult
Repair.refine(hypothesis, residual) -> iterator[Hypothesis]
Ranker.select_pass2(exact_hypotheses) -> (attempt_1, attempt_2)
```

## Search and reasoning policy

Search proceeds from cheapest to most expensive:

1. exact verified rules and normalized deterministic transforms;
2. retrieved schema instantiation;
3. typed enumeration constrained by the task's object/relation deltas;
4. recursive residual repair of near-miss structured hypotheses;
5. one offline model proposer emitting only the typed schema;
6. legacy MCTS as a bounded independent fallback, not the main engine.

Partial cell accuracy may prioritize which failed hypothesis to repair. It may
not promote a candidate to prediction status. Residual classes include wrong
canvas construction, missed object selection, wrong correspondence, incorrect
parameter binding, incomplete iteration, wrong ordering, and local pixel
cleanup.

## Runtime budget

The release target is 9.5 hours or less on the official 12-hour limit, leaving
margin for Kaggle variance and packaging overhead.

| Stage | Full-run budget | Failure action |
|---|---:|---|
| Load, parse, validate inputs | 5 minutes | Fail closed |
| Perception and task signatures | 20 minutes | Fall back to pixel view |
| Retrieval and schema instantiation | 20 minutes | Continue without memory |
| Deterministic and typed search | 3 hours | Per-task budget cutoff |
| Recursive repair | 2 hours | Stop on repeated residual/state |
| Model proposal, if retained | 3 hours | Fixed token/time ceiling |
| Ranking, output, validation | 15 minutes | Fail closed |
| Reserved margin | 1 hour 45 minutes | Not allocatable |

Every channel logs wall time, peak memory, candidates executed, exact fits,
unique held-out solves, and fallback reason. A channel that adds no unique
exact solves is removed from the scoring notebook.

## Measured gates

The current pinned floor is 2/120 public-evaluation tasks at pass@2 from the
verified-symbolic ensemble. The contender advances only through these gates:

| Gate | Exit criterion |
|---|---|
| Evaluation harness | Reproducible pass@1/pass@2, per-output scoring, channel attribution, runtime and failure reports |
| Perception | Invariance tests pass for color permutations, translations, and irrelevant canvas padding |
| Seed bank v1 | At least three unique exact internal-holdout solves beyond the no-memory ablation |
| Typed AST | At least ten unique exact internal-holdout solves and zero accepted non-exact programs |
| Recursive repair | At least 20% relative gain in unique internal-holdout solves over typed search alone |
| Model proposer | Adds at least five unique public-evaluation solves within its runtime allocation or is excluded |
| Kaggle release | Two clean offline runs, valid outputs for every test input, and runtime below 9.5 hours |

These are kill gates, not aspirational reporting metrics. If a component misses
its gate, simplify or remove it before adding another component.

## First seven-day implementation slice

1. Freeze the Version 6 archive hash, Kaggle version, logs, and score as the
   legacy baseline.
2. Add the deterministic 800/200 split manifest and contamination registry.
3. Build an exact evaluation ledger with pass@1, pass@2, per-output results,
   channel attribution, and runtime.
4. Add the typed records and serialization contracts without changing solver
   behavior.
5. Implement canonical pixel/object/relation perception with invariance tests.
6. Define the typed AST and migrate existing deterministic primitives into it.
7. Build seed-bank mining from exact deterministic and verified-symbolic
   traces; do not yet add learned embeddings.
8. Run the no-memory versus structured-memory ablation on the fixed internal
   holdout.

At the end of this slice, continue to recursive repair only if the evaluator,
typed executor, provenance, and structured retrieval are trustworthy.

### Implementation checkpoint — 2026-08-31

- Completed: frozen Version 6 artifact hashes, archive inventory, local
  submission validation, and reproduced public-evaluation floor of 2/120 tasks.
- Completed: hash-versioned 800/200 development/holdout split, source hashes,
  write-scope policy, and contamination registry.
- Completed: strict pass@1/pass@2 evaluator with per-output results, optional
  channel attribution, measured wall/CPU/peak-RSS/CUDA telemetry, explicit
  timeout/fallback/failure counts, coarse output-size-family metrics, and an
  append-only JSONL ledger.
- Completed: typed task, object, relation, program AST, trace, residual,
  hypothesis, and memory records under the isolated contender package.
- Completed: canonical grid/object perception with color- and translation-
  invariant signatures, 4/8-connectivity, holes, sparse spatial relations,
  symmetry, periodicity, separators, correspondences, and example deltas.
- Completed: typed multi-register execution, closed operation dispatch, exact
  traces/residuals, and adapters reproducing all legacy deterministic forms.
- Bloat gate: the first all-pairs relation design produced 21,414,310 edges in
  141.5 seconds across the training corpus. Sparse spatial order, touching,
  and immediate containment reduced this to 131,673 edges in 8.3 seconds.
- Completed: object-set selection, per-object typed mapping, recolor/translation,
  blank and input-sized canvas construction, rendering, typed conditionals,
  constrained registers, and executable pre/postconditions.
- Completed: stable AST complexity cost, strict JSON round-trip serialization,
  closed operation dispatch, and node/depth/trace/iteration execution budgets.
- Completed: seed-bank v1 mined 12 normalized legacy schemas from 13 successful
  development-task outcomes, with task-local color binding and exact replay.
- Failed value gate: on the untouched 200-task holdout, no-memory and structured
  memory both solved 7 tasks at pass@2. Memory added zero unique solves and
  caused zero regressions, below the required three-unique-solve threshold.
- Decision: preserve seed-bank v1 as negative evidence but do not integrate it
  into the competition runtime. The next bank must mine richer object programs.
- Completed: typed object-program v1 mines exact crop, extract, erase, recolor,
  move, copy, and move-recolor programs with selector and execution budgets.
- Validation result: on the 160-task nested development fold, stable-object
  improved deterministic pass@2 from 5 to 8 (+3 unique, zero regressions).
- Holdout result: on the untouched 200-task internal holdout, stable-object
  improved deterministic pass@2 from 7 to 8 (+1 unique, zero regressions).
- Production-aware gate failed: the verified-symbolic plus deterministic
  ensemble scored 69/200 before and after object-channel insertion. The one
  deterministic-only gain was already covered, so object v1 remains isolated.
- Provenance correction: the pinned verified-symbolic repository predates the
  split and explicitly names 111 IDs from the 1,000-task training corpus,
  including 18 IDs in the 160-task validation fold. All 18 of those named fold
  tasks are solved by that channel. Its 57/160 validation and 69/200 holdout
  scores are therefore training-contaminated diagnostics, not held-out
  generalization evidence. Imported pre-split channels may run in the
  competition ensemble, but must be excluded from claims about clean internal
  validation gains.
- Runtime: bounded perception caching plus no-trace/first-mismatch candidate
  filtering reduced nested-validation time from 253.86 to 17.56 seconds.
- Completed: residual-repair v1 classifies structured failures and recursively
  wraps only error-reducing typed programs. Demonstration-behavior
  deduplication reduced 45 syntactic exact variants to 8 distinct behaviors
  (5 revised), while preserving the improvement from 8 to 9 typed-object
  validation tasks. Production plus repair improved from 57 to 58 (+1 unique,
  zero regressions), still below the 20% integration gate.
- Completed: a fail-closed local model proposer accepts only operation-specific
  JSON-schema ASTs and exact demonstration replays. On three production-unsolved
  validation tasks, Qwen2.5-Coder 7B and Qwen3 8B both produced zero exact fits;
  Qwen3 was also substantially slower and less schema-compliant.
- Diagnosed: those tasks (`c62e2108`, `5adee1b2`, and `db118e2a`) require
  marker-conditioned motif replication, key-driven region framing, and
  frame/interior extraction plus repeated placement. The v1 proposal contract
  exposes only whole-grid wrappers, so larger models cannot express the needed
  programs. Define and test a closed relational-plan v2 before acquiring the
  remaining 20B/30B models.
- Completed: recursive-specialist v1 is a 135,750-parameter shared-weight grid
  model trained on 640 builder tasks only. Loss and partial cell accuracy
  improved, but 0/160 validation tasks passed all leave-one-demo-out checks, so
  no specialist prediction was eligible for the ensemble.
- Measured: the 160-task diagnostic run took 155.77 wall seconds and 155.16 CPU
  seconds, peaked at 317,673,472 RSS bytes, used no CUDA memory, and recorded no
  timeout, fallback, or failure. Coarse size-family production pass@2 was
  14/31 contract, 6/12 expand, 0/1 mixed-axis, 1/5 mixed-demonstration, and
  36/111 same-size; the sole repair gain was in the same-size family. These
  production-family counts inherit the verified-symbolic contamination caveat.
- Verified: the completed Kaggle kernel processed all 240 hidden-test tasks and
  259 test inputs in 20,908.7 seconds, emitted a 264,927-byte submission with
  ordered attempt fields, and passed independent local structural validation.
  The packaged production files match the current local production files.
- Verified: contender lint is clean. Production now imports only the promoted
  typed relational runtime; the contender package has no eager experimental
  imports, and the Kaggle archive excludes model, repair, seed-bank, training,
  and evaluation modules.
- Completed: relational-plan v2 adds three closed, deterministic plan families:
  marker-conditioned motif propagation, key-conditioned object framing, and
  normalized frame/interior extraction with repeated placement. Parameters are
  bound from demonstrations and candidates enter the ensemble only after exact
  replay of every demonstration.
- System-lift result: holding Qwen2.5-Coder 7B and the three development cases
  fixed, wrapper-v1 solved 0/3, model-selected relational-v2 solved 2/3, and
  bounded deterministic enumeration solved 3/3. The model chooses only a
  closed plan name; it cannot emit executable code or unbound parameters.
- Transfer result: relational-v2 added zero non-design solves on the 160-task
  validation diagnostic, zero unique solves on the 120-task public evaluation,
  and zero unique solves on the exhausted 200-task holdout. It caused zero
  regressions on all three sets. These results do not establish generalization.
- Deployment decision: expose relational-v2 only as a fail-closed upside
  attempt-2 channel and preserve the previous production attempt as attempt 1.
  No model or API is required in the Kaggle runtime. Full local suite: 186
  passed; runtime-source lint clean; archive SHA-256
  `3b38aad14a41ac490763307beeaeaae0bd26cd9d4edc5e48d56cb181b4ceab3c`.
- Kaggle release: private dataset version containing that archive was uploaded
  and downloaded back with matching source hashes. Kernel Version 7 completed
  all 240 hidden tasks and 259 test inputs in 20,909 seconds. Its 264,927-byte
  `submission.json` passed the notebook post-flight and an independent local
  structure/grid audit; 137/259 attempt pairs are distinct and the file SHA-256
  is `e0851f2822ab490fc5040d7658d7f05099e4f8d04ae03e0d1316a92f796d637a`.
  Kaggle submission 55973568 was created from exact kernel Version 7 at
  2026-09-03 03:30 UTC and completed with public score `0.00`. This is the
  external generalization verdict: the relational-v2 channel produced no
  hidden exact solve, so it remains a validated mechanism but failed its
  competition-value gate.
- External-reference boundary: the public Apache-2.0 NVARC Qwen3-4B notebook
  remains useful comparative evidence—its published ARC Prize 2025 page reports
  24.44 private and 23.33 best scores with a 26m35s L4x4 run—but it is not a
  submission backbone for this project. Provenance remains recorded in
  `config/external_nvarc_qwen_2026.json` solely for research traceability.
- Withdrawn experiment: two private NVARC compatibility runs were attempted;
  neither reached model execution or produced a submission. The private Kaggle
  copy was deleted, its automation path removed, and Linear NEU-228 canceled.
  No NVARC-derived competition submission was made. Future competition entries
  must run under the Hyper-ARC identity and consist of our implementation;
  external systems may inform comparison and general lessons only.
- Evidence policy: use hidden Kaggle scoring as the highest-confidence external
  measure. Treat the existing 2/120 public-evaluation result as the current
  reproducible generalization floor. Treat training-derived validation and
  holdout comparisons only as diagnostic until every participating channel has
  auditable pre-split provenance or a new untouched dataset is available.
- Still open: broader relational families with clean external transfer evidence.
- Completed: experience-bank v2 mines 1,920 versioned records from the frozen
  640-task builder fold: one deterministic, object, and relational outcome per
  task. Records include color-invariant structural fingerprints, typed programs
  for successful abstractions, exact-replay status, candidate counts, failure
  modes, hashed source provenance, and a tamper-evident bank digest.
- Runtime integration: retrieval ranks analogous success and failure evidence;
  stored programs are replayed through the closed typed executor; task-local
  object and relational programs are rebound; only exact demonstration fits may
  produce a test prediction. Validation and inference never write to the bank.
- Owned-channel evidence: 8/160 to 11/160 on nested validation (+3 unique) and
  7/200 to 8/200 on the exhausted internal holdout (+1 unique), with zero
  regressions. Against the contaminated reference ensemble it produced +4 on
  validation and +0 on holdout, so those figures are diagnostic only.
- Clean public-evaluation gate: the frozen owned system solved 0/120 and memory
  generated no exact candidates. The separate reference benchmark remained
  2/120. Therefore memory-v2 is a working deployment baseline, not yet a
  competitive generalizer, and no claim of expected leaderboard lift is valid.
- Ownership correction: the production runner and `solver_code.zip` contain no
  verified-symbolic/NVARC code. The external repositories remain benchmark and
  research references only. The packaged bank has ID `3501eb3591c7...`; lint is
  clean, 189 tests pass, and a fresh-extraction two-output submission smoke test
  passed.
- Still open: broader learned program families and clean external transfer
  evidence. The bank is now the production reasoning substrate, not an optional
  MCTS prior.
- World-model v1 checkpoint (2026-09-06): added an owned, task-conditioned
  Poincare-ball memory index, order- and argument-sensitive typed-program
  embeddings, recursive world-state simulation, structured residual states,
  strictly error-reducing typed repairs, pass@2 behavioral diversity, and
  session-only exact-demonstration learning. External verified-symbolic/NVARC
  code is not imported by this path.
- Exact nested-validation result: the combined owned policy improved from
  8/160 to 12/160 tasks with zero regressions. The isolated world-model channel
  solved 11/160. Four wins were unique relative to the deterministic plus
  relational comparison policy, but three were existing object-program wins;
  recursion supplied the fourth (`d2abd087`) by composing an object recolor
  program with a residual color-remap repair.
- Causal ablation: disabling hyperbolic retrieval left the result unchanged at
  11/160 world-model and 12/160 combined. Disabling both retrieval and recursion
  reduced those figures to 10/160 and 11/160. Thus recursion has one exact
  validation contribution; hyperbolic memory has none yet.
- Exhausted internal-holdout result: combined improved from 7/200 to 8/200 with
  zero regressions, but the sole unique task (`810b9b61`) was an already-known
  object-selector capability rather than a recursive or memory transfer win.
  This is not new architectural lift and does not justify a Kaggle submission.
- Public-evaluation result: 0/120 exact tasks for the world-model and combined
  owned policies; retrieval ran on all tasks but yielded no demonstration-exact
  hypothesis. This confirms no externally measured transfer gain.
- Deployment decision: keep world-model v1 isolated from `main.py`. Promote it
  only after hyperbolic retrieval contributes an independently attributable
  exact win and the combined policy clears a clean external/public gate.

## Component design details

### A. Perception and canonical state

For every grid, compute both pixel and object views:

- dimensions, palette, background candidates, symmetry, repetition;
- connected components under 4- and 8-connectivity;
- object masks, colors, bounding boxes, holes, containment, adjacency;
- rows, columns, lines, panels, separators, periodicity, and correspondences;
- input/output object matching and explicit deltas.

The canonical task record remains ordinary typed data with provenance. It is
not a learned embedding and does not depend on PHCG geometry.

### B. Typed transformation grammar

Replace the one-state linear DSL with a typed AST that supports:

- object selection by relational predicates;
- transformation of a selected object or set;
- construction of new canvases and multiple working registers;
- composition, iteration over objects, conditionals, and constrained variables;
- color, geometry, topology, counting, pattern completion, and correspondence;
- explicit preconditions, postconditions, complexity cost, and execution trace.

Existing primitives can become leaves in this grammar. Do not throw away their
tests.

### C. Recursive hypothesis loop

Each hypothesis contains:

```text
task features + selected entities + typed program + predicted demonstrations
+ residual map + confidence + provenance
```

At each recursion:

1. generate structurally plausible candidates;
2. execute against every training example;
3. reject candidates with type, dimension, palette, or exact-output violations;
4. classify residuals by object, location, color, and relationship;
5. repair the smallest failed assumption or program fragment;
6. stop on exact replay, repeated state, exhausted budget, or confidence floor.

Use minimum-description-length cost after exactness, never instead of exactness.

### D. Candidate ensemble and pass@2

Generate candidates through independent channels:

1. deterministic object/relationship rules;
2. enumerative typed-program search;
3. a small recursive grid specialist;
4. an offline code/reasoning model that emits only the typed schema;
5. retrieval of previously successful transformation schemas.

Deduplicate by behavior across all demonstrations. Rank exact candidates by
structural fit, simplicity, perturbation stability, and channel agreement. The
two output attempts should be the two best behaviorally distinct exact
hypotheses—not the same program twice and not identity by default.

### E. Memory and PHCG boundary

Use three scopes:

- task-local: current hypotheses, traces, residuals, and rejected repairs;
- competition-static: abstractions learned only from permitted development
  data, frozen before hidden evaluation;
- PHCG export: generalized concepts, transformation schemas, failure patterns,
  evidence, and outcomes after an experiment is complete.

PHCG retrieval returns candidate schemas and analogous task features. It never
overrides demonstration evidence. Store stable logical IDs and versioned
records; derive color, embedding, and hyperbolic position from those records.

## Model options

Approximate weight memory excludes runtime overhead and must be measured in the
actual Kaggle image.

| Option | Role | Approximate fit | Decision |
|---|---|---:|---|
| TRM-style ~7M recursive specialist | Direct grid candidate and ranker | Tiny | Build as a cheap independent branch |
| Qwen3-8B, Apache 2.0 | Schema/program proposer | ~16 GB BF16; ~5 GB 4-bit | Fast baseline and ablation model |
| gpt-oss-20b, Apache 2.0 | Reasoning/program proposer | 21B total, 3.6B active; quantize | Preferred lean reasoning candidate |
| Qwen3-Coder-30B-A3B, Apache 2.0 | Python/typed-AST proposer and repairer | ~61 GB BF16; quantized is comfortable | Preferred code-specialist candidate |
| DeepSeek-R1-Distill-Qwen-14B, MIT | Alternate reasoning branch | ~30 GB BF16 | Optional benchmark, not first build |
| Inkling-Small, Apache 2.0 | Research ceiling | 276B total/12B active; ordinary 4-bit weights alone are ~138 GB | Exclude from initial Kaggle build |

Inkling-Small's verified 40.1% ARC-AGI-2 semi-private score makes it important
research evidence, but active parameter count is not storage size. It does not
fit the 96 GB pool under ordinary four-bit storage before runtime overhead.

Model sources:

- [Inkling-Small model card](https://huggingface.co/thinkingmachines/Inkling-Small)
- [Verified Inkling-Small ARC results](https://arcprize.org/results/thinky-inkling-small)
- [gpt-oss-20b model card](https://openai.com/index/gpt-oss-model-card/)
- [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

Initial bakeoff: Qwen3-8B vs gpt-oss-20b vs Qwen3-Coder-30B-A3B, using the same
typed output contract and inference budget. Keep only models that improve exact
held-out solves enough to justify their time and memory. Do not use direct grid
generation as the main path.

## Validation design

### Metrics

Primary:

- exact task accuracy;
- exact test-grid accuracy with both attempts;
- pass@1 and pass@2;
- fraction with at least one exact demonstration-consistent hypothesis.

Diagnostic:

- object extraction accuracy;
- residual cell/object counts;
- hypothesis executions per exact solve;
- fallback and timeout rate;
- seconds and peak memory per task;
- contribution and unique solves by ensemble channel.

### Data discipline

- Create fixed development and validation folds before model work.
- Stratify by transformation family and output-size behavior where possible.
- Never train or write reusable memory from validation answers.
- Maintain a contamination registry for every external dataset and checkpoint.
- Run mutation tests: change colors, positions, counts, and canvas sizes while
  preserving the inferred rule. A genuine program should remain valid.

### Gates

1. **Plumbing gate:** a clean notebook emits a structurally valid submission.
2. **Representation gate:** object and relation features are invariant under
   irrelevant color/translation changes.
3. **Executor gate:** every accepted program exactly replays all demonstrations.
4. **Value gate:** each new channel adds held-out exact solves or is removed.
5. **Runtime gate:** full simulation stays below 9.5 hours, reserving margin
   against the 12-hour hard limit.

## Schedule

| Dates | Deliverable | Exit criterion |
|---|---|---|
| Aug 30-Aug 31 | Join, obtain data, deploy Version 6 baseline | Complete: valid notebook, private dataset, and measured deterministic floor |
| Sep 1-Sep 7 | Split manifest, evaluator, typed records, perception, seed-bank v1 | Fixed reports, invariance tests, and memory/no-memory ablation |
| Sep 8-Sep 16 | Typed AST, executor, and constrained synthesis | Demonstration-exact synthesis adds unique internal-holdout solves |
| Sep 17-Sep 25 | Residual classification and recursive repair | At least 20% relative gain over typed search alone |
| Sep 26-Oct 5 | One model-proposer bakeoff and optional recursive specialist | Retain only channels with unique exact public-evaluation solves |
| Oct 6-Oct 18 | Pass@2 diversity, retrieval ablations, runtime optimization | Stable ensemble below 9.5-hour runtime budget |
| Oct 19-Oct 23 | Failure mining, robustness mutations, full clean runs | Release candidate passes all correctness and contamination gates |
| Oct 24-Oct 26 | Freeze models/dependencies; final entry/team deadline | Reproducible release candidate |
| Oct 27-Nov 2 | Two final full runs and submission selection | Stable and upside submissions selected |

## Post-sprint backlog

1. Add residual-driven repair and behavioral deduplication after the first
   seven-day slice passes its gates.
2. Replace identity `attempt_2` with a genuinely distinct ranked candidate.
3. Add one small model adapter only after the deterministic loop is measurable.
4. Add a recursive specialist only if its unique-solve/runtime ablation passes.
5. Integrate PHCG retrieval last, through the narrow schema-retrieval interface.

## Explicit non-goals for version one

- no free-form agent framework inside the scoring notebook;
- no online API calls;
- no learned Poincare coordinates as the source of truth;
- no hidden-evaluation writes to global memory;
- no model ensemble without unique-solve ablation evidence;
- no visualization work until it helps diagnose solver errors.

# Hyper-ARC Kaggle Reference Architecture

**Status:** implementation contract for the ARC Prize 2026 contender  
**Scope:** ARC-AGI-2 competition runtime, training pipeline, evaluation, and
reproducible open-source release  
**Not in scope:** the general Triarch platform, personal/collective PHCG,
commercial orchestration, customer data, or the private product roadmap

> Phase A implementation evidence and the capability/no-submission decision are
> recorded in `PHASE_A_KAGGLE_READINESS_REPORT.md`.

## 1. Architectural decision

Build one cohesive solver, not a collection of agents or independent guesses.
The competition runtime is a **single-process modular monolith** with one shared
task state and one recursive hypothesis loop:

```text
ARC task
   |
   v
deterministic perception ---> TaskState / WorkingGraph (ephemeral)
                                      |
                                      v
                     initial typed hypotheses and latent proposals
                                      |
                                      v
               execute -> measure residual -> revise / retrieve / specialize
                    ^                                      |
                    |                                      v
                    +----------- bounded recursion --------+
                                      |
                                      v
                exact verifier + diversity-aware pass@2 selector
                                      |
                                      v
                              submission.json
```

Triarch supplies a useful separation of responsibilities without requiring the
whole platform in Kaggle:

- **Neuro:** schedules bounded reasoning operations using uncertainty, expected
  information gain, and remaining time.
- **Aegis:** rejects invalid or unsupported hypotheses through exact replay,
  invariance checks, leave-one-out validation, and resource limits.
- **Morphos:** runs offline only. It learns generators, curates procedural
  memory, performs ablations, and promotes only improvements that pass held-out
  gates.

This division is functional rather than theatrical. There are no communicating
LLM personas, network services, mutable production memory, or distributed graph
databases in the Kaggle notebook.

## 2. Core research hypothesis

ARC tasks can be solved more efficiently when one compact recursive model and
one typed executor share a common representation of objects, transformations,
and residual errors, while a small hyperbolic procedural memory supplies
task-conditioned priors. Exact symbolic execution supplies correctness;
learned recursion supplies search guidance and repair; memory supplies reusable
experience; the controller spends computation only where uncertainty remains.

Every clause is independently testable. A component remains in the contender
only when it adds untouched-task exact matches or materially reduces runtime at
equal accuracy.

## 3. One state, one loop, explicit contracts

### 3.1 `TaskState`

The sole mutable runtime object contains:

- canonicalized training and test grids;
- deterministic pixel, object, component, symmetry, repetition, topology, and
  relation views;
- palette mappings expressed relationally rather than as raw color identities;
- typed hypotheses and their parent/derivation lineage;
- exact per-pair residuals and violated constraints;
- retrieved procedural priors with provenance and similarity;
- elapsed time, candidate budget, recursion depth, and deterministic RNG state.

It is ephemeral. Nothing learned from hidden test tasks is persisted or shared
between competition runs.

### 3.2 Perception

Perception is deterministic and exhaustive within a fixed budget. It creates
multiple compatible views instead of prematurely choosing one segmentation:

- connected components under 4- and 8-connectivity;
- bounding boxes, masks, holes, borders, intersections, and containment;
- rows, columns, diagonals, panels, repeated tiles, and separators;
- translation, rotation, reflection, scaling, recoloring, counting, and
  correspondence features;
- object and scene graphs with stable canonical hashes.

All downstream components consume these same views. Learned perception may
rank views later, but may not erase deterministic alternatives.

### 3.3 Typed transformation graph

The executable language is a small typed transformation graph rather than a
large flat DSL. Initial types are:

- `Grid`, `Mask`, `Object`, `ObjectSet`, `Color`, `Vector`, `Region`, `Count`;
- selectors, relational predicates, transforms, compositions, and renderers;
- bounded loops only over finite grid objects or coordinates.

Each program must be total, deterministic, sandboxed, and traceable. Invalid
types and out-of-bounds effects are rejected before execution. The grammar
starts small; Morphos may promote a new primitive offline only after it solves
multiple training tasks, survives leave-one-out tests, and improves a frozen
validation split.

### 3.4 Generative recursive core

Use an 8-15 million parameter recurrent grid/object model implemented with
ordinary PyTorch operations. At each reasoning step it receives:

- the canonical task views;
- the current typed hypothesis embedding;
- executed candidate grids;
- structural residual maps and constraint violations;
- the retrieved procedural priors.

It returns a distribution over typed edits, program fragments, view choices,
and stopping confidence. Stochasticity creates several bounded trajectories,
but the outputs must pass the same exact executor and verifier. The model is a
search policy and repair engine, not an oracle that bypasses verification.

### 3.5 Procedural memory

The competition memory is a frozen, compact bank built only from permitted
public training tasks and generated synthetic tasks. A memory item stores:

- a canonical structural signature;
- a short typed procedure or procedure fragment;
- preconditions and known failure conditions;
- empirical success counts and validation provenance;
- an embedding used only to rank retrieval candidates.

Retrieval is two stage:

1. exact/coarse structural filtering prevents semantic-nearest but structurally
   impossible matches;
2. learned hyperbolic distance ranks the remaining hierarchical procedure
   families and specializations.

Hyperbolic geometry is not accepted on aesthetics. It ships only if it beats a
hash/index baseline and an equal-dimensional Euclidean embedding on frozen
retrieval and end-to-end exact-match ablations. The runtime math uses plain
PyTorch; no `geoopt`, graph database, or graph-neural-network dependency is
required.

### 3.6 Residual and counterfactual tests

Residuals are the common currency of the system. They describe wrong cells,
wrong shape, missing/excess objects, broken correspondences, palette mismatch,
and violated invariants. Both neural and symbolic revisions consume them.

Counterfactuals remain lean: apply only typed, task-preserving perturbations
such as palette permutations, translations inside available canvas space,
object-order permutations, and supported rotations/reflections. A hypothesis
loses confidence when its predicted equivariance fails. There is no free-form
counterfactual generator in the first contender.

### 3.7 Neuro controller

The controller chooses the next bounded operation from:

- execute a deterministic template;
- expand or repair a typed program;
- retrieve another procedure family;
- run another recursive trajectory;
- perform task-local adaptation when the training pairs justify it;
- verify and stop.

Scheduling uses a small table or learned value model based on exact training
fit, residual reduction, hypothesis novelty, uncertainty, elapsed time, and
historical yield. It may never exceed per-task or global budgets.

### 3.8 Aegis verifier and pass@2 selection

A candidate is eligible only if it:

- has valid dimensions and ARC color values;
- replays every available training pair exactly, or is explicitly admitted to
  a separately measured fallback tier;
- survives leave-one-out reconstruction when enough training pairs exist;
- satisfies its declared structural invariants and supported counterfactuals;
- is not merely an identity fallback unless identity is itself evidenced.

`attempt_1` maximizes calibrated expected exact match. `attempt_2` maximizes
expected exact match subject to a meaningful behavioral difference from the
first candidate. Identity is not a universal second attempt.

## 4. What is deliberately excluded

The following ideas do not enter the competition runtime without ablation
evidence:

- general-purpose LLM calls or internet access;
- multi-agent conversation or consensus;
- Neo4j, a vector service, or a persistent graph server;
- online self-modification or cross-task memory writes;
- unrestricted program synthesis;
- a full Triarch executive/security/governance deployment;
- full PHCG domain geometry or the personal knowledge taxonomy;
- custom CUDA, Triton, FlashAttention, `bitsandbytes`, `torch-geometric`, or
  `geoopt`;
- a large pretrained transformer included merely because GPU memory is
  available;
- any external solver copied or relabeled as Hyper-ARC.

## 5. Kaggle execution profile

### 5.1 Runtime target

- one Python process with bounded worker threads;
- Python standard library, NumPy, and PyTorch as the primary runtime surface;
- eager PyTorch is authoritative; optional compilation is never required;
- batch only compatible grid sizes or pad with explicit masks;
- weights target below 100 MB and all owned runtime assets below 1 GB;
- hard internal deadline of 10 hours, with the final two hours reserved for
  recovery, validation, and writing the submission;
- compact structured logs in `/tmp`; only `submission.json`, a run manifest,
  and bounded diagnostics go to `/kaggle/working`;
- no network access and no runtime package downloads.

### 5.2 Accelerator compatibility

| Environment | Required path | Numeric policy | Notes |
|---|---|---|---|
| Kaggle CPU | smoke, packaging, reduced solver | FP32 | must complete a small end-to-end fixture |
| P100 | full fallback | FP16 or FP32 | no BF16 assumption; standard CUDA ops only |
| T4 x2 | full contender | FP16 | single GPU is valid; second GPU is optional batching |
| L4 x4 | accelerated contender | BF16 preferred, FP16 fallback | multi-GPU is an optimization, never a requirement |

At startup the runner records Python, NumPy, PyTorch, CUDA, device capability,
free memory, selected dtype, and deterministic seed. Unsupported combinations
fail before task inference rather than silently disabling a channel.

### 5.3 Dependency policy

The runner must use packages already present in the Kaggle image whenever
possible. The official image changes over time, so exact versions are captured
by a commit-mode probe and never assumed from a local machine.

If a small missing dependency is indispensable, its Linux-compatible wheel is
included in the immutable bundle and installed with `--no-index --no-deps
--target /tmp/hyper_arc_deps`. Only the solver child process receives that path.
No global `pip install`, Pillow replacement, or environment-wide dependency
mutation is allowed.

## 6. Hermetic package and fail-closed startup

Publish and attach one immutable `hyper_arc_solver_bundle.zip`. Its manifest
contains a SHA-256 digest, source revision, model and memory hashes, license
inventory, configuration, expected file list, and smoke-test fixtures.

The notebook performs exactly these steps:

1. locate the bundle and verify its digest and every manifest entry;
2. extract it under `/tmp/hyper_arc_runtime`;
3. run import, executor, model-load, memory-retrieval, and verifier sentinels;
4. confirm each required candidate channel produces its expected fixture;
5. enumerate the actual ARC test keys and establish the global deadline;
6. invoke the solver once;
7. validate every task key, test index, grid value, and ordered pair of attempts;
8. atomically write `submission.json` plus a compact run manifest.

There is no loose-source fallback and no exception path that converts a missing
required subsystem into an empty candidate list. Optional channels are named in
the manifest and their absence is reported explicitly.

## 7. Training and promotion pipeline

Morphos is an offline pipeline, not a Kaggle service:

```text
public ARC data + licensed synthetic generators
                  |
             frozen split registry
                  |
       train policy / build procedural traces
                  |
       evaluate exact pass@1 and pass@2
                  |
    ablate, stress, contamination/license check
                  |
        Aegis promotion decision and freeze
                  |
         signed/hash-addressed solver bundle
```

Training tasks, augmentation descendants, synthetic generator families, and
validation tasks receive lineage IDs. Validation descendants may never enter
training or memory. Public leaderboard feedback is not training data.

Task-local adaptation may run in Kaggle using only that task's demonstrated
training pairs. It starts from the frozen model, is bounded, discards its state
after the task, and competes against the non-adapted path in the ablation suite.

## 8. Evidence gates

No architectural claim substitutes for an exact grid score.

| Gate | Promotion requirement |
|---|---|
| Unit | typed operators, hyperbolic math, budgets, and serialization pass property tests |
| Replay | every proposed program reproduces its claimed training outputs exactly |
| Component | each channel adds exact held-out solves or a measured runtime benefit |
| Geometry | hyperbolic retrieval beats hash/index and Euclidean controls |
| Integration | two independent full runs produce identical manifests and valid outputs |
| Runtime | full simulated run completes within 10.5 hours on the selected Kaggle accelerator |
| Kaggle | fresh `Save & Run All` succeeds with internet off and only declared inputs |
| Competition | public-score lift is independently observed; no lift triggers diagnosis, not narrative reinterpretation |

Required ablations are deterministic-only, recursive-only, memory-only,
deterministic+recursive, deterministic+memory, Euclidean-memory, hyperbolic-
memory, with/without task-local adaptation, and the complete system. Report
exact pass@1, pass@2, unique tasks solved, solve overlap, median/p95 runtime,
peak memory, and candidate count.

## 9. Open-source boundary and IP discipline

Prize eligibility and reproducibility mean the code and methods actually used
by the entry cannot be treated as a concealed trade secret. The defensible
boundary is **scope**, not obfuscation.

The public competition repository includes:

- this ARC-specific architecture and all code needed to reproduce it;
- training, evaluation, bundle, and Kaggle notebook entrypoints;
- owned frozen weights and procedural memory used by the submission, or a fully
  reproducible build process when redistribution rules require it;
- exact public/held-out results, ablations, limitations, model/data cards;
- dependency lock, software bill of materials, provenance, and licenses;
- a short research paper describing the ARC method accurately.

It excludes because they are not needed to reproduce the entry:

- the general Triarch runtime and its production orchestration contracts;
- the personal or commercial PHCG schema, data, security model, and deployment;
- collective/customer memory ingestion, governance, privacy, and tenancy;
- non-ARC domain adapters, business logic, and unpublished roadmap;
- private conversations, personal knowledge, credentials, and real-world data;
- experiments and inventions that are not used by the submitted solver.

Do not imply the public ARC repository is the complete Triarch or PHCG product.
Describe it as an **ARC-specific research implementation of selected principles**.
Do not hide a required competition method in opaque weights, undocumented data,
or a private build service. Before public disclosure, obtain qualified IP advice
if patent protection for any disclosed mechanism is desired.

Use a competition-compatible permissive license selected after verifying every
dependency and dataset. Maintain `THIRD_PARTY_NOTICES`, `LICENSES/`, an SBOM,
and a machine-readable provenance registry. External research may inform the
design, but third-party solver code is included only when licensing, attribution,
and authorship are unambiguous. Hyper-ARC claims only owned implementation and
measured results.

## 10. Repository shape

```text
hyper-arc/
  README.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  pyproject.toml
  docs/
    architecture.md
    method.md
    results.md
    limitations.md
    model-card.md
    data-card.md
  src/hyper_arc/
    state.py
    perception/
    grammar/
    executor/
    recursive/
    memory/
    controller/
    verifier/
    runtime/
  training/
    splits.py
    generators/
    traces/
    train_recursive.py
    build_memory.py
    promote.py
  evaluation/
    exact.py
    ablations.py
    contamination.py
    kaggle_sim.py
  kaggle/
    notebook.ipynb
    build_bundle.py
    runtime_probe.py
  tests/
  manifests/
```

This is a target layout, not authorization for a disruptive rewrite. Existing
working code should migrate behind these contracts incrementally.

## 11. Build order

### Phase A: make the existing runner trustworthy

1. Remove loose-source and silent-degradation paths.
2. Add the immutable manifest, subsystem sentinels, deadline, and submission
   validator.
3. Produce a clean Kaggle environment probe and full dry-run artifact.
4. Preserve current solvers as a measured baseline; make no score claim.

### Phase B: establish the common state and executor

1. Implement `TaskState`, deterministic multi-view perception, residuals, and
   typed transformation graph.
2. Port only existing operators that pass exact replay and property tests.
3. Establish pass@1/pass@2, overlap, runtime, and identity-rate reports.

### Phase C: make memory real

1. Build successful procedural traces with provenance and failure conditions.
2. Implement structural filtering plus a simple index baseline.
3. Train Euclidean and hyperbolic rankers under identical conditions.
4. Ship the winner only if end-to-end exact-match evidence supports it.

### Phase D: add recursive generation and control

1. Train the compact recursive policy to propose typed edits from residuals.
2. Add bounded stochastic trajectories and the Neuro scheduler.
3. Add task-local adaptation as an independently gated branch.
4. Run all required ablations; remove components that do not earn their cost.

### Phase E: contender freeze and public release

1. Freeze split, data, license, contamination, and provenance registries.
2. Select a configuration by untouched exact pass@2, not training fit.
3. Build and hash the hermetic bundle; run two clean local simulations.
4. Complete a Kaggle commit with internet off on the intended accelerator.
5. Publish code, method, weights/data artifacts, reproducibility instructions,
   honest limitations, and exact evidence required by the competition.

## 12. Definition of serious

The project demonstrates seriousness when a third party can reproduce the
reported result from the public repository, every candidate has traceable
provenance, no required channel can disappear silently, the architecture has
component ablations rather than only a compelling story, and the public claims
match exact evidence. Novelty is valuable only after those conditions hold.

## 13. Current implementation compatibility audit

**Audit updated:** 2026-09-09  
**Result:** Phase A deployment contract implemented; capability remains 0/120

| Area | Current evidence | Compatibility decision |
|---|---|---|
| Test health | `206 passed` locally after Phase A | retain as regression floor; tests do not establish ARC solving ability |
| Owned public-evaluation result | experience memory `0/120`; recursive world model `0/120`; recursive specialist adds `0` unique solves | none of these components is promoted as effective yet |
| Notebook staging | one SHA-addressed bundle with internal file manifest | Phase A complete; Kaggle commit verification remains |
| Dependency isolation | no runtime installation or global environment mutation | Phase A complete |
| Hyperbolic runtime | both legacy and contender memory use dependency-free Poincare math | Phase A complete; geometry efficacy remains unproven |
| Recursive model | a small specialist exists locally but is absent from the current solver bundle; local tests emit TorchScript deprecation warnings | replace serialization/runtime assumptions and package a model only after exact-score lift |
| World-model packaging | owned world-model code and frozen memory are present in `solver_code.zip` | structurally usable, but current evidence is `0/120`; keep disabled from contender ranking until improved |
| Second attempt | unsupported identity injection removed; a single candidate is honestly duplicated | Phase A complete; Phase B must produce useful diversity |
| Failure behavior | required assets fail before inference; task fallback is declared and counted | Phase A complete |
| Bundle integrity | content-addressed archive, internal hashes, channel sentinels, and release status | Phase A complete; SBOM and final licenses remain release blockers |
| External symbolic path | excluded from the owned solver bundle | correct ownership boundary; do not restore it under Hyper-ARC branding |

The immediate engineering sequence is therefore unambiguous: fix packaging and
failure semantics first; establish a trustworthy baseline report second; then
implement and ablate the shared state, procedural memory, and recursive policy.
Adding another inference channel before those two steps would increase apparent
complexity without increasing demonstrated capability.

## 14. Constraint and research basis

Authoritative operational constraints:

- [ARC Prize 2026 competition page](https://arcprize.org/competitions/2026)
- [Kaggle ARC Prize 2026 code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)
- [Kaggle notebook environment documentation](https://www.kaggle.com/docs/notebooks)
- [Official Kaggle Python image](https://github.com/kaggle/docker-python)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)

Research inputs, treated as hypotheses to reproduce rather than leaderboard
proof:

- [GRAM: stochastic recursive trajectories](https://openreview.net/pdf?id=Vxu6kcIjwV)
- [CausalARC: observational, interventional, and counterfactual reasoning](https://arxiv.org/abs/2509.03636)
- [Test-time adaptation of Tiny Recursive Models](https://arxiv.org/abs/2511.02886)

No reported paper score is counted as Hyper-ARC evidence. Only results produced
by this repository under the declared split and runtime enter its claims.

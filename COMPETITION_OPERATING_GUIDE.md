# ARC Prize 2026 Kaggle Operating Guide

Status checked: 2026-08-30

This guide is the shared operating procedure for `arc-2` and `arc-3`. The two
competitions share an offline Kaggle execution model, but they are not the same
problem:

- ARC-AGI-2 is static induction from input/output demonstrations. A submission
  supplies two complete output-grid attempts for each test input.
- ARC-AGI-3 is an interactive control problem. An agent must explore unfamiliar
  environments, infer goals and dynamics, and act efficiently.

## Current deadlines and hard constraints

| Item | ARC-AGI-2 | ARC-AGI-3 |
|---|---:|---:|
| Final submission | 2026-11-02 | 2026-11-02 |
| Earlier milestone | None | Milestone 2: 2026-09-30 |
| Notebook runtime | 12 hours | 9 hours |
| Internet during scoring | No | No |
| External models/data | Publicly available and rule-compliant | Publicly available and rule-compliant |
| Evaluation hardware | 4 x NVIDIA L4, 96 GB total VRAM | RTX 6000, 48 GB VRAM |
| Prize requirement | Open-source winning solution | Open-source winning solution |

Rule implications:

1. API-only designs are ineligible as production submissions. APIs may help
   research outside the scoring notebook, but the final solver must carry its
   code, weights, dependencies, and assets into Kaggle.
2. A model fitting in aggregate VRAM is not enough. Leave room for KV cache,
   activations, the executor, data, and framework overhead.
3. The notebook must run cleanly from top to bottom with internet disabled.
4. Keep private team code inside the registered Kaggle team. Rule-compliant
   public sharing belongs in Kaggle's discussion/code areas or an explicitly
   licensed public repository.
5. The private leaderboard determines the result. Public-leaderboard tuning is
   a weak and potentially misleading evaluation strategy.

Before relying on this summary, recheck the official pages if Kaggle posts a
rule update:

- [ARC Prize 2026 competition overview](https://arcprize.org/competitions/2026)
- [ARC-AGI-2 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
- [ARC-AGI-2 rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)
- [ARC-AGI-3 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [Kaggle competition documentation](https://www.kaggle.com/docs/competitions)
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)

## First-competition workflow

### 1. Join before building

For each competition:

1. Open the competition page.
2. Select **Join Competition** and accept the rules.
3. Confirm the account can open the **Data**, **Code**, and **Submit** tabs.
4. Remain a one-person team initially. Kaggle automatically creates that team.
   Do not merge teams until there is an actual collaborator and the rules have
   been checked.

### 2. Make a plumbing submission immediately

The first submission should prove the delivery path, not intelligence.

- ARC-AGI-2: start a competition notebook, attach the competition data, write a
  structurally valid `/kaggle/working/submission.json`, choose **Save Version**
  and **Save & Run All**, then submit the completed notebook version.
- ARC-AGI-3: copy the official starter, implement the required agent interface,
  run it locally with competition mode, then submit a clean Kaggle notebook
  version. The competition runtime produces the interaction submission.

This exposes account, dataset, package, path, runtime, and output-format errors
before they are entangled with model work.

### 3. Develop locally, package deliberately

Use the local project as the source of truth. Kaggle is the reproduction and
scoring environment.

1. Pin packages and model revisions.
2. Store public model weights as an attached Kaggle Model or Dataset; never
   download them during the scoring run.
3. Vendor small dependencies when Kaggle's base image may not contain them.
4. Use only `/kaggle/input/...` for read-only inputs and `/kaggle/working/...`
   for generated artifacts.
5. Set deterministic seeds where possible. Log version IDs, elapsed time,
   memory peaks, solved task/game counts, and fallback counts.
6. Test once with networking disabled and from a fresh process before upload.

Kaggle CLI is optional. It is useful later for versioned uploads, but the web UI
is simpler for the first valid submission. If used, authenticate with Kaggle's
current login flow and never commit credentials or display token contents.

### 4. Treat every saved version as an experiment

Maintain one experiment ledger per project with:

- commit or archive hash;
- notebook version and attached model/dataset versions;
- hypothesis being tested;
- exact local validation score and uncertainty;
- runtime and peak CPU/GPU/RAM usage;
- public leaderboard score, recorded but not optimized in isolation;
- failures and the next decision.

Change one major variable at a time. Run ablations before adding a component to
the ensemble. Complexity that does not improve exact held-out performance or
runtime reliability does not ship.

### 5. Final-submission discipline

Kaggle permits a limited daily submission cadence and selection of up to two
final submissions for ARC-AGI-2. Preserve diversity between the final choices:

- one stable, lower-variance submission;
- one validated higher-upside ensemble.

At least one week before the deadline:

1. Freeze dependency and weight versions.
2. Run the full notebook twice from a clean state.
3. Confirm it stays below 80% of the time limit on representative workload.
4. Inspect outputs for every required task/game; fail closed rather than emit a
   malformed submission.
5. Write the public-source and license inventory needed for release.

## Evaluation principles shared by both projects

- Exact outcome is the primary signal. Partial metrics diagnose; they do not
  establish competition performance.
- Separate development, validation, and final-selection data. Do not let a
  reusable memory store ingest held-out answers.
- Prefer executable hypotheses over unconstrained prose. Models propose;
  deterministic code verifies and controls.
- Record failures as structured traces. A failed hypothesis is useful memory
  only if its inputs, assumptions, result, and rejection reason remain linked.
- PHCG is the reusable skill/experience index, not the scorer and not the
  authoritative copy of competition data. Logical records and provenance are
  canonical; hyperbolic coordinates and color are rebuildable retrieval views.

## Official implementation resources

- [ARC-AGI-2 dataset repository](https://github.com/arcprize/ARC-AGI-2)
- [ARC toolkit](https://github.com/arcprize/ARC-AGI)
- [ARC benchmarking tools](https://github.com/arcprize/arc-agi-benchmarking)
- [ARC-AGI-3 agents](https://github.com/arcprize/ARC-AGI-3-Agents)
- [ARC-AGI-3 Kaggle starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [ARC-AGI-3 agent quickstart](https://docs.arcprize.org/agents-quickstart)
- [ARC-AGI-3 agent interface](https://docs.arcprize.org/create-agent)
- [ARC toolkit overview](https://docs.arcprize.org/toolkit/overview)


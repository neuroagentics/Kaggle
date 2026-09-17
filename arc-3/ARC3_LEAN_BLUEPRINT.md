# ARC-3 lean internal-world blueprint

Draft for implementation and independent agent handoff, 2026-09-10.
This document controls the competition implementation, alongside the preserved
reasoning requirements in ARC3_INTERNAL_WORLD_ARCHITECTURE.md. It supersedes that
document's PHCG integration and fixes the initial implementation choices below.
It is a target specification, not proof of a trained or competitive system.

## 1. Non-negotiable objective

Build an AI that acquires an unfamiliar game's mechanics from observations,
constructs a playable internal world, recursively explores possible futures,
acts through the official interface, and uses discrepancies and successes to
improve subsequent decisions within the same game. Preserve the user's reasoning
design. Do not replace it with graph traversal, action frequencies, a chat loop,
or a model that is loaded but never influences actions.

ARC-3 only: sequential interaction, level completion and action efficiency.
ARC-2 output grids, exact-match submission schemas, and its solver stay separate.
An imagined success is never recorded as a real completion.

## 2. Competition boundary and originality

Use ordinary game-local records instead of PHCG/PCHG. Do not import proprietary
geometry, chromatic routing, collective learning, tenant federation, customer
experience, personal conversations, or private procedural archives. No proprietary
service, key, or undisclosed component may be necessary to reproduce this entry.
This limits disclosure; it is not a guarantee of legal IP protection. The public
entry necessarily discloses its own functioning method if prize rules require it.

Implement our reasoning controller, evidence semantics, integration, and tests
from this specification. Research may inform algorithms and provide comparisons;
do not import or rebrand another entrant's solver, prompts, trained competition
solution, or game-specific policies. General pretrained models and ordinary ML
libraries are allowed candidates subject to licenses and competition rules.
Retain attribution for the existing official starter in UPSTREAM.md. Originality
means honest implementation provenance, not a claim to have invented recurrence,
beam search, or experience replay. Maintain a dependency/source/license ledger.

Current constraints: evaluation is offline; the competition overview retrieved
during this discussion lists a nine-hour CPU/GPU notebook limit and permits freely
publicly available pretrained models. Recheck these against the full rules before
release. Competition mode uses the official API, one instance per environment,
one scorecard, level-only resets, and no in-flight scorecard query. Never inspect
hidden engine internals or use official environment clones for imagined play.

Rules evidence:
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules
- https://docs.arcprize.org/toolkit/competition_mode

The complete Rules page has NOT been retrieved. Release remains blocked pending
review of training-data permissions, online adaptation, inter-game transfer,
model redistribution, prize disclosure and applicable compute limits. Do not
substitute ARC-2 rules. Default to fresh mutable memory/adapter per hidden game;
carry across that game's levels with applicability checks, not between games.

## 3. Architecture and ownership

```text
Permitted training trajectories -> train/test -> versioned offline model bundle
                                                     |
Real frame/status -> encoder + recurrent posterior <--+-- local evidence memory
                              |
                  current internal game state
                              |
         reasoner proposes goals/mechanics/subgoals (bounded, event-triggered)
                              |
         learned dynamics -> decoded predicted world -> repeat on branches
                              |
              consequence checks + bounded planner
                              |
                    selected first action
                              |
                validated official action adapter
                              |
        actual outcome -> prediction error -> memory + adaptation -> next plan
```

Two learned roles, one deterministic controller, no multi-agent messaging service:
- Deliberator: initially evaluate a license-compatible 4-8B open-weight reasoner
  at four-bit inference. Select one checkpoint after the compatibility gate;
  do not silently change model or quantizer. It proposes bounded structured
  mechanic/goal hypotheses and diagnoses failed predictions. It does not certify
  truth, write executable code into the runtime, or send game actions directly.
- Simulator: compact action-conditioned recurrent neural model with shared
  encoder, dynamics, decoder, outcome and uncertainty heads. Initial build target
  5-20M parameters; this is a budget, not an established capacity requirement.
  It performs most frequent numerical work, including recursive imagined steps.
- Controller: legal-action checks, branch budgets, evidence validation, deadlines,
  official adapter, and packaging. Button dispatch does not require a third model.

All roles share schema_version, game_id, level_id, observation_id, model_version,
action vocabulary and hypothesis IDs. Hypotheses cite evidence record IDs.
Reject malformed outputs and unsupported evidence references. One bounded repair
is allowed; then retain the last valid hypotheses and use learned-model planning
or uncertainty-directed probing. Never silently fall back to the old explorer
and describe the result as this architecture.

## 4. Concrete initial simulator specification

Preserve every observed categorical frame (up to 64x64, values 0-15) and animation
ordering. Pad with a validity mask; never treat padding as background. Encode
color categories into 16-dimensional embeddings, then a small convolutional
encoder (32/64/128 channels, stride-two stages) to a 256-dimensional feature.
Retain spatial features for decoding. Object features are derived observations
with confidence, not replacements for the original grid.

Maintain recurrent hidden state h of width 256 and stochastic latent z of width
64 (diagonal Gaussian initial implementation). Posterior conditions on new
observations; prior conditions on previous h, z and action. Action encoding uses
ID plus normalized coordinates and a coordinate-valid flag. Distinguish RESET
and unknown effects; never invent coordinate meaning. Decode from predicted
latent state to categorical next-frame logits, shape/validity, progress/failure
and candidate legal-action probabilities. Actual API availability always wins.

The simulator must advance from its OWN predicted state without reading a new
real frame. Fork complete recurrent/latent state; freeze weights and hypotheses
during a plan. A renderable decoded grid is mandatory. Preserve uncertain object
correspondences rather than falsely asserting identity from a single frame.

Train on permitted observation/action/outcome sequences, including no-ops,
failures, resets and successful transitions. Split by whole game/mechanic family
before fitting any model or tuning hyperparameters. Record provenance, licenses,
split hashes and action mappings. Independently generated curricula must also be
held out by mechanic combinations; synthetic success alone is not ARC evidence.

Initial loss: masked categorical frame loss with changed/unchanged regions
reported separately, multi-step prediction loss (1/2/4 steps), posterior/prior KL
regularization, and supervised outcome/availability losses where labels are
actually observed. Tune weights only on development games and publish them.
No fabricated goal labels or use of hidden state. Calibrate prediction error
risk on held-out development trajectories; verbal confidence is not calibration.

Measure 1/2/4/8-step error and object consequences against persistence and simple
motion baselines. Sparse motion can make an identity predictor look excellent;
overall pixel accuracy alone is explicitly inadequate. Architecture dimensions
may change on evidence without changing the user's reasoning requirements.

## 5. Lean memory: bounded game-local working memory

Use Python records, compact arrays and optional local JSONL diagnostics. No graph
database, vector database, embeddings service or geometry library. Retrieval is
exact contextual filtering plus a deterministic relevance score over mechanic,
object-relation and goal tags, with outcome evidence and recency tie-breaks.

| Record | Contents | Initial cap |
|---|---|---|
| Episode | Real before/action/predicted/actual IDs, discrepancy, status, version | 512 transitions |
| Mechanic belief | Hypothesis, context, supporting and contradicting IDs, revision | 128 |
| Goal hypothesis | Observable target, supporting evidence, disconfirming test | 16 |
| Procedure | Preconditions, object-relative parameters, steps, milestones, exceptions | 64 |
| Imagined branch | Predicted states and risk, explicitly unverified | Current plan only |

Memory ceiling: 32 MiB initial target, enforce bytes as well as record counts.
Store grids as uint8 with shape metadata; avoid Python integer-per-cell storage
and repeated tensors. Evict unreferenced low-value episodes first. If evidence
must be evicted, invalidate unsupported beliefs/procedures or retain a bounded
evidence summary with provenance; never leave a falsely verified dangling record.
These limits concern working memory, not model activations or training buffers.

Record real feedback before the next decision. Version beliefs and procedures;
contradictions supersede rather than erase failures. One success creates a
provisional contextual procedure; a second independent applicable success permits
promotion. A failed precondition or milestone stops execution and triggers revision.
Chaining requires compatible postconditions/preconditions, not similar labels.
Retain level context; do not assume controls or mechanics persist unchanged.

Online learning has two explicit mechanisms: posterior/belief updates immediately,
and bounded adapter updates from real replay. Freeze the base model; propose at
most one small adapter update per decision, after enough real samples exist.
Validate on a disjoint recent real replay slice; roll back non-finite or degraded
updates. Version adopted adapters; no gradient changes during imagined branches.
Final training legality must pass the rules gate. If restricted, report that
constraint and redesign within rules with user visibility, not silently remove it.

## 6. Reasoning, stopping and resource budget

Call the deliberator on game initialization, meaningful new level/context,
repeated model mismatch, or stalled progress. Initial cap: 1024 new tokens per
call, 8192 input tokens, maximum one call per eight real actions except initial
orientation. These are benchmarkable defaults, not promises of adequacy.
Feed structured observations and evidence, not the entire chat/event history.

Planner defaults: horizon 4, beam 8, at most 128 model transitions per decision;
extend toward horizon 8 only when calibration and remaining time justify it.
Evaluate progress, goal uncertainty, failure risk, information gain and action
cost. Replan after each real action. Save the chosen prediction before execution.
Probe uncertainty when confident planning is unavailable; random button cycling
is not the intended reasoning policy.

Set overall run deadline from verified rules, reserve >=10% for startup/recovery/
finalization, allocate the remainder dynamically across games. Bound both model
calls and worker shutdown; checking time only BETWEEN unbounded calls is not enough.
Limit deliberation initially to 20% of measured inference time and adaptation to
10%; measure the actual tradeoff and change caps with recorded justification.

Estimates, not measurements: 4-8B at four bits is 2-4 GB raw parameter storage,
excluding scales, KV cache, activations and backend overhead. A 5-20M simulator
at FP32 is 20-80 MB raw weights; gradients/optimizer/activations add substantially.
Initial integration target is one 16-24 GB GPU with restricted contexts/batches;
admission requires measured peak usage <=80% of assigned VRAM and a full bounded
run. CPU is for tests, not an assumed competitive inference tier. No paid API
dependency. Track training GPU-hours separately from evaluation cost and Kaggle
quota; free access is not a guaranteed unlimited resource.

## 7. Dependencies and offline deployment

Use Python 3.12 and a single tested PyTorch/CUDA backend for both learned roles
where practical; one selected quantization implementation, not a collection of
optional engines. Pin exact versions only after testing the actual Kaggle image.
Do not invent a compatible version matrix from imports on a different machine.
Keep official ARC wheels isolated from global notebook libraries. The prior
arc-agi 0.9.8 / arcengine 0.9.3 / Pillow 12.2.0 tuple is a baseline to verify, not
a permanent requirement or proof of compatibility with the ML backend.

Dev and Kaggle dependency sets are separated and must never be mixed:
- `requirements-dev.txt` installs a CPU-only PyTorch (`torch==2.14.0+cpu`) so the
  simulator/training tests run on a developer machine without a GPU. It is a
  local convenience and MUST NOT be shipped to Kaggle.
- `requirements-kaggle.txt` is the runtime set. On Kaggle the notebook runs on a
  CUDA GPU using the base image's CUDA PyTorch; torch stays unpinned there until
  the G8 rehearsal confirms the exact Kaggle torch/CUDA tuple.
Shipping the `+cpu` wheel to Kaggle would silently run on CPU and blow the time
budget. `scripts/preflight.py` fails closed (`check_torch_build_matches_device`)
if a `+cpu` build is detected while a `cuda:*` device is requested, so the wrong
wheel cannot pass preflight unnoticed. At runtime, `agent.device.resolve_device`
selects CUDA when requested and available and otherwise degrades to CPU with an
explicit recorded warning — never a silent downgrade.

Bundle weights, tokenizer, configs, schemas, offline wheels, hashes, license
notices, training manifest and test evidence as versioned Kaggle inputs. Load
from explicit mounted paths with networking disabled; no runtime download,
secret-dependent service, or implicit remote-code execution. The notebook
entrypoint must consume the same source bundle tested locally and on Kaggle.

Preflight must instantiate BOTH real model roles, verify artifact hashes, run
nontrivial action-conditioned multi-step predictions, validate legal action
serialization and confirm the intended device. Missing weights or failed
sentinels block release. An unscored commit placeholder is not model validation.
During evaluation, bounded recoveries may use the last valid learned state;
irrecoverable model failure must be explicit, never relabeled as normal success.

## 8. Evidence and research use

No claim of higher score until a scored run establishes it. No claim of internal
learning from storing records alone. Report real neural calls, imagined transitions,
adapter updates accepted/rejected, prediction errors, memory hits actually used,
level completions, real actions, elapsed time and peak CPU/GPU memory.

Research supports components, not this entry's success:
- https://danijar.com/project/dreamerv3/ : learned dynamics and latent imagination.
- https://arxiv.org/abs/2605.05138 : observation-verified executable worlds; not
  proof of offline quantized latent-model performance and not code to import.

Source review and licensing are prerequisites; full competition compliance and
competitiveness remain unproven. Follow ARC3_LEAN_DELIVERY_PLAN.md for evidence gates.

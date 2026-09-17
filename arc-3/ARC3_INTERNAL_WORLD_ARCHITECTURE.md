# ARC-3 internal game world: controlling architecture

> Lean competition revision, 2026-09-10: ARC3_LEAN_BLUEPRINT.md and
> ARC3_LEAN_DELIVERY_PLAN.md now control implementation details. The reasoning
> requirements below remain binding; PHCG integration below is superseded by
> bounded game-local records with no proprietary collective dependency.

Updated 2026-09-10. This is the authoritative ARC-3 target contract. The old
ARC_AGI_3_2026_PLAN.md describes the baseline and historical delivery sequence;
where it allows the transition explorer to substitute for this design, this
document supersedes it. A design contract is not evidence of implementation.

## User requirements preserved

The user requested a preloaded world model, construction of the current game
inside a latent reasoning space, multiple internal simulated plays, reflection
over outcomes, and useful memory during the same run. The user explicitly rejects
substitution with action scripts or a transition ledger branded as a world model.

Acceptance requires all of the following:

1. A pretrained model brings transferable knowledge of objects, spatial relations,
   transformations, controls, consequences, and candidate goals.
2. Actual observations condition that prior into a model of this particular game.
3. The model can advance an imagined state under an action, then advance that
   resulting state again. It can branch alternatives and play toward an inferred goal.
4. Simulation precedes action selection. Its predictions are recorded before the
   real action and compared against the subsequent real observation.
5. Outcomes revise game-local beliefs and useful procedures immediately.
6. Previous success supplies contextual skills rather than raw button repetition.
7. Model, simulator, control harness, memory, tests, and offline package must work
   together. Successful imports alone do not meet this requirement.

The known action interface and the inferred mechanics are different inputs to this
same design. ACTION6 has known coordinate syntax; what a click causes is inferred.
An untested action effect remains uncertain. The simulated game becomes more
faithful through interaction; its hidden rules are not supplied by the competition.

## The two competitions

| Concern | ARC-2 | ARC-3: current focus |
|---|---|---|
| Input | Static demonstration input/output grids, then test inputs | Sequential frames, available actions, status and level metadata |
| Unknown | Transformation rule | State, mechanics, action effects, goal and useful strategy |
| Internal computation | Candidate programs/transformations checked against demonstrations | Action-conditioned game simulation and recursive planning |
| External behavior | Two output grids per test input | Valid actions through the official game interface |
| Feedback | Demonstration verification; no test answer oracle | Observed consequences after every real action |
| Score | Exact output matches | Completed levels and action efficiency relative to humans |
| Geometry | Typically up to 30x30, ten color categories | Up to 64x64, sixteen color categories |
| Memory | Task-scoped hypotheses and verified procedural priors | Game-local mechanics, episodes, goals and validated procedures |
| Shared code | General evidence/version/schema utilities only | Separate perception, dynamics, planner, training and release gates |

The ARC-2 240-task/259-output checks and attempt_1/attempt_2 schema do not belong
in ARC-3. ARC-3's previous 0.21 score is not ARC-2 evidence.

## Entire competition framework

![Target framework with implementation status](docs/diagrams/arc3_internal_world.png)

### Offline development and model preparation

Public permitted observations and independently generated mechanics curricula
feed an observation/action/outcome dataset. Split by complete game/mechanic family
before training; frame-level random splits are insufficient. Store source, license,
split, action mapping, observation version and hashes. Do not use private game
source, introspected engine state or hidden answers as model input.

Train a categorical-grid/object encoder, recurrent latent transition model and
decoders for next-frame changes, object relations, progress, failure and action
availability. Build procedural priors from validated experiences. Freeze the base
checkpoint and record its training evidence; attach weights and dependencies offline.

Public data is not a description of each hidden game. The checkpoint represents
general priors; the game-local posterior completes and revises them.

### Runtime: Memory, Reasoning, Learning, Adaptation

1. **Memory:** load base weights and permissible frozen skills. Start isolated
   game-local state and retrieve only contextually applicable procedures.
2. **Perception:** preserve the categorical grid and animation history; distinguish
   settled observations from intermediate frames. Track objects and possible hidden
   state. Fixed masking of border cells must not destroy the model's raw observations.
3. **Reasoning:** infer mechanics and goals; encode the observed world; recursively
   simulate multiple sequences in cloned latent worlds; evaluate likely success,
   information value, failure modes and computation/action costs.
4. **Validation and execution:** reject impossible or unsupported predictions;
   issue only the selected first action using the actual available-action set.
5. **Feedback and learning:** compare actual outcomes with prior predictions,
   attribute errors to objects/mechanics/actions, update uncertainty and evidence.
6. **Adaptation:** revise the game-local posterior and bounded adapter, reject stale
   plans and compile validated contextual procedures. Return to Memory immediately.

The external game is the source of truth. An imagined win is a prediction. The
official game must still be played and confirm completion. Internal simulation
uses compute but sends no environment actions; actual resets and probes have cost.

### Latent dynamics and playable internal state

Use a recurrent latent state h, stochastic/ensemble state z, and an object graph G:

```
observed posterior: (h_t, z_t, G_t) = encode(o_0:t, a_0:t-1, prior, memory)
imagined transition: (h_next, z_next) = dynamics(h_t, z_t, action, G_t)
predicted outputs:  (grid_next, G_next, progress, risk, legal_actions) = decode(h_next, z_next)
recursive step:     feed (h_next, z_next, G_next) into dynamics again
correction:         compare the recorded prediction with the next real observation
```

The decoder is required for this project: it lets us inspect the internal game,
check object and grid consequences, and measure model errors. A value-only latent
planner may be useful research but would not satisfy this requirement by itself.
The runtime must retain recurrent hidden state for rules that cannot be inferred
from a single frame. A grid hash is only a lookup identifier, not this state.

Support branch fork, step, reset-to-imagined-snapshot and render-prediction. Internal
reset never resets the real environment. Actions carry the same IDs/coordinate
contract at both boundaries. Known API syntax is deterministic; inferred effects
carry confidence, applicable context and falsification evidence.

### Recursive reasoning policy

Generate competing mechanic and goal hypotheses. Consider likely success paths
and plausible failure paths; where useful, trace a failed outcome back to the
required precondition and test alternatives. This operationalizes the user's
reflection and consequence-based reasoning without treating geometric distance
or a model's verbal confidence as proof.

Bounded beam/MPC search operates on imagined futures. A learned actor may propose
sequences, but predictions and critic checks determine which survive. Stop a branch
when uncertainty becomes excessive, the state is invalid, its budget expires or a
terminal state is predicted. Shorten the horizon after prediction failure. When no
reliable plan exists, choose an informative real probe and refine the model.

First implementation budget targets, not measured Kaggle guarantees: horizon 4
extensible to 8, beam 8, maximum 128 transition predictions per decision. Batch GPU
predictions where possible. All model calls must obey the outer deadline; checking
the clock only between unbounded inference calls is insufficient. Measure actual
peak memory and latency before selecting the Kaggle accelerator.

### Memory and PHCG

Observed episodes contain before state, action, predicted consequence, actual
consequence, model version, confidence change and outcome evidence. Procedures
contain preconditions, object-relative parameters, ordered actions, expected
milestones, termination conditions and known exceptions. Preserve failure evidence.

Imagined trajectories remain transient and explicitly tagged. They may train a
planning policy as predictions, but never become confirmed environmental facts.
Game-local learning is immediately available; durable collective promotion is a
separate validated process. The default release boundary isolates hidden games
from each other until the exact competition rules establish permitted transfer.

PHCG supplies typed, versioned memory and may index analogous skills. Hyperbolic
coordinates do not run the game, prove truth or substitute for neural latent states.
Keep customer data, the commercial collective and personal PHCG archives outside
the public competition package. Any method necessary to reproduce the entry must
be disclosed as required for prize eligibility; a private service cannot be essential.

## Research comparison and decisions

| Source | Alignment | Decision for this project |
|---|---|---|
| DreamerV3 | Recurrent world model and learning from imagined trajectories | Closest conceptual support; borrow the learning pattern, not an unvalidated claim that a released checkpoint already understands ARC games |
| TD-MPC2 | Latent trajectory optimization and receding-horizon control | Borrow bounded planning; its continuous-control/action assumptions need adaptation and a decoder is required here |
| Graph-based ARC-3 exploration | Observed state graph, salience and frontier search | Retain as evaluation baseline and exploration aid; insufficient as the user's internal game world |
| Current ARC-3 code | Settled frames, connected components, effect estimates, BFS over known edges | Useful baseline; missing trained generative dynamics, goal hypotheses, recursive imagined play and learned adaptation |
| Dual-model technical design | Planner/critic separation and outcome adjudication | Keep structured plans and independent checks; MRLA remains the top-level lifecycle |
| Latent thinker/executable-world blueprint | Internal dynamics, recursive hypotheses and validation | Preserve core intent; adapt 30x30/ten-color assumptions to ARC-3 and avoid a fixed 32+32-worker deployment |

Sources checked 2026-09-10:

- https://arcprize.org/competitions/2026/arc-agi-3
- https://arcprize.org/competitions/2026/arc-agi-2
- https://docs.arcprize.org/actions
- https://docs.arcprize.org/games
- https://docs.arcprize.org/methodology
- https://docs.arcprize.org/toolkit/competition_mode
- https://docs.arcprize.org/arc-prize-2026
- https://www.nature.com/articles/s41586-025-08744-2
- https://arxiv.org/abs/2310.16828
- https://arxiv.org/abs/2512.24156
- https://arxiv.org/abs/2603.24621

The detailed Kaggle rules page returned no readable content in the web fetch.
The official docs confirm API interaction, a single game instance/scorecard and
level-only resets in competition mode. Exact runtime caps, data-transfer and
training permissions still require the complete current Kaggle rules before a
release. No complete competition-compliance claim is made here.

## Current status and corrections

- Restored `agent/my_agent.py` to the source embedded in the existing notebook.
  Preserved the interrupted effect-statistics experiment under
  `experiments/rejected_effect_model_v3.py`; it is not packaged.
- Added `agent/internal_world.py`: immutable latent/grid states, valid controls,
  forkable internal worlds, bounded recursive planning through predicted states,
  uncertainty rejection and immediate observed-feedback memory.
- Its dynamics backend is an explicit interface. There are no pretrained ARC-3
  dynamics weights in the project, and the new controller is not wired into
  `MyAgent`. A missing model must never be replaced by a random or constant predictor.
- Added tests for multi-step imagined success, branch isolation, control validity,
  planning limits and rejection of imagined outcomes as observed evidence.
  Tests use a known corridor fixture, not an ARC game and not a neural model.
- The existing local runner uses NORMAL mode, so historical local scores cannot be
  described as verified competition-mode runs. Its output now identifies the mode.
- No source/weights/publication or competition submission is authorized by a
  passing fixture. The full release gates below remain required.

## Release evidence required

| Gate | Required proof | Status |
|---|---|---|
| Prior | Actual weights, model/data/license hashes and training report | Missing |
| Prediction | Changed-cell and object prediction on unseen games; report unchanged-cell baseline | Missing |
| Recursion | 1/2/4/8-step prediction error and internal paths to inferred goals | Controller fixture only |
| Controls | Available action and coordinate validation; inferred effects tested in context | Contract tests only |
| Calibration | Uncertainty tracks prediction failures; misleading high-confidence futures rejected | Threshold plumbing only |
| Same-game learning | Later decisions demonstrably benefit from earlier observed success/failure | Episode storage only |
| Procedures | State-conditioned reuse and multi-step chaining improve later levels | Missing |
| Integration | Actual neural calls and imagined steps occur in the official adapter | Missing |
| Performance | Whole-game holdout with memory/no-memory and imagination/no-imagination ablations | Missing |
| Packaging | Offline real-model startup and simulated-play sentinel on target hardware | Missing |
| Reliability | Failure/deadline injection, bounded inference, schema and dependency tests | Baseline/contract tests only |
| Competition | Complete current rules and actual scoring-mode run, then explicit submission | Not performed |

The trained predictor, goal learner and end-to-end integration are substantive
remaining work. This document and the controller must not be advertised as a
completed recursive neural world model or a higher-scoring solver.

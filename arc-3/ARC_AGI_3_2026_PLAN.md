# ARC-AGI-3 2026 Implementation Plan

> Current build plan: [ARC3_LEAN_DELIVERY_PLAN.md](ARC3_LEAN_DELIVERY_PLAN.md).
> Current implementation blueprint: [ARC3_LEAN_BLUEPRINT.md](ARC3_LEAN_BLUEPRINT.md).
> PHCG is excluded from the competition package; use bounded game-local memory.
> The remaining historical plan is retained for provenance, not release authority.

> Architecture correction, 2026-09-10: [ARC3_INTERNAL_WORLD_ARCHITECTURE.md](ARC3_INTERNAL_WORLD_ARCHITECTURE.md)
> now controls the target reasoning design. The required system learns a playable
> internal game world and recursively simulates action sequences before acting.
> The transition explorer below is a baseline, not that completed model. Historical
> local results used NORMAL mode and are not verified competition-mode scores.

Status: settled-frame v2 scored 0.21; output/runtime hardening validated locally

Verified baseline and local iteration as of 2026-09-06:

- Kaggle kernel `jthomaslockhart/arc-agi-3-world-model-v0` completed.
- Competition submission `56054211` completed with public score `0.21`.
- The project virtual environment passes all 15 pre-hardening tests.
- Transition and action-efficiency telemetry now records bounded JSON traces,
  frame-delta classes, no-op/death rates, planner activity, decision latency,
  level progress, and actions per level.
- A 25-game, 200-step public run completed 4 levels across 3 games with local
  score `0.10815967461807642`; 2,477 of 5,000 transitions were no-ops.
- A direction-constrained motion-learning experiment was rejected: it reduced
  no-ops on `g50t` but regressed the full public score to `0.0927238876441148`
  and lost the `sp80` completion. The submitted controller remains unchanged.
- Linear `NEU-216` is complete. `NEU-217` telemetry is implemented locally;
  it exposed that the controller was using the first frame of each action's
  animation rather than the settled final frame. Correcting that observation
  boundary reproducibly raised the 25-game local score from
  `0.10815967461807642` to `0.26524352271442553`, with 5 levels completed
  across `lf52`, `lp85`, `r11l`, and `vc33`. The result repeated exactly at
  both 200- and 400-action caps. The next architecture gate is the latent
  simulator and hypothesis-driven planner.

## Decision

Build an explicit stateful world-model agent around the official ARC-AGI-3
interface. The primary loop should be a structural frame parser, task-local
transition graph, information-seeking explorer, goal detector, and lightweight
planner. A large language model must not sit in the per-action hot path.

```text
frame history
  -> frame differencing and object tracking
  -> canonical state + task-local transition graph
  -> goal/progress hypotheses
  -> informative action or planned action sequence
  -> observed transition
  -> world-model update, loop/no-op pruning, and replanning
```

ARC-AGI-3 measures successful and efficient interaction, not explanation. The
agent therefore needs fast feedback control more than long prose reasoning.

## Competition target

- Complete unfamiliar interactive games with no written instructions.
- Maximize level completion and action efficiency relative to human play.
- Run in the official Kaggle environment within 9 hours, without internet.
- Implement the required `is_done(frames, latest_frame)` and
  `choose_action(frames, latest_frame)` behavior through the official starter.
- Use only public, rule-compliant dependencies, weights, and training assets.

Current environment facts:

- 25 public game files and 110 private games are described by the competition;
- frames are grids up to 64 x 64 with values 0-15;
- actions include reset, five simple actions, a coordinate action, and another
  game-defined action slot;
- states include `NOT_FINISHED`, `WIN`, and `GAME_OVER`;
- later levels are weighted more heavily, and inefficient extra actions reduce
  score quadratically relative to human action counts.

Official references:

- [ARC-AGI-3 competition](https://arcprize.org/competitions/2026/arc-agi-3)
- [Kaggle competition and data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [Official agent repository](https://github.com/arcprize/ARC-AGI-3-Agents)
- [Official Kaggle starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [Agent quickstart](https://docs.arcprize.org/agents-quickstart)
- [Agent interface](https://docs.arcprize.org/create-agent)
- [Preview competition findings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)

## Lean target architecture

### A. Official runtime adapter

Start from the official starter and keep competition-specific I/O isolated.
Create a small adapter that passes normalized observations to the agent core and
returns a valid action. The core must also support local execution through the
ARC toolkit's competition mode.

### B. Structural frame state

For every observation, record:

- full grid hash and cropped/translated canonical hashes;
- frame delta, changed cells, change bounding boxes, and color transitions;
- connected objects, masks, centroids, motion, appearance, disappearance;
- persistent controllable objects, obstacles, targets, counters, and UI-like
  regions inferred from repeated interaction;
- game status, level index, action count, and remaining local budget.

Do not convert the grid to an image unless the selected visual model requires
it. Direct categorical-grid tensors preserve exact values and are cheaper.

### C. Task-local transition graph

The graph is empirical, not imagined:

```text
(canonical state, action, coordinates)
  -> observed next state, delta class, status, novelty, progress estimate
```

Use it to:

- recognize no-ops and reversible loops;
- reuse successful action sequences within the current game;
- distinguish movement, selection, transformation, collision, and reset;
- identify controllable entities and action preconditions;
- replan when the same action has a different result in a new context.

Hidden-game experience remains task-local during scoring. General policies may
be frozen from permitted public data, but the evaluation run must not silently
turn hidden outcomes into a cross-game answer store.

### D. Exploration controller

Score candidate actions by:

```text
expected progress + information gain + novelty
- predicted no-op - loop risk - death/reset risk - action cost
```

Use a staged policy:

1. calibrate simple actions cheaply;
2. identify the controllable object or cursor;
3. test uncertain transition hypotheses;
4. exploit the best supported plan;
5. reset only when the expected recovery value exceeds its cost.

Coordinate actions need proposal points from object centroids, corners, changed
regions, and symmetry correspondences rather than exhaustive 64 x 64 clicking.

### E. Goal and progress inference

Detect progress through multiple weak signals:

- explicit status or level transition;
- newly persistent frame structure;
- stable changes to counters or target-like regions;
- object alignment, collection, completion, or path opening;
- recurrence of known milestone patterns.

Maintain competing goal hypotheses with evidence and falsification conditions.
Do not promote a visual change to progress merely because it is large.

### F. Planner and learned models

Use graph search, bounded model-predictive control, or shallow MCTS over the
learned transition graph. Learned components estimate:

- probability an action changes the frame;
- transition/delta class;
- progress value and terminal risk;
- likely controllable object and useful coordinate proposals.

The preview competition's strongest published pattern was not LLM-only play:
compact CNN/ResNet-style models combined with exploration or state-graph
pruning produced the leading scores. Frontier language models also scored very
poorly when used as general agents. This makes a hybrid controller the evidence-
based starting point.

### G. Memory and PHCG boundary

Use:

- episodic task-local memory for every observed state/action transition;
- frozen procedural memory for general interaction skills;
- PHCG after experiments for versioned mechanic schemas, successful policies,
  failure/loop patterns, and evidence-backed outcomes.

PHCG can retrieve analogous mechanics, but the live transition graph must
confirm them. Canonical records and provenance are authoritative; embeddings,
color domains, and hyperbolic placement are derived indexes.

## Model options

| Option | Role | Resource profile | Decision |
|---|---|---:|---|
| Small CNN or ResNet18 on categorical frame/delta channels | Change, action, value, and risk prediction | Low latency; easily fits 48 GB | Primary learned component |
| Compact recurrent/transformer transition model | Short history and latent mechanic state | Small to medium | Add after state-graph baseline |
| Qwen3.5-9B, Apache 2.0 | Sparse visual/structural hypothesis critic | Quantized single-GPU fit | Optional, invoked only on stalls/milestones |
| Qwen3-8B, Apache 2.0 | Text/JSON world-model hypothesis critic | Small quantized footprint | Cheaper alternative to multimodal model |
| gpt-oss-20b, Apache 2.0 | High-level planner/critic | Quantize; higher latency | ARC-2 priority, ARC-3 ablation only |
| 30B-35B or larger reasoners | General planner | Poor per-action latency | Do not place in initial ARC-3 loop |

Relevant model source:

- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [gpt-oss-20b](https://openai.com/index/gpt-oss-model-card/)

The first learned policy should be the small CNN/ResNet branch, not a language
model. It directly predicts interaction effects, can be trained on public
rollouts, runs cheaply for every action, and matches the strongest lessons from
the preview competition. An LLM/VLM is admitted only if a measured stall policy
adds completed levels after accounting for latency and action cost.

## Validation design

### Primary metrics

- games and levels completed;
- official action-efficiency score;
- completion per wall-clock minute;
- catastrophic reset/game-over rate.

### Diagnostic metrics

- unique states discovered per action;
- no-op and repeated-loop actions;
- controllable-object identification time;
- transition prediction accuracy;
- progress/milestone precision and recall;
- coordinate-action proposal hit rate;
- planner replan count and model-stall count;
- latency distribution by controller branch.

### Generalization protocol

- Hold out entire public games, not random frames from the same game.
- Add transformation-based variants only when they preserve mechanics.
- Report performance by unseen-game and unseen-level conditions.
- Keep learned weights and procedural memory frozen for held-out evaluation.
- Use the official local competition runtime for every release gate.

### Runtime gates

1. `choose_action` always returns a valid action before its deadline.
2. The controller terminates promptly after `WIN` or `GAME_OVER`.
3. No-op/loop pruning reduces actions without reducing completions.
4. Learned components add held-out completions or action efficiency.
5. Full simulation stays below 7.2 hours, reserving 20% margin against the
   9-hour Kaggle limit.

## Schedule

ARC-AGI-3 has the earlier milestone, so it cannot wait for ARC-2 to finish.
Share schemas and experiment tooling, but keep its runtime and solver separate.

| Dates | Deliverable | Exit criterion |
|---|---|---|
| Aug 30-Sep 3 | Join competition, clone official starter, plumbing submission | Valid Kaggle run and local competition-mode replay |
| Sep 4-Sep 10 | Frame differencer, canonical state, telemetry | Deterministic replay and state hashes |
| Sep 11-Sep 17 | Transition graph, no-op and loop pruning | Fewer actions without fewer public completions |
| Sep 18-Sep 23 | Information-gain exploration and goal hypotheses | Faster discovery on held-out public games |
| Sep 24-Sep 29 | CNN/ResNet change-value model and ensemble | Nonzero, reproducible milestone submission |
| Sep 30 | Milestone 2 submission | Clean final-version notebook selected |
| Oct 1-Oct 10 | Model-based planner and coordinate proposals | More held-out levels at equal/lower actions |
| Oct 11-Oct 20 | Procedural retrieval, failure mining, sparse critic ablation | Unique completed levels justify each component |
| Oct 21-Oct 26 | Runtime optimization and freeze | Full run below budget; versions pinned |
| Oct 27-Nov 2 | Reproducibility runs and final selection | Stable final submission selected |

## First implementation backlog

1. Join the competition and make a starter-based plumbing submission.
2. Vendor/pin the official agent runtime and confirm local competition mode.
3. Implement observation, action, transition, episode, and hypothesis schemas.
4. Add exact frame differencing, state hashes, and replay logs.
5. Add a random/calibration controller with no-op and loop suppression.
6. Build the task-local transition graph and information-gain action scoring.
7. Train the small change/value model on public rollouts with game-level splits.
8. Add goal hypotheses and shallow planning.
9. Test one sparse Qwen3.5-9B critic only after the fast loop is competitive.
10. Export generalized, evidence-linked experience to PHCG after evaluation.

## Explicit non-goals for version one

- no language model call on every action;
- no monolithic end-to-end agent without observable state and transition traces;
- no exhaustive coordinate clicking;
- no PHCG geometry in the control loop;
- no hidden-game cross-contamination;
- no learned component retained without held-out completion or efficiency gains.

# Step 3 — Testable hypotheses that improve predictions AND actions (2026-09-11)

Milestone: trustworthy internal game. The intended reasoning:

    propose competing explanations
      -> simulate their consequences
      -> choose a distinguishing action
      -> observe
      -> reject or strengthen explanations
      -> replan.

Success is NOT "changing the hypothesis text changes an output". Success is a
supported hypothesis improving PREDICTIONS and ACTION SELECTION, with observed
evidence — not text — deciding which hypotheses survive.

No submission, controlled-world validation only.

## What was built

`agent/hypothesis_loop.py`:
- `Hypothesis` — a competing explanation that makes a CONCRETE, falsifiable
  prediction: `predict(grid, action_id, x, y) -> expected next grid`. Carries an
  evidence-updated `confidence`, `alive` flag, and hit/miss counters.
- `HypothesisLoop`:
  - `distinguishing_action(grid, candidates)` — picks the action under which the
    alive hypotheses DISAGREE most (an action they all agree on teaches nothing).
  - `update(grid, action, observed_next)` — scores each alive hypothesis against
    the REAL outcome; confidence = smoothed hit-rate; contradicted hypotheses are
    rejected. Text/labels never enter the update — only observed agreement.
  - `best()` — the surviving hypothesis with the highest evidence-based confidence.

The loop is PROPOSER-AGNOSTIC: it only uses each hypothesis's concrete prediction
and the observed outcome. A rule-based proposer is used here to validate the logic;
the same interface is what a Qwen proposer fills (see "Qwen wiring" below).

## Scenario and result

`scripts/step3_hypothesis_loop.py`: a movement world where the action->direction
mapping is UNKNOWN and must be discovered. Four competing hypotheses each assert a
different mapping (exactly one is true). Discovery phase runs distinguishing probes;
then the discovered mapping is used to PLAN toward a goal cell, compared against a
no-hypothesis baseline (random legal moves — cannot aim).

Report: `artifacts/step3_hypothesis_report.json`.

| metric | 200 trials, 8×8, 8 probes | 150 trials, 12×12, 4 probes |
|---|---|---|
| correct_survivor_rate (prediction) | **1.00** | **1.00** |
| rejected_all_wrong_rate | 0.90 | 0.96 |
| hypothesis reached goal | **1.00** | **1.00** |
| baseline reached goal | 0.26 | 0.17 |
| hypothesis avg steps | **5.3** | **7.9** |
| baseline avg steps | 16.9 | 24.3 |

### Honest reading

- **Prediction:** the surviving hypothesis was the TRUE mapping in 100% of trials,
  chosen purely from observed outcomes. (rejected_all_wrong < 1.0 only means a wrong
  hypothesis occasionally stayed "alive" at lower confidence — `best()` still picked
  the true one, so survivor rate is 1.0.)
- **Action selection:** the discovered hypothesis lets the agent aim — 100% reach
  the goal in ~5–8 steps, versus a no-hypothesis baseline that reaches it far less
  often and takes 3× as many steps. The supported hypothesis measurably improves
  decisions, not just an output string.
- **Evidence, not text, decides:** a unit test (`test_text_does_not_decide_only_evidence`)
  confirms renaming a hypothesis does not change which survives — only the match to
  the observed outcome does.
- Robust across grid size and probe budget (two configs above), so it is not tuned
  to one setting.

## Qwen wiring — adapter DONE; live-loop integration deferred

`hypothesis_from_mechanic(mech)` in `agent/hypothesis_loop.py` converts a real
`MechanicHypothesis` (agent/model.py — exactly what the deliberator/Qwen emits) into
a testable `Hypothesis`:
- `movement` + `direction_vector` -> a predict that moves the `affected_colors`
  cells by that vector,
- `interaction`/`transformation` + `effect_tags` like `to_color:9` -> a predict that
  recolors the `affected_colors` cells,
- unsupported types -> an honest no-op predictor (predicts "no change"), which the
  loop rejects if the world actually changes.

Tests (`tests/test_hypothesis_loop.py`) construct real `MechanicHypothesis` objects
and confirm the adapter yields correct predictions. So the proposer boundary is
closed at the data level: whatever Qwen proposes becomes a concrete, falsifiable
prediction the evidence loop can distinguish and score.

What remains (deferred, distinct task): wiring this loop into the LIVE controller
decision path — calling the deliberator to propose, running probes against the live
environment, and feeding the surviving hypothesis into the planner. That touches the
live loop (controller.py/world_model_agent.py) and is best done alongside the
spatial-simulator integration (task #7 follow-up), so the live agent is changed once,
carefully, rather than piecemeal.

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/step3_hypothesis_loop.py --n-trials 200 --grid 8 `
    --max-probes 8 --max-steps 40 --out artifacts/step3_hypothesis_report.json
```

# Step 5 — Substantive release gate (2026-09-11)

Milestone: trustworthy internal game. The gate must require IMPROVED level
completion or action efficiency on UNTOUCHED dev games, across REPEATED SEEDS,
against MATCHED component-disabled controls — before any offline bundle, competition
rehearsal, or submission. Official interaction/reset restrictions stay enforced; the
full track-specific rules review is still outstanding.

No submission. The gate is read-only: it does not upload, submit, or grant approval.

## What the gate is

`scripts/release_gate.py` (extended this step) reads `release/readiness.json` and
BLOCKS unless EVERY required check is `PASS` with hash-bound evidence and a named
reviewer, and every bound release input's SHA256 still matches. Required checks:

- arc3_rules, licenses_disclosure, data_permissions — external human/legal review.
- runtime_offline — offline Kaggle compatibility proven on the real image.
- dynamics_weights, reasoner_weights — the actual attached model inputs.
- model_integration — the improvements wired into the LIVE agent.
- same_game_learning, heldout_evaluation — genuine learning + held-out metrics.
- **substantive_improvement (NEW, Step 5)** — improved level completion OR action
  efficiency on UNTOUCHED dev games, across REPEATED SEEDS, vs MATCHED
  component-disabled controls, produced by the LIVE agent.
- regression_tests — full suite green.
- competition_rehearsal — offline official-mode rehearsal with reset semantics.
- user_release_approval — explicit human go-ahead.

## Current status: BLOCKED (honest)

`release/readiness.json` is intentionally empty; the gate reports BLOCKED with every
check unresolved, including `substantive_improvement`. This is correct: none of the
criteria are met yet.

Why `substantive_improvement` cannot PASS today — the honest boundary:

- The milestone's wins (spatial simulator fixing movement, the testable hypothesis
  loop, memory-as-transfer) are all validated in CONTROLLED WORLDS with known ground
  truth. They are NOT wired into the live agent (`controller.py`/`world_model_agent.py`
  still use the vector `ArcSimulator` interface). Controlled-world results are the
  right proof of CAPABILITY, but they are explicitly NOT the release criterion.
- Therefore the substantive criterion — better completion/efficiency on real,
  untouched dev games from the live agent, over repeated seeds, vs matched controls —
  has no evidence yet, and the gate must (and does) block.

## What evidence would satisfy `substantive_improvement`

A hash-bound report showing, for a set of dev games the tuning never touched:
1. the LIVE agent (with the new spatial sim + hypothesis loop + procedure reuse),
2. run across multiple seeds,
3. against matched controls that disable each component independently,
4. with a real, non-trivial improvement in level completion OR action efficiency
   (and no regression on the preserved baseline),
5. reviewed and named by a human.

Producing that requires the deferred LIVE-AGENT INTEGRATION (shared across the
spatial sim, hypothesis loop, and procedure reuse) plus a fresh paired study. That
is the honest gate to the release, and it remains ahead of us.

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/release_gate.py
# -> {"status": "BLOCKED", "issues": [... "Unresolved: substantive_improvement" ...]}
```

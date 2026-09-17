# Step 4 — Memory as demonstrated transfer (2026-09-11)

Milestone: trustworthy internal game. The prior color-bias adapter was a useful
integration check but not the experiential reasoning intended. This step stores a
successful action sequence with its full context and shows REUSE in a DIFFERENT
applicable situation produces measurably better decisions.

No submission, controlled-world validation only.

## What was built

Uses the REAL `agent/memory.py WorkingMemory` and `ProcedureRecord` — no parallel
store. `scripts/step4_memory_transfer.py`:

- **Learn (situation A):** discover the action->direction mapping by probing a
  movement world, navigate to a goal to confirm success.
- **Store:** a `ProcedureRecord` with the full context the milestone requires —
  starting conditions (`preconditions`: avatar_present, target_reachable), the action
  sequence (`steps`, object-relative: move dr/dc — NO hardcoded coordinates),
  expected intermediate states (`milestones`: distance_to_target_decreases), failure
  conditions (`exceptions`: avatar_blocked, target_unreachable), and supporting
  observations (`supporting_ids`: a real stored `EpisodeRecord`).
- **Reuse (situation B):** a DIFFERENT start/goal in the same mechanic. Memory-ON
  reuses the stored mapping/procedure (preconditions match B, so it is applicable) and
  navigates directly. Memory-OFF has no stored knowledge and must REDISCOVER by
  probing first — and those probes cost real steps.

Comparison is matched: memory-ON and memory-OFF navigate the SAME B world, start, and
goal; averaged over seeds.

## Result

`artifacts/step4_memory_transfer.json`.

| metric | 200 trials, 10×10 | 150 trials, 14×14 |
|---|---|---|
| transfer_applicable_rate | 1.00 | 1.00 |
| memory-ON reached goal | 1.00 | 1.00 |
| memory-OFF reached goal | 1.00 | 1.00 |
| memory-ON avg TOTAL steps | **6.76** | **9.13** |
| memory-OFF avg TOTAL steps | 10.41 | 12.92 |

### Honest reading

- **Transfer is real and applicable:** the stored procedure's preconditions matched
  the NEW situation in 100% of trials — this is reuse in a different situation, not
  replay of the same one.
- **The benefit is efficiency, and it is honestly the avoided re-discovery cost.**
  Both conditions ultimately reach the goal (once the mapping is known, navigation is
  the same), so the win is NOT a success-rate gap — it is the ~3.6–3.8 fewer total
  steps memory-ON saves by not re-probing. That gap matches the ~4 probes needed to
  disambiguate the four hypotheses, and it holds across grid sizes, so it is the
  mechanism, not noise.
- **No overclaim:** memory here transfers a discovered rule/procedure; it does not
  claim generalization beyond situations where the preconditions actually match.

## Relationship to the live memory

The store/promote/query path used here is the same WorkingMemory the live controller
uses (`promote_procedure_diverse` requires a second DIVERSE-context success before a
procedure goes 'active'; `revoke_procedure_on_contradiction` retires it if a later
observation contradicts it). Wiring this transfer into the live decision loop — query
applicable procedures at decision time and prefer a matching stored plan over
re-discovery — is part of the deferred live-integration task (shared with the spatial
sim and the hypothesis loop, so the live loop changes once, carefully).

## Reproduce (from repo root)

```powershell
.venv\Scripts\python.exe scripts/step4_memory_transfer.py --n-trials 200 --grid 10 `
    --max-probes 8 --max-steps 60 --out artifacts/step4_memory_transfer.json
```

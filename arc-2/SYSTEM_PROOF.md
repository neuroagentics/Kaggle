> Historical document, superseded for ARC-2 deployment on 2026-09-20. See [README.md](README.md) and release/current/release.json. Retained results do not certify the current candidate.

# Hyper-ARC Phase C proof of operation

Date: 2026-09-09

## Verdict

Hyper-ARC is a real, runnable owned solver system. Its recursive symbolic world
model, exact-replay transformation channel, builder-only procedural memory,
submission orchestration, checkpointing, validation, and Kaggle bundle execute
end to end. It is not yet proven to be a competitive leaderboard solver: the
labeled public-evaluation partition remains 0/120 exact tasks.

## Machine-checkable evidence

- `artifacts/system_proof_v1.json` SHA-256:
  `3a9aa33797885649c89d6126f31855e6aede468e3f8a985b89f14f3ed5a13b5b`
- Full suite: 214 passed in 29.14 seconds.
- Final bundle-focused suite: 18 passed in 6.96 seconds.
- End-to-end proof: two of two fixtures solved exactly, valid submission schema,
  matching submission hash, no failures, and no timeouts.

## Exact solver evidence

All counts below are full-task exact pass@2, not cell accuracy.

| Partition | Tasks | Owned baseline | World model | Combined | Combined regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen validation | 160 | 10 | 12 | 14 | 0 |
| Frozen holdout | 200 | 8 | 10 | 12 | 0 |
| Labeled public evaluation | 120 | - | - | 0 | - |

The holdout has been inspected during development and is no longer an untouched
estimate. The Kaggle hidden score remains the final independent test.

## Procedural memory

`hyper_arc/procedural_memory_v1.json` is mined only from the frozen 640-task
builder fold. It contains 25 operation chains, 30 positive exact-test episodes,
and 6 demonstration-fit/test-failure boundary episodes. Each record preserves:

- an executable procedure name and ordered steps;
- a semantic family;
- applicability invariants;
- a structural episode fingerprint;
- exact success or transfer-failure evidence.

Memory only ranks hypotheses that already replay every current demonstration
exactly. It cannot force an unverified answer into the submission. On the final
validation and holdout gates, ranking caused zero regressions and retained every
correct deterministic candidate within pass@2.

## Neural specialist

The neural model exists and runs:

- class: `RecursiveGridSpecialist`;
- checkpoint: `artifacts/recursive_specialist_full_v2.pt`;
- checkpoint bytes: 550,401;
- checkpoint SHA-256:
  `b6746f37d885a92d6a33a8e764349cb3bcd51111d96cc89dd8b72560d665e9d5`;
- parameters: 135,750;
- strict state load: passed;
- deterministic inference repeat: passed.

It is intentionally not enabled in the submission solver because its clean
promotion evidence is zero eligible leave-one-out tasks, zero pass@2, and zero
unique exact contribution. Existence is proven; usefulness is not.

## Kaggle release

Final archive:

`artifacts/bundle_phase_c_final/hyper_arc_solver_bundle.44f6df8b48159aec4396fb016d0ec770e2b5873ee971f22e62e6777c63e328ec.zip`

- bytes: 347,698;
- SHA-256:
  `44f6df8b48159aec4396fb016d0ec770e2b5873ee971f22e62e6777c63e328ec`;
- ownership boundary: Hyper-ARC-only; external verified/NVARC adapter excluded;
- runtime network dependency: none;
- notebook enables the recursive world model and procedural memory.

The archive is ready to attach to the Kaggle notebook. Upload and competition
submission remain manual because this machine has no Kaggle API credential and
the available authenticated browser-control surface cannot operate the file
picker.

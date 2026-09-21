# ARC Prize 2026 Readiness Status — 2026-08-31

> Historical snapshot, not current readiness. For ARC-2 use
> [arc-2/README.md](arc-2/README.md) and its active release record. This note does
> not change ARC-3's independently maintained status.

## Decision

Both projects now produce rule-compliant offline artifacts and measurable
nonzero capability, but neither is competition-ready yet. The remaining two
months must be managed by score gates: keep only changes that add exact ARC-2
solves or ARC-3 completed levels/action efficiency on held-out evaluation.

## ARC-AGI-2

Current evidence:

- 126 tests pass.
- The new small exact-replay channel solves 20/1,000 ARC-2 training tasks but
  0/120 ARC-2 public-evaluation tasks.
- The pinned MIT verified-symbolic ensemble reproduces 2/120 public-evaluation
  tasks (1.67% task pass@2; 2/172 output grids) and fits 18/120 tasks.
- `solver_code.zip` is rebuilt offline with 321 verified rules, their source
  license, the local deterministic channel, and the existing MCTS fallback.
- External source is pinned to commit
  `e151937e34c8b34f953833a0dab75797fc737ba4`.

Assessment: the deterministic ensemble is now a credible scoring floor, not a
competitive ceiling. The next channel must generate novel typed programs or
executable Python and pass the same exact demonstration-replay gate.

Immediate gates:

1. Upload the rebuilt solver archive and make a fresh Kaggle submission.
2. Confirm at least the reproduced public-evaluation floor transfers to the
   competition leaderboard.
3. Add one offline program-proposer model and measure unique exact solves.
4. Add recursive residual repair only if it improves held-out pass@2.

## ARC-AGI-3

Current evidence:

- 7 tests pass.
- Stable hashing now masks engine UI bands.
- The agent adds component-snapped clicks, action/color/region effect learning,
  death/no-op suppression, a task-local transition graph, frontier replay, and
  basic learned movement targeting.
- On the full 25 public-game sweep at 400 actions/game, it completed four
  levels across three games: `vc33=2`, `sp80=1`, `lp85=1`.
- The aggregate local score was `0.10815967461807642`; the previous local
  baseline completed zero levels and scored `0.0` on the two cached games.
- A private, internet-disabled T4 Kaggle notebook has been rebuilt from the
  tested agent source.

Assessment: the agent has crossed the nonzero-capability gate, but broad
exploration still wastes most actions. Goal inference and obstacle-aware path
planning are now higher priority than expanding the action search space.

Immediate gates:

1. Upload and run the rebuilt private Kaggle notebook, then submit the result.
2. Preserve the 25-game sweep as the fixed regression benchmark.
3. Add shortest-path movement planning and replay of successful level traces.
4. Compare each change against completed levels and official action efficiency;
   remove any component without unique gains.

## Two-month operating rule

Run one stable submission and one experimental submission. Freeze the stable
branch whenever it improves. Do not add PHCG, visualization, multi-agent
orchestration, or large-model calls to either hot path until they demonstrate a
unique solve on the fixed evaluation gates.

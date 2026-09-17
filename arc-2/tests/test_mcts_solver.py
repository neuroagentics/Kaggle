"""Regression tests for task-wide MCTS scoring and memory guidance."""

from __future__ import annotations

from hyper_arc.esb import ESB
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer
from hyper_arc.mcts import MCTSEngine, MCTSNode


def make(grid: list[list[int]]) -> ESB:
    return ESB.from_grid(grid)


def engine(**kwargs) -> MCTSEngine:
    return MCTSEngine(
        GlobalMemoryBank(),
        LocalTaskBuffer(),
        global_prior_weight=0.0,
        **kwargs,
    )


def test_score_program_requires_every_training_pair_to_match():
    rotate = [("rotate", {"degrees": 90})]
    first = (make([[1, 0], [0, 0]]), make([[0, 1], [0, 0]]))
    second = (make([[2, 0], [0, 0]]), make([[2, 0], [0, 0]]))

    score, exact = engine().score_program(rotate, [first, second])

    assert score == 0.75
    assert exact is False


def test_memory_prior_rewards_matching_next_action_only():
    local = LocalTaskBuffer()
    local.add([("rotate", {"degrees": 90})])
    search = MCTSEngine(
        GlobalMemoryBank(),
        local,
        global_prior_weight=0.0,
    )
    root = MCTSNode(make([[1]]), None, None)

    matching = search._get_prior(root, ("rotate", {"degrees": 90}))
    unrelated = search._get_prior(root, ("color_remap", {"mapping": {1: 2}}))

    assert matching == 1.0
    assert unrelated == 0.0


def test_rollout_returns_the_program_that_earned_its_reward(monkeypatch):
    action = ("rotate", {"degrees": 90})
    monkeypatch.setattr("hyper_arc.mcts.enumerate_actions", lambda _state: [action])
    search = engine(max_rollout_depth=1, random_seed=0)
    source = make([[1, 0], [0, 0]])
    target = make([[0, 1], [0, 0]])
    root = MCTSNode(source, None, None)

    reward, program, solved = search._rollout(root, [(source, target)])

    assert solved is True
    assert reward == 1.0
    assert program == [action]
    assert search.apply_program(program, source).to_grid() == target.to_grid()


def test_identity_program_is_accepted_across_multiple_pairs():
    pairs = [
        (make([[1, 2]]), make([[1, 2]])),
        (make([[3], [4]]), make([[3], [4]])),
    ]

    assert engine(max_iterations=1).solve(pairs, timeout_sec=1.0) == []


def test_one_step_sweep_solves_all_training_pairs_deterministically():
    pairs = [
        (make([[1, 0], [0, 0]]), make([[0, 1], [0, 0]])),
        (make([[2, 0], [3, 0]]), make([[3, 2], [0, 0]])),
    ]

    program = engine(max_iterations=0, random_seed=99).solve(pairs, timeout_sec=1.0)

    assert program == [("rotate", {"degrees": 90})]

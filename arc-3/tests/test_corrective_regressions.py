"""Regression proofs for the September architecture review, not score evidence."""
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import pytest
import torch

from agent.controller import Controller, ControllerConfig, Observation
from agent.deliberator import DeliberatorConfig, StubDeliberator
from agent.internal_world import Action, Prediction, WorldState
from agent.memory import WorkingMemory
from agent.simulator import ArcSimulator
from scripts.evaluate import EnvStep, run_game


def obs(**kw):
    return replace(Observation("o", "g", 0, ((1, 0), (0, 0)), (3, 4), False, False, "v"), **kw)


class FixtureDynamics:
    def predict(self, state, action):
        return Prediction(replace(state, imagined=True), 0, 0.8, 0)


def controller(dynamics=None, deliberator=None):
    c = Controller(ControllerConfig(game_id="g", model_version="v", max_predictions=2),
                   dynamics or FixtureDynamics(), deliberator, WorkingMemory("g"))
    c.start_game(monotonic() + 10)
    return c


def test_unknown_prediction_does_not_break_reasoning():
    c = controller()
    d = c.decide(obs())
    c.correct(obs(observation_id="after"), replace(d, predicted_state=None))
    c.deliberator = StubDeliberator(DeliberatorConfig(model_name="stub", model_version="v"))
    c._reason(obs(), c._measure(obs()), is_initial=True, planning_deadline=monotonic()+5)
    assert c.memory.query_episodes(limit=1)[0].prediction_available is False


def test_probe_keeps_prediction_and_feedback_is_idempotent():
    c = controller()
    d = c.decide(obs())
    assert d.predicted_state is not None
    after = obs(observation_id="after", level_id=1)
    c.correct(after, d)
    c.correct(after, d)
    assert len(c.memory.query_episodes(limit=10)) == 1
    assert c.memory.query_episodes(limit=1)[0].actual_success
    assert c.stats()["total_corrections"] == 1


def test_corrected_state_drives_imagination_and_real_actions_advance_h():
    torch.manual_seed(1)
    model = ArcSimulator(feature_dim=16, latent_dim=8, action_dim=8).eval()
    c = controller(model)
    w = c._measure(obs())
    altered = replace(w, latent=w.latent[:16] + tuple(v + 10 for v in w.latent[16:]))
    assert model.predict(w, Action(4)).state.latent != model.predict(altered, Action(4)).state.latent
    d = c.decide(obs())
    before = c._sim_state.h.clone()
    c.correct(obs(observation_id="after"), d)
    assert not torch.equal(c._sim_state.h, before)
    assert c._sim_state.h.grad_fn is None


def test_untrained_hypotheses_are_explicitly_unsupported():
    model = ArcSimulator(feature_dim=16, latent_dim=8, action_dim=8)
    assert not model.supports_conditioning
    with pytest.raises(NotImplementedError, match="conditioning"):
        model.predict_conditioned(None, Action(4), SimpleNamespace(mechanic_hypotheses=(object(),)))


class SpyEnv:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    def reset(self):
        self.calls.append("reset")
        return EnvStep([[0]], 0, False, False, (6,))
    def step(self, aid, x, y):
        self.calls.append((aid, x, y))
        if self.fail:
            raise RuntimeError("injected transport failure")
        return EnvStep([[1]], 1, True, True, (6,))


class SpyAgent:
    def __init__(self):
        self.corrected = []
    def decide(self, observation):
        return SimpleNamespace(action=Action(6, 0, 0), predicted_state=SimpleNamespace(grid=((1,),)))
    def correct(self, observation, decision):
        self.corrected.append(observation)


def evaluate(agent, env, **kw):
    return run_game(agent, "g", "full", 0, max_actions=2, max_wall_seconds=3,
                    model_version="v", code_hash="x", data_hash="y",
                    env_factory=SimpleNamespace(make=lambda gid: env), **kw)


def test_transport_steps_once_and_delivers_terminal_feedback():
    a, e = SpyAgent(), SpyEnv()
    result = evaluate(a, e)
    assert e.calls == ["reset", (6, 0, 0)]
    assert result.total_actions == 1 and not result.crashed
    assert len(a.corrected) == 1 and a.corrected[0].actual_success


def test_transport_failure_counts_attempt_and_crash():
    result = evaluate(SpyAgent(), SpyEnv(fail=True))
    assert result.crashed and result.total_actions == 1
    assert "injected transport failure" in result.crash_detail


def test_real_environment_does_not_disguise_exception():
    from scripts.run_ablations import ArcEnvironment
    class Bad:
        def perform_action(self, action):
            raise RuntimeError("transport")
    e = ArcEnvironment(Bad)
    e._game = Bad()
    with pytest.raises(RuntimeError, match="transport"):
        e.step(1, None, None)


def test_click_only_actions_and_baseline_are_legal():
    from scripts.run_ablations import _PersistenceBaseline
    o = obs(available_action_ids=(6,))
    assert all(a.id == 6 and a.x is not None for a in o.to_actions())
    assert _PersistenceBaseline((1, 2)).decide(o).action.id == 6
    assert obs(available_action_ids=()).to_actions() == ()


def test_no_planning_means_zero_predictions():
    class Forbidden:
        def predict(self, *args):
            pytest.fail("no-planning ablation invoked dynamics")
    c = controller(Forbidden())
    c.planning_enabled = False
    assert c.decide(obs()).imagined_steps == 0


def test_initial_reasoner_gets_current_grid():
    class Capture(StubDeliberator):
        def call(self, inp, **kwargs):
            self.captured = inp
            return super().call(inp, **kwargs)
    d = Capture(DeliberatorConfig(model_name="stub", model_version="v"))
    c = controller(deliberator=d)
    c.decide(obs())
    assert d.captured.current_grid == obs().grid


def test_deadline_prevents_dispatch_after_slow_decision():
    import time
    a, e = SpyAgent(), SpyEnv()
    original = a.decide
    def slow(o):
        time.sleep(0.02)
        return original(o)
    a.decide = slow
    result = evaluate(a, e, started_at=monotonic()-2.99)
    assert result.timed_out and result.total_actions == 0


def test_framework_adapter_delivers_final_feedback_and_prediction():
    from scripts.run_ablations import FrameworkAgentAdapter
    from arcengine import GameAction
    class Agent:
        action_counter = 0
        _pending_decision = SimpleNamespace(predicted_state="prediction")
        def choose_action(self, frames, latest):
            return GameAction.ACTION1
        def observe_outcome(self, latest):
            self.seen = latest
    a = Agent()
    adapter = FrameworkAgentAdapter(a)
    d = adapter.decide(obs(available_action_ids=(1,)))
    assert d.predicted_state == "prediction"
    adapter.correct(obs(is_terminal=True, actual_success=True), d)
    assert a.action_counter == 1
    assert a.seen.state.name == "WIN"


def test_live_adapter_updates_only_selected_bias_and_preserves_rng():
    from agent.online_replay import episode_batch
    from training.adapter import AdapterManager
    from training.train_simulator import TrainingConfig
    memory = WorkingMemory("g")
    for i in range(16):
        memory.add_episode(level_id=0, observation_id=str(i), model_version="v",
                           action_id=4, action_x=None, action_y=None,
                           before_grid=((0, 0), (0, 0)), predicted_grid=((0, 0), (0, 0)),
                           observed_grid=((1, 1), (1, 1)), actual_success=False)
    model = ArcSimulator(feature_dim=16, latent_dim=8, action_dim=8).eval()
    mgr = AdapterManager(TrainingConfig(learning_rate=0.01), regression_tolerance=1.0,
                         parameter_names={"decoder.deconv1.bias"})
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = mgr.maybe_update(model, episode_batch(memory), "cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    assert result.accepted, result.reason
    changed = [k for k,v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed == ["decoder.deconv1.bias"]
    assert not mgr.maybe_update(model, episode_batch(memory), "cpu").accepted


def test_runtime_and_validation_rollout_match():
    from training.validate_simulator import _rollout_predict
    from agent.online_replay import episode_batch
    memory = WorkingMemory("g")
    for i in range(16):
        memory.add_episode(level_id=0, observation_id=str(i), model_version="v",
                           action_id=4, action_x=None, action_y=None,
                           before_grid=obs().grid, predicted_grid=obs().grid,
                           observed_grid=obs().grid, actual_success=False)
    model = ArcSimulator(feature_dim=16, latent_dim=8, action_dim=8).eval()
    c = controller(model)
    prediction = model.predict(c._measure(obs()), Action(4)).state.grid
    batch = episode_batch(memory)
    with torch.no_grad():
        predicted, *_ = _rollout_predict(model, batch, 1, "cpu")[0]
    assert predicted.tolist() == [list(r) for r in prediction]


def test_procedure_promotes_across_contexts_and_preserves_coordinates():
    from agent.internal_world import Plan
    c = controller()
    for level in (0, 1):
        d = c.decide(obs(level_id=level, observation_id=f"before-{level}", available_action_ids=(6,)))
        c.correct(obs(level_id=level+1, observation_id=f"after-{level}"), d)
    active = c.memory.query_procedures(status="active")
    assert len(active) == 1
    class Confident(FixtureDynamics):
        def predict(self, state, action):
            return replace(super().predict(state, action), uncertainty=0.1)
    c.dynamics = Confident()
    w = c._measure(obs(level_id=2, available_action_ids=(6,)))
    action, prediction, *_ = c._select_action(Plan((), (), 0), w)
    assert action.id == 6 and action.x is not None and prediction is not None
    assert c.stats()["procedure_reuses"] == 1


def test_deliberator_rejects_expired_generation():
    from agent.deliberator_transformers import TransformersDeliberator
    d = TransformersDeliberator(DeliberatorConfig(model_name="fixture", model_version="v"),
                                model_path="unused", device="cpu",
                                generate_fn=lambda p,n: "{}")
    with pytest.raises(TimeoutError):
        d._generate("test", 1, deadline=monotonic()-1)

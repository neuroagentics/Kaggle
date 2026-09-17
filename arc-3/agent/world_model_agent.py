"""WorldModelAgent — R13 integration of the internal-world loop into the agent.

Wires the full Measure→Reason→Look-ahead→Act loop into the official ARC-AGI-3
Agent interface:

    FrameData  ──► Observation ──► Controller.decide() ──► DecisionRecord
        ▲                                                       │
        └──────────── correct(next frame) ◄── GameAction ◄──────┘

Honesty contract (blueprint §3, §7):
  - The learned simulator drives action selection ONLY when trained weights are
    actually loaded. When they are absent (the current state: R10 has no trained
    checkpoint yet), the agent falls back to the proven baseline explorer
    (MyAgent) and records `world_model_active=False` in telemetry. The fallback
    is NEVER relabeled as the internal-world architecture succeeding.
  - The deliberator is a StubDeliberator until R11 supplies a real quantized
    backend; `deliberator_backend` in telemetry names what is actually running.
  - No imagined success is ever reported as a real completion.

This makes the whole agent runnable and testable NOW, and turns into the full
learned system by dropping a trained checkpoint into place (no rewiring).
"""
from __future__ import annotations

import logging
import os
from time import monotonic, perf_counter
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agent.my_agent import MyAgent, frame_grid, state_key
from agent.controller import Controller, ControllerConfig, Observation
from agent.memory import WorkingMemory
from agent.deliberator import DeliberatorConfig, StubDeliberator
from agent.device import resolve_device
from agent.internal_world import Action, Prediction, WorldState

logger = logging.getLogger(__name__)


# Environment variable naming the trained simulator checkpoint. When unset or
# the file is missing, the agent runs in explicit baseline-fallback mode.
SIMULATOR_CHECKPOINT_ENV = "ARC3_SIMULATOR_CHECKPOINT"

# Environment variable naming the deliberator (HF) checkpoint directory. When
# unset/missing or CUDA is unavailable, a StubDeliberator is used and the
# backend name in telemetry says so — never silently claimed as the real model.
DELIBERATOR_PATH_ENV = "ARC3_DELIBERATOR_PATH"

MODEL_VERSION = "world-model-v0"
_PROCESS_DEADLINE = monotonic() + 8 * 3600


class _UntrainedSimulatorError(RuntimeError):
    """Raised internally when no trained simulator is available."""


def _load_simulator(device: str):
    """Attempt to load a trained ArcSimulator checkpoint.

    Returns (model, descriptor) on success. Raises _UntrainedSimulatorError when
    torch is unavailable, no checkpoint path is configured, or the file is
    missing. NEVER returns a random-initialised model as if it were trained —
    an untrained model must not drive real actions (blueprint §7).
    """
    ckpt_path = os.getenv(SIMULATOR_CHECKPOINT_ENV)
    if not ckpt_path:
        raise _UntrainedSimulatorError(
            f"{SIMULATOR_CHECKPOINT_ENV} not set; no trained simulator to load"
        )
    from pathlib import Path
    path = Path(ckpt_path)
    if not path.is_file():
        raise _UntrainedSimulatorError(
            f"Simulator checkpoint not found at {ckpt_path!r}"
        )
    try:
        from agent.simulator import ArcSimulator
    except ImportError as exc:
        raise _UntrainedSimulatorError(f"PyTorch unavailable: {exc}") from exc

    model, descriptor = ArcSimulator.load_checkpoint(path)
    try:
        import torch  # noqa: PLC0415
        model.to(torch.device(device))
    except Exception as exc:
        raise _UntrainedSimulatorError(f"Could not move model to {device}: {exc}") from exc
    model.eval()
    return model, descriptor


class WorldModelAgent(MyAgent):
    """Internal-world agent with an explicit, honest baseline fallback.

    Subclasses MyAgent so it inherits:
      - all frame↔grid plumbing and telemetry,
      - the proven baseline `choose_action` (reached via `_baseline_choose_action`),
      - the same `is_done` (WIN) contract.

    It overrides `choose_action` to run the internal-world Controller when a
    trained simulator is loaded, and to fall back to the baseline otherwise.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Resolve device (auto → cuda when available, else cpu with logged reason)
        resolution = resolve_device(os.getenv("ARC3_DEVICE", "auto"))
        self._device = resolution.resolved
        if resolution.degraded:
            logger.warning(
                "WorldModelAgent device degraded: %s (using %s)",
                resolution.reason, resolution.resolved,
            )

        # Attempt to load a trained simulator. On any failure, fall back.
        self._world_model_active = False
        self._simulator = None
        self._simulator_descriptor = None
        self._fallback_reason = ""
        try:
            self._simulator, self._simulator_descriptor = _load_simulator(self._device)
            self._world_model_active = True
            logger.info(
                "WorldModelAgent: trained simulator loaded (%s) on %s",
                getattr(self._simulator_descriptor, "model_version", "?"),
                self._device,
            )
        except _UntrainedSimulatorError as exc:
            self._fallback_reason = str(exc)
            logger.warning(
                "WorldModelAgent: no trained simulator (%s). "
                "Running in explicit baseline-fallback mode.", exc,
            )

        # Build the controller only when the world model is active.
        self._controller: Controller | None = None
        self._deliberator_backend = "none"
        if self._world_model_active:
            self._controller = self._build_controller()
        if os.getenv("ARC3_REQUIRE_LEARNED") == "1":
            if not self._world_model_active or self._deliberator_backend == "stub":
                raise RuntimeError("Scored run requires both learned models; fallback is forbidden")
            if getattr(self._simulator, "checkpoint_transition_version", None) != self._simulator.transition_version:
                raise RuntimeError("Simulator checkpoint requires retraining for current transition semantics")

        # Track the last decision so correct() can be paired with the next frame.
        self._pending_decision = None
        self._last_observation: Observation | None = None
        self._wm_telemetry: dict[str, Any] = {
            "world_model_active": self._world_model_active,
            "deliberator_backend": self._deliberator_backend,
            "fallback_reason": self._fallback_reason,
            "controller_decisions": 0,
            "baseline_decisions": 0,
        }

    @property
    def name(self) -> str:
        mode = "world-model" if self._world_model_active else "baseline-fallback"
        return f"{super(MyAgent, self).name}.world-model-v0.{mode}.{self.MAX_ACTIONS}"

    def _build_controller(self) -> Controller:
        """Construct the Controller with memory, deliberator, and the simulator."""
        cfg = ControllerConfig(game_id=self.game_id, model_version=MODEL_VERSION)
        memory = WorkingMemory(self.game_id, schema_version="1.0")

        deliberator = self._build_deliberator()

        controller = Controller(cfg, self._simulator, deliberator, memory)
        # Begin with a tiny, reversible game-local adapter, not full-model SGD.
        if hasattr(self._simulator, "decoder"):
            from training.adapter import AdapterManager
            from training.train_simulator import TrainingConfig
            from agent.online_replay import episode_batch
            manager = AdapterManager(TrainingConfig(learning_rate=1e-3),
                                     regression_tolerance=1.0,
                                     parameter_names={"decoder.deconv1.bias"})
            def adapt(mgr):
                # At most once per eight real corrections; never train imagined data.
                if (controller._total_corrections + 1) % 8:
                    return
                batch = episode_batch(controller.memory)
                if batch is not None:
                    mgr.maybe_update(self._simulator, batch, self._device,
                                     deadline=min(controller._run_deadline, monotonic() + 1.0))
            controller.set_adapter(manager, adapt)
        # Set a generous per-run deadline; the framework enforces MAX_ACTIONS too.
        controller.start_game(_PROCESS_DEADLINE)
        return controller

    def _build_deliberator(self):
        """Return the real transformers deliberator when possible, else the stub.

        Uses TransformersDeliberator (Qwen2.5-7B 4-bit) only when its checkpoint
        directory exists AND the resolved device is CUDA. Otherwise falls back to
        StubDeliberator, and `self._deliberator_backend` names what actually runs
        (never claims the real backend when the stub is used).
        """
        delib_path = os.getenv(DELIBERATOR_PATH_ENV)
        want_real = bool(delib_path) and self._device.startswith("cuda")

        if want_real:
            try:
                from pathlib import Path
                if not Path(delib_path).is_dir():
                    raise FileNotFoundError(f"deliberator dir missing: {delib_path}")
                from agent.deliberator_transformers import TransformersDeliberator
                cfg = DeliberatorConfig(
                    model_name="qwen2.5-7b-instruct", model_version=MODEL_VERSION,
                )
                delib = TransformersDeliberator(
                    cfg, model_path=delib_path, device=self._device, load_now=True,
                )
                self._deliberator_backend = "transformers-qwen2.5-7b-4bit"
                logger.info("WorldModelAgent: real deliberator loaded from %s", delib_path)
                return delib
            except Exception as exc:
                logger.warning(
                    "WorldModelAgent: real deliberator unavailable (%s); "
                    "using stub deliberator.", exc,
                )

        cfg = DeliberatorConfig(
            model_name="stub-deliberator", model_version=MODEL_VERSION,
        )
        self._deliberator_backend = "stub"
        return StubDeliberator(cfg)

    # ------------------------------------------------------------------
    # FrameData → Observation
    # ------------------------------------------------------------------

    def _observation_from_frame(
        self, latest_frame: FrameData, *, observation_id: str
    ) -> Observation:
        grid = frame_grid(latest_frame)
        if not grid:
            grid = [[0]]
        available = [int(v) for v in latest_frame.available_actions]
        available = [v for v in available if v != GameAction.RESET.value]
        return Observation(
            observation_id=observation_id,
            game_id=self.game_id,
            level_id=int(latest_frame.levels_completed),
            grid=tuple(tuple(int(c) for c in row) for row in grid),
            available_action_ids=tuple(available),
            is_terminal=latest_frame.state in (GameState.WIN, GameState.GAME_OVER),
            actual_success=(latest_frame.state is GameState.WIN or (
                self._pending_decision is not None
                and int(latest_frame.levels_completed) > self._pending_decision.level_id)),
            model_version=MODEL_VERSION,
        )

    def _action_to_game_action(self, action: Action, why: str) -> GameAction:
        """Convert an internal-world Action to a framework GameAction.

        Reuses the baseline `_materialize` so ACTION6 coordinate data and the
        reasoning dict are attached exactly as the framework expects.
        """
        key = (action.id, action.x, action.y)
        return self._materialize(key, why)

    # ------------------------------------------------------------------
    # choose_action override
    # ------------------------------------------------------------------

    def observe_outcome(self, latest_frame: FrameData) -> None:
        """Flush pending feedback once, including the final frame with no next action."""
        if self._controller is None or self._pending_decision is None:
            return
        decision = self._pending_decision
        obs = self._observation_from_frame(
            latest_frame, observation_id=f"{decision.decision_id}-outcome"
        )
        self._pending_decision = None
        self._controller.correct(obs, decision)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # The official loop checks this BEFORE requesting its next action.
        self.observe_outcome(latest_frame)
        return super().is_done(frames, latest_frame)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # RESET handling: start/recover the level. BUT first flush any pending
        # correction so the transition INTO a terminal state (which carries the
        # success/failure signal) is still recorded as feedback — previously this
        # early return dropped that final, most-informative transition.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if (self._world_model_active and self._controller is not None
                    and self._pending_decision is not None):
                try:
                    term_obs = self._observation_from_frame(
                        latest_frame,
                        observation_id=f"{self.game_id}-{self.action_counter:06d}-term",
                    )
                    self._controller.correct(term_obs, self._pending_decision)
                except Exception as exc:
                    logger.debug("terminal correct skipped: %s", exc)
                finally:
                    self._pending_decision = None
            # Keep the baseline's bookkeeping in sync so a later fallback is clean.
            return self._baseline_choose_action(frames, latest_frame)

        if not self._world_model_active or self._controller is None:
            self._wm_telemetry["baseline_decisions"] += 1
            action = self._baseline_choose_action(frames, latest_frame)
            self._annotate_fallback(action)
            return action

        # --- World-model path ---
        try:
            return self._world_model_choose_action(frames, latest_frame)
        except Exception as exc:
            # Never crash the game loop: degrade to baseline for this step and
            # record it honestly. A repeated failure keeps using the baseline.
            logger.warning("World-model step failed (%s); using baseline this step", exc)
            self._wm_telemetry["baseline_decisions"] += 1
            action = self._baseline_choose_action(frames, latest_frame)
            self._annotate_fallback(action, runtime_error=str(exc))
            return action

    def _world_model_choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        started = perf_counter()
        assert self._controller is not None

        step_idx = self.action_counter
        obs = self._observation_from_frame(
            latest_frame, observation_id=f"{self.game_id}-{step_idx:06d}"
        )

        # Pair the previous decision with this freshly observed frame.
        if self._pending_decision is not None:
            self.observe_outcome(latest_frame)

        decision = self._controller.decide(obs)
        self._pending_decision = decision
        self._last_observation = obs

        why = (
            "world-model plan"
            if not decision.plan_was_empty
            else "world-model uncertainty probe"
        )
        action = self._action_to_game_action(decision.action, why)

        # Attach honest telemetry.
        self._wm_telemetry["controller_decisions"] += 1
        elapsed_ms = (perf_counter() - started) * 1000.0
        action.reasoning = {
            "controller": "internal-world-v0",
            "why": why,
            "world_model_active": True,
            "deliberator_backend": self._deliberator_backend,
            "plan_length": decision.plan_length,
            "plan_was_empty": decision.plan_was_empty,
            "plan_was_ambiguous": decision.plan_was_ambiguous,
            "imagined_steps": decision.imagined_steps,
            "decision_latency_ms": round(elapsed_ms, 3),
            "device": self._device,
        }
        return action

    def _baseline_choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Delegate to the proven baseline explorer (MyAgent.choose_action)."""
        return MyAgent.choose_action(self, frames, latest_frame)

    def _annotate_fallback(
        self, action: GameAction, *, runtime_error: str | None = None
    ) -> None:
        """Mark a baseline-produced action as an explicit fallback (never hidden)."""
        reasoning = getattr(action, "reasoning", None)
        if not isinstance(reasoning, dict):
            reasoning = {"why": str(reasoning) if reasoning else "baseline"}
        reasoning["world_model_active"] = False
        reasoning["fallback_reason"] = runtime_error or self._fallback_reason
        reasoning["deliberator_backend"] = self._deliberator_backend
        action.reasoning = reasoning

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def world_model_summary(self) -> dict[str, Any]:
        """Honest report of what actually ran this game."""
        summary = dict(self._wm_telemetry)
        if self._controller is not None:
            summary["controller"] = self._controller.stats()
        if self._simulator_descriptor is not None:
            summary["simulator"] = {
                "model_version": self._simulator_descriptor.model_version,
                "parameter_count": self._simulator_descriptor.parameter_count,
                "device": self._device,
            }
        return summary

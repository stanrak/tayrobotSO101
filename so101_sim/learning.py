"""Training-only reward scaling and diagnostics from already collected PPO data."""

from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch


class RewardScale(gym.RewardWrapper):
    """Scale learning rewards; place Monitor inside this wrapper to log raw returns."""

    def __init__(self, env: gym.Env, scale: float):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Reward scale must be finite and positive.")
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


class DiagnosticPPO(PPO):
    """Measure PPO work without collecting extra transitions or sampling actions."""

    diagnostic_sample_size = 256

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rollout_update_count = 0
        self.diagnostic_optimizer_steps = 0
        self.rollout_seconds = 0.0
        self.optimization_seconds = 0.0
        self.diagnostics_seconds = 0.0

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> bool:
        # Includes callbacks (and therefore validation/viewer time). The trainer
        # reports callback evaluation separately rather than treating this as
        # pure environment simulation time.
        started = time.perf_counter()
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self.rollout_seconds += time.perf_counter() - started
            self.logger.record("timing/rollout_including_callbacks_seconds", self.rollout_seconds)

    def train(self) -> None:
        optimizer = self.policy.optimizer
        original_step = optimizer.step
        optimizer_steps = 0
        first_gradient_norm = None

        def counted_step(*args: Any, **kwargs: Any):
            nonlocal optimizer_steps, first_gradient_norm
            if optimizer_steps == 0:
                # PPO has already clipped gradients immediately before this
                # call. Label this explicitly rather than claiming a raw norm.
                squared_norm = sum(
                    parameter.grad.detach().square().sum().item()
                    for parameter in self.policy.parameters() if parameter.grad is not None
                )
                first_gradient_norm = math.sqrt(squared_norm)
            result = original_step(*args, **kwargs)
            optimizer_steps += 1
            self.diagnostic_optimizer_steps += 1
            return result

        started = time.perf_counter()
        optimizer.step = counted_step
        try:
            super().train()
        except BaseException:
            # Partial phases retain their real optimizer count, but are not
            # described as completed rollout updates.
            raise
        else:
            if optimizer_steps:
                self.rollout_update_count += 1
        finally:
            optimizer.step = original_step
            self.optimization_seconds += time.perf_counter() - started
            self.logger.record("timing/optimization_seconds", self.optimization_seconds)
            self.logger.record("train/rollout_update_count", self.rollout_update_count)
            self.logger.record("train/actual_optimizer_steps", self.diagnostic_optimizer_steps)

        if first_gradient_norm is not None:
            self.logger.record("diagnostics/first_gradient_norm_after_clip", first_gradient_norm)
        started = time.perf_counter()
        try:
            self._record_rollout_diagnostics()
        finally:
            self.diagnostics_seconds += time.perf_counter() - started
            self.logger.record("timing/diagnostics_seconds", self.diagnostics_seconds)

    def _record_rollout_diagnostics(self) -> None:
        buffer = self.rollout_buffer

        def flattened(array: np.ndarray) -> np.ndarray:
            # PPO.get() flattens arrays in-place on its first minibatch. Do not
            # call get() here: it permutes data and advances the training RNG.
            return np.asarray(array) if buffer.generator_ready else buffer.swap_and_flatten(np.asarray(array))

        old_values = flattened(buffer.values).reshape(-1)
        targets = flattened(buffer.returns).reshape(-1)
        if not len(targets):
            return
        self.logger.record("diagnostics/rollout_value_mean", float(np.mean(old_values)))
        self.logger.record("diagnostics/rollout_value_std", float(np.std(old_values)))
        self.logger.record("diagnostics/target_mean", float(np.mean(targets)))
        self.logger.record("diagnostics/target_std", float(np.std(targets)))

        if isinstance(self.action_space, gym.spaces.Box):
            actions = flattened(buffer.actions).reshape(len(targets), -1)
            clipped = (actions < self.action_space.low.reshape(-1)) | (actions > self.action_space.high.reshape(-1))
            self.logger.record("diagnostics/action_component_clip_fraction", float(np.mean(clipped)))
            for joint in range(actions.shape[1]):
                self.logger.record(f"diagnostics/action_{joint}_clip_fraction", float(np.mean(clipped[:, joint])))

        indices = np.linspace(0, len(targets) - 1, min(len(targets), self.diagnostic_sample_size), dtype=int)
        observations = buffer.observations
        sample = ({key: flattened(value)[indices] for key, value in observations.items()}
                  if isinstance(observations, dict) else flattened(observations)[indices])
        hooks = []
        saturation: dict[str, float] = {}
        value_layers = getattr(getattr(self.policy, "mlp_extractor", None), "value_net", None)
        if value_layers is not None:
            for index, layer in enumerate(module for module in value_layers.modules() if isinstance(module, torch.nn.Tanh)):
                def record_saturation(module, arguments, result, index=index):
                    saturation[f"diagnostics/critic_tanh_{index + 1}_saturation"] = float(
                        (result.detach().abs() > 0.999).float().mean().item()
                    )
                hooks.append(layer.register_forward_hook(record_saturation))

        was_training = self.policy.training
        try:
            self.policy.set_training_mode(False)
            with torch.no_grad():
                tensor, _ = self.policy.obs_to_tensor(sample)
                predictions = self.policy.predict_values(tensor).flatten().cpu().numpy()
                self.logger.record("diagnostics/critic_value_mean", float(np.mean(predictions)))
                self.logger.record("diagnostics/critic_value_std", float(np.std(predictions)))
                self.logger.record("diagnostics/critic_target_rmse", float(np.sqrt(np.mean((predictions - targets[indices]) ** 2))))
                self.logger.record("diagnostics/sample_count", len(indices))
                # Constructing the distribution does not sample an action. Its
                # conditional std also works for gSDE's feature-by-action matrix.
                distribution = self.policy.get_distribution(tensor).distribution
                std = getattr(distribution, "stddev", None)
                if std is not None:
                    for joint, value in enumerate(std.reshape(len(indices), -1).mean(dim=0).cpu().tolist()):
                        self.logger.record(f"diagnostics/action_{joint}_std_mean", float(value))
            for name, value in saturation.items():
                self.logger.record(name, value)
        finally:
            for hook in hooks:
                hook.remove()
            self.policy.set_training_mode(was_training)

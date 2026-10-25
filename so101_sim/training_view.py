"""Observe actual training states without taking actions or editing training physics."""

import copy
import math
import time

import mujoco
from stable_baselines3.common.callbacks import BaseCallback


class TrainingViewer(BaseCallback):
    """Display environment zero using an independent model/data snapshot."""

    def __init__(self, *, speed=1.0, progress=lambda: {}):
        super().__init__()
        self.speed = speed
        self.progress = progress
        self.viewer = None
        self.episodes = 0
        self.last_status = 0.0

    def _on_training_start(self):
        import mujoco.viewer

        self.observed = self.training_env.envs[0].unwrapped
        self.view_model = copy.copy(self.observed.model)
        self.view_data = mujoco.MjData(self.view_model)
        self._copy_state()
        self.viewer = mujoco.viewer.launch_passive(
            self.view_model, self.view_data, show_left_ui=False, show_right_ui=False,
        )
        with self.viewer.lock():
            self.observed.configure_camera(self.viewer.cam)
        self.next_frame = time.monotonic()
        print("Live training: observing environment 0. Camera controls only; close the window to stop training.", flush=True)

    def _copy_state(self):
        self.view_model.site_pos[:] = self.observed.model.site_pos
        mujoco.mj_copyData(self.view_data, self.view_model, self.observed.data)

    def _on_step(self):
        if not self.viewer.is_running():
            raise KeyboardInterrupt("Training viewer closed.")
        delay = self.next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        with self.viewer.lock():
            self._copy_state()
        self.viewer.sync()
        self.next_frame = time.monotonic() + self.observed.dt / self.speed
        done = self.locals.get("dones", [False])[0]
        self.episodes += int(done)
        now = time.monotonic()
        if now - self.last_status >= 1.0:
            info = self.locals.get("infos", [{}])[0]
            distance = info.get("distance", float("nan")) * 1000
            updates = self.progress().get("optimizer_steps", 0)
            total = self.model._total_timesteps
            target = f"{total:,}" if math.isfinite(total) else "unlimited"
            print(f"Training steps: {self.num_timesteps:,}/{target} | "
                  f"env 0 episodes: {self.episodes} | distance: {distance:.1f} mm | "
                  f"network updates: {updates}", flush=True)
            self.last_status = now
        return True

    def _on_rollout_end(self):
        print(f"Rollout collected at {self.num_timesteps} steps; updating the network...", flush=True)

    def _on_training_end(self):
        print(f"Network updates completed: {self.progress().get('optimizer_steps', 0)}", flush=True)
        self.close()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

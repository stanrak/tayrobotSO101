"""Physics-based, local Cartesian reach-and-hold task for the SO-101."""

from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "assets" / "robotstudio_so101"
MANIFEST_PATH = ROOT / "assets" / "model_manifest.json"


@lru_cache(maxsize=1)
def _template() -> mujoco.MjModel:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Missing model assets. Run: python scripts/fetch_model.py")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = MODEL_DIR / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Model integrity check failed: {path}")
    spec = mujoco.MjSpec.from_file(str(MODEL_DIR / "scene.xml"))
    spec.worldbody.add_site(
        name="reach_target", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.3, 0, 0.2], size=[0.015, 0, 0], rgba=[0.1, 0.9, 0.3, 0.5], group=0,
    )
    return spec.compile()


class SO101ReachEnv(gym.Env):
    """Learn 3-joint reaching with fixed wrist orientation and gripper opening.

    Forward kinematics samples reachable targets; generating joint angles never
    enter observations or controls. Only policy actions move the robot in step().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    home = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 0.5], dtype=np.float64)
    action_low = np.array([-0.5, -1.15, 0.8])
    action_high = np.array([0.5, -0.45, 1.6])
    goal_low = np.array([-0.35, -1.05, 0.9])
    goal_high = np.array([0.35, -0.55, 1.45])
    max_steps = 120
    frame_skip = 10
    success_distance = 0.015
    hold_required = 10
    max_joint_speed = 0.8

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Use render_mode='rgb_array', or the teleop/evaluate viewer CLI.")
        self.render_mode = render_mode
        self.model = copy.copy(_template())
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        if [self.model.joint(i).name for i in range(self.model.njnt)] != list(self.joint_names):
            raise RuntimeError("The pinned model joint layout has changed.")
        self.joint_limits = self.model.jnt_range.copy()
        self.ctrl_limits = np.column_stack((
            np.maximum(self.joint_limits[:, 0], self.model.actuator_ctrlrange[:, 0]),
            np.minimum(self.joint_limits[:, 1], self.model.actuator_ctrlrange[:, 1]),
        ))
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.ee_site = self.model.site("gripperframe").id
        self.target_site = self.model.site("reach_target").id
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32)
        self.goal = np.zeros(3)
        self.steps = 0
        self.hold_steps = 0
        self._renderer = None
        self.model_manifest_digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()

    @property
    def task_config(self) -> dict:
        return {
            "task": "SO101LocalReach-v1", "observation_version": 1,
            "home": self.home.tolist(), "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(), "goal_low": self.goal_low.tolist(),
            "goal_high": self.goal_high.tolist(), "max_steps": self.max_steps,
            "frame_skip": self.frame_skip, "dt": self.dt,
            "success_distance_m": self.success_distance, "hold_required": self.hold_required,
            "max_joint_speed_rad_s": self.max_joint_speed, "reset_jitter_rad": 0.04,
            "minimum_initial_goal_distance_m": 0.04,
            "maximum_hold_joint_speed_rad_s": 0.3,
            "collision_penetration_m": 0.004,
            "reward": "-5*d + 2*(exp(-(d/0.04)^2)-1) - 0.002*sum(qvel[:3]^2) + 10*success - 10*failure",
            "goal_sampling": "collision-free FK in joint box; z>0.08m; min distance 0.04m",
            "observation": "normalized q[:3], qvel[:3]*0.1, normalized ctrl[:3], goal*3, (goal-ee)*5",
        }

    def action_from_joint_positions(self, q: np.ndarray) -> np.ndarray:
        return np.clip(2 * (np.asarray(q)[:3] - self.action_low) / (self.action_high - self.action_low) - 1, -1, 1).astype(np.float32)

    def neutral_action(self) -> np.ndarray:
        return self.action_from_joint_positions(self.data.ctrl[:3])

    @property
    def ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def _observation(self) -> np.ndarray:
        q = 2 * (self.data.qpos[:3] - self.action_low) / (self.action_high - self.action_low) - 1
        return np.concatenate((q, self.data.qvel[:3] * 0.1,
                               self.neutral_action(), self.goal * 3,
                               (self.goal - self.ee_position) * 5)).astype(np.float32)

    def _collision(self, data: mujoco.MjData) -> bool:
        return any(c.dist < -0.004 for c in data.contact)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home
        self.data.qpos[:3] += self.np_random.uniform(-0.04, 0.04, 3)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        initial_position = self.ee_position
        for _ in range(1000):
            self._scratch.qpos[:] = self.home
            self._scratch.qpos[:3] = self.np_random.uniform(self.goal_low, self.goal_high)
            mujoco.mj_forward(self.model, self._scratch)
            candidate = self._scratch.site_xpos[self.ee_site].copy()
            if candidate[2] > 0.08 and np.linalg.norm(candidate - initial_position) >= 0.04 and not self._collision(self._scratch):
                self.goal = candidate
                break
        else:
            raise RuntimeError("Could not sample a collision-free reachable target.")
        self.model.site_pos[self.target_site] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold_steps = 0
        return self._observation(), self._info(False)

    def _info(self, success: bool, failure: bool = False) -> dict:
        return {"is_success": bool(success), "distance": float(np.linalg.norm(self.goal - self.ee_position)),
                "hold_steps": self.hold_steps, "collision": bool(failure), "goal": self.goal.copy()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain three finite numbers.")
        requested = self.home.copy()
        requested[:3] = self.action_low + (np.clip(action, -1, 1) + 1) * 0.5 * (self.action_high - self.action_low)
        # Slew each position command at the physics rate, preserving actuator dynamics.
        for _ in range(self.frame_skip):
            delta = np.clip(requested - self.data.ctrl, -self.max_joint_speed * self.model.opt.timestep,
                            self.max_joint_speed * self.model.opt.timestep)
            self.data.ctrl[:] = np.clip(self.data.ctrl + delta, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        distance = float(np.linalg.norm(self.goal - self.ee_position))
        stable = distance < self.success_distance and np.linalg.norm(self.data.qvel[:3]) < 0.3
        self.hold_steps = self.hold_steps + 1 if stable else 0
        failed = self._collision(self.data) or not np.all(np.isfinite(self.data.qpos))
        success = self.hold_steps >= self.hold_required and not failed
        # Nonpositive step reward prevents collecting reward by delaying success.
        reward = -5 * distance + 2 * (np.exp(-(distance / 0.04) ** 2) - 1) - 0.002 * np.square(self.data.qvel[:3]).sum()
        reward += 10 * success - 10 * failed
        return self._observation(), float(reward), bool(success or failed), bool(self.steps >= self.max_steps and not (success or failed)), self._info(success, failed)

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Construct the environment with render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.17, 0, 0.17]
        camera.distance = 0.8
        camera.azimuth = 135
        camera.elevation = -25
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

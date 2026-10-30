"""SO-101 reaching toward a physical ball resting on a desk or the floor."""

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
TABLE_SIZE = (1.2, 0.7, 0.75)  # Width, depth, and floor-to-top height, in metres.
TABLE_CENTER_XY = (0.4, 0.0)
TABLE_THICKNESS = 0.03
BALL_RADIUS = 0.015
BALL_MASS = 0.02
CONTACT_FRICTION = (0.6, 0.005, 0.0005)  # Sliding, torsional, rolling.
PHYSICS_TIMESTEP = 0.001
CONTACT_SOLREF = (0.002, 1.0)
GOAL_JOINT_LOW = (-0.35, -0.55, 1.48)
GOAL_JOINT_HIGH = (0.35, -0.45, 1.6)
GOAL_RADIAL_OFFSET = (0.006, 0.010)
GOAL_REACH_DISTANCE = 0.014
GOAL_CONTACT_TOLERANCE = 0.000001
MINIMUM_TARGET_CHANGE = 0.02
FLOOR_SPAWN_X = (0.18, 0.5)
FLOOR_SPAWN_ABS_Y = (0.43, 0.65)


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
    spec.option.gravity = [0, 0, -9.81]
    # Resolve impacts of the 30 mm ball while keeping control steps at 20 Hz.
    spec.option.timestep = PHYSICS_TIMESTEP
    spec.option.iterations = 50
    # Compose the workspace without changing any pinned upstream robot asset.
    spec.body("base").pos = [0, 0, TABLE_SIZE[2]]
    floor = spec.geom("floor")
    floor.condim = 6
    floor.friction = CONTACT_FRICTION
    floor.solref = CONTACT_SOLREF
    table = spec.worldbody.add_body(name="table")
    table.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[*TABLE_CENTER_XY, TABLE_SIZE[2] - TABLE_THICKNESS / 2],
        size=[TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_THICKNESS / 2],
        rgba=[0.64, 0.45, 0.26, 1], condim=6,
        friction=CONTACT_FRICTION, solref=CONTACT_SOLREF,
    )
    leg_height = TABLE_SIZE[2] - TABLE_THICKNESS
    for index, (x, y) in enumerate(((-0.14, -0.29), (-0.14, 0.29), (0.94, -0.29), (0.94, 0.29))):
        table.add_geom(
            name=f"table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, leg_height / 2], size=[0.025, 0.025, leg_height / 2],
            rgba=[0.22, 0.24, 0.27, 1], condim=6,
            friction=CONTACT_FRICTION, solref=CONTACT_SOLREF,
        )
    ball = spec.worldbody.add_body(name="ball", pos=[0.24, 0, TABLE_SIZE[2] + BALL_RADIUS])
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0, 0], mass=BALL_MASS, rgba=[0.1, 0.9, 0.3, 1],
        condim=6, friction=CONTACT_FRICTION, solref=CONTACT_SOLREF,
    )
    ball.add_site(name="reach_target", size=[0.001, 0, 0], rgba=[0, 0, 0, 0])
    return spec.compile()


class SO101ReachEnv(gym.Env):
    """Learn 3-joint reaching with fixed wrist orientation and gripper opening.

    The ball moves under gravity and contact forces; its current centre is the
    goal. Default table placements have a reachable, contact-checked FK witness.
    Automatic resets reuse the target until its interaction budget expires.
    Only policy actions move the robot in step().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    home = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 0.5], dtype=np.float64)
    action_low = np.array([-0.5, -1.15, 0.8])
    action_high = np.array([0.5, -0.45, 1.6])
    max_steps = 1200
    target_min_steps = 6000  # Five simulated minutes at 20 Hz, across attempts.
    frame_skip = 50
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
        if [self.model.joint(i).name for i in range(self.model.njnt)] != [*self.joint_names, "ball_free"]:
            raise RuntimeError("The pinned model joint layout has changed.")
        self.joint_limits = self.model.jnt_range[:6].copy()
        self.ctrl_limits = np.column_stack((
            np.maximum(self.joint_limits[:, 0], self.model.actuator_ctrlrange[:, 0]),
            np.minimum(self.joint_limits[:, 1], self.model.actuator_ctrlrange[:, 1]),
        ))
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.ee_site = self.model.site("gripperframe").id
        self.shoulder_body = self.model.body("shoulder").id
        self.target_site = self.model.site("reach_target").id
        self.ball_body = self.model.body("ball").id
        self.ball_geom = self.model.geom("ball_geom").id
        ball_joint = self.model.joint("ball_free").id
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint])
        self.ball_dof_adr = int(self.model.jnt_dofadr[ball_joint])
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32)
        self.ball_surface = "table"
        self._target_spawn_position = None
        self.target_steps = 0
        self.steps = 0
        self.hold_steps = 0
        self._renderer = None
        self.model_manifest_digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()

    @property
    def task_config(self) -> dict:
        return {
            "task": "SO101ReachableSurfaceBall-v4", "observation_version": 2,
            "home": self.home.tolist(), "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(), "max_steps": self.max_steps,
            "frame_skip": self.frame_skip, "dt": self.dt,
            "physics_timestep_s": float(self.model.opt.timestep),
            "episode_duration_seconds": self.max_steps * self.dt,
            "target_min_steps": self.target_min_steps,
            "target_duration_seconds": self.target_min_steps * self.dt,
            "target_reset": "reuse spawn across unseeded resets; change at cumulative step budget; seed or new_target overrides",
            "success_distance_m": self.success_distance, "hold_required": self.hold_required,
            "max_joint_speed_rad_s": self.max_joint_speed, "reset_jitter_rad": 0.0,
            "minimum_initial_goal_distance_m": 0.04,
            "maximum_hold_joint_speed_rad_s": 0.3,
            "collision_penetration_m": 0.004,
            "reward": "-5*d + 2*(exp(-(d/0.04)^2)-1) - 0.002*sum(qvel[:3]^2) + 10*success - 10*failure",
            "goal_sampling": "table only by default; FK witness within action limits and 14mm of ball, all contacts >=-1um; zero initial ball velocity",
            "default_ball_surface": "table",
            "table_size_m": list(TABLE_SIZE), "table_center_xy_m": list(TABLE_CENTER_XY),
            "table_thickness_m": TABLE_THICKNESS, "robot_base_position_m": [0, 0, TABLE_SIZE[2]],
            "table_spawn": {"fk_joint_low_rad": list(GOAL_JOINT_LOW), "fk_joint_high_rad": list(GOAL_JOINT_HIGH),
                            "radial_offset_m": list(GOAL_RADIAL_OFFSET), "maximum_reach_distance_m": GOAL_REACH_DISTANCE,
                            "contact_penetration_tolerance_m": GOAL_CONTACT_TOLERANCE,
                            "minimum_new_target_distance_m": MINIMUM_TARGET_CHANGE},
            "floor_spawn": {"x_m": list(FLOOR_SPAWN_X), "abs_y_m": list(FLOOR_SPAWN_ABS_Y)},
            "floor_goals_reachable": False,
            "ball_radius_m": BALL_RADIUS, "ball_mass_kg": BALL_MASS,
            "gravity_m_s2": self.model.opt.gravity.tolist(),
            "contact_friction_slide_spin_roll": list(CONTACT_FRICTION),
            "contact_solref": list(CONTACT_SOLREF), "contact_dimensions": 6,
            "solver_iterations": int(self.model.opt.iterations),
            "physics_parameters": "nominal uncalibrated ball/surface parameters; no aerodynamic forces",
            "collision_failure": "robot penetration >4mm, excluding ball contacts; support contacts are allowed",
            "observation": "normalized q[:3], qvel[:3]*0.1, normalized ctrl[:3], current ball centre*3, (ball-ee)*5",
        }

    def action_from_joint_positions(self, q: np.ndarray) -> np.ndarray:
        return np.clip(2 * (np.asarray(q)[:3] - self.action_low) / (self.action_high - self.action_low) - 1, -1, 1).astype(np.float32)

    def neutral_action(self) -> np.ndarray:
        return self.action_from_joint_positions(self.data.ctrl[:3])

    @property
    def ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    @property
    def goal(self) -> np.ndarray:
        """Read the current physical ball position, including in manual mode."""
        return self.data.xpos[self.ball_body].copy()

    def _observation(self) -> np.ndarray:
        q = 2 * (self.data.qpos[:3] - self.action_low) / (self.action_high - self.action_low) - 1
        return np.concatenate((q, self.data.qvel[:3] * 0.1,
                               self.neutral_action(), self.goal * 3,
                               (self.goal - self.ee_position) * 5)).astype(np.float32)

    def _collision(self, data: mujoco.MjData) -> bool:
        # Ball/surface and robot/ball contacts are part of the physical task.
        return any(c.dist < -0.004 and self.ball_geom not in (c.geom1, c.geom2)
                   for c in data.contact)

    def _sample_table_target(self, initial_position: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        """Verify an attainable ball placement without controlling the robot.

        FK is evaluated only on scratch state. Its joint angles never enter
        observations, info, controls, or the actual robot's reset pose.
        """
        mujoco.mj_resetData(self.model, self._scratch)
        low = np.maximum(GOAL_JOINT_LOW, self.action_low)
        high = np.minimum(GOAL_JOINT_HIGH, self.action_high)
        for _ in range(5000):
            self._scratch.qpos[:6] = self.home
            self._scratch.qpos[:3] = self.np_random.uniform(low, high)
            mujoco.mj_forward(self.model, self._scratch)
            ee = self._scratch.site_xpos[self.ee_site].copy()
            if abs(ee[2] - (TABLE_SIZE[2] + BALL_RADIUS)) > GOAL_REACH_DISTANCE:
                continue
            candidate = np.array([*ee[:2], TABLE_SIZE[2] + BALL_RADIUS])
            outward = ee[:2] - self._scratch.xpos[self.shoulder_body, :2]
            # Move outside the fixed jaw instead of overlapping its collider.
            candidate[:2] += outward / np.linalg.norm(outward) * self.np_random.uniform(*GOAL_RADIAL_OFFSET)
            if np.linalg.norm(candidate - ee) > min(GOAL_REACH_DISTANCE, self.success_distance):
                continue
            if np.any(np.abs(candidate[:2] - TABLE_CENTER_XY) > np.array(TABLE_SIZE[:2]) / 2 - BALL_RADIUS):
                continue
            if np.linalg.norm(candidate - initial_position) < 0.04:
                continue
            if previous is not None and np.linalg.norm(candidate - previous) < MINIMUM_TARGET_CHANGE:
                continue
            self._scratch.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3] = candidate
            mujoco.mj_forward(self.model, self._scratch)
            if any(contact.dist < -GOAL_CONTACT_TOLERANCE for contact in self._scratch.contact):
                continue
            return candidate
        raise RuntimeError("Could not sample a reachable, contact-safe ball placement on the table.")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        surface = options.get("ball_surface")
        if surface is not None and surface not in ("table", "floor"):
            raise ValueError("ball_surface must be 'table' or 'floor'.")
        if "new_target" in options and not isinstance(options["new_target"], bool):
            raise ValueError("new_target must be a boolean.")
        change_target = (self._target_spawn_position is None or seed is not None
                         or options.get("new_target", False)
                         or self.target_steps >= self.target_min_steps
                         or (surface is not None and surface != self.ball_surface))
        # An explicit seed reproduces a fresh target, independent of past resets.
        previous = self._target_spawn_position if seed is None else None
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        if change_target:
            # Floor placement is an explicit physics diagnostic, never a random
            # learning target for this table-mounted arm.
            self.ball_surface = surface or "table"
            if self.ball_surface == "table":
                position = self._sample_table_target(self.ee_position, previous)
            else:
                side = self.np_random.choice([-1, 1])
                position = [self.np_random.uniform(*FLOOR_SPAWN_X),
                            side * self.np_random.uniform(*FLOOR_SPAWN_ABS_Y), BALL_RADIUS]
            self._target_spawn_position = np.asarray(position, dtype=np.float64).copy()
            self.target_steps = 0
        # The centre starts one radius above its support, never randomly aloft.
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3] = self._target_spawn_position
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold_steps = 0
        return self._observation(), {**self._info(False), "target_changed": bool(change_target)}

    def _info(self, success: bool, failure: bool = False) -> dict:
        return {"is_success": bool(success), "distance": float(np.linalg.norm(self.goal - self.ee_position)),
                "hold_steps": self.hold_steps, "collision": bool(failure), "goal": self.goal,
                "ball_spawn_surface": self.ball_surface,
                "ball_position": self.goal,
                "ball_linear_velocity": self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3].copy(),
                "target_spawn_position": self._target_spawn_position.copy(),
                "target_steps": self.target_steps, "target_min_steps": self.target_min_steps,
                "target_remaining_steps": max(0, self.target_min_steps - self.target_steps),
                "target_elapsed_seconds": self.target_steps * self.dt,
                "target_duration_seconds": self.target_min_steps * self.dt}

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
        self.target_steps += 1
        distance = float(np.linalg.norm(self.goal - self.ee_position))
        stable = distance < self.success_distance and np.linalg.norm(self.data.qvel[:3]) < 0.3
        self.hold_steps = self.hold_steps + 1 if stable else 0
        failed = self._collision(self.data) or not np.all(np.isfinite(self.data.qpos))
        success = self.hold_steps >= self.hold_required and not failed
        # Nonpositive step reward prevents collecting reward by delaying success.
        reward = -5 * distance + 2 * (np.exp(-(distance / 0.04) ** 2) - 1) - 0.002 * np.square(self.data.qvel[:3]).sum()
        reward += 10 * success - 10 * failed
        timed_out = self.steps >= self.max_steps or self.target_steps >= self.target_min_steps
        return self._observation(), float(reward), bool(success or failed), bool(timed_out and not (success or failed)), self._info(success, failed)

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Construct the environment with render_mode='rgb_array'.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        self.configure_camera(camera)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    @staticmethod
    def configure_camera(camera) -> None:
        """Frame the standard desk, robot, and adjacent floor placements."""
        camera.lookat[:] = [0.35, 0, 0.5]
        camera.distance = 2.3
        camera.azimuth = 135
        camera.elevation = -25

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

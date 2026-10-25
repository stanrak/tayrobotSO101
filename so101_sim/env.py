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
TABLE_SPAWN_CENTER = (0.0388353, 0.0)
TABLE_SPAWN_RADIUS = (0.201, 0.207)
TABLE_SPAWN_ANGLE = (-0.32, 0.32)
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
    goal. Floor placements are environmental cases outside this fixed arm's
    reach. Only policy actions move the robot in step().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    home = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 0.5], dtype=np.float64)
    action_low = np.array([-0.5, -1.15, 0.8])
    action_high = np.array([0.5, -0.45, 1.6])
    max_steps = 120
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
        if [self.model.joint(i).name for i in range(self.model.njnt)] != [*self.joint_names, "ball_free"]:
            raise RuntimeError("The pinned model joint layout has changed.")
        self.joint_limits = self.model.jnt_range[:6].copy()
        self.ctrl_limits = np.column_stack((
            np.maximum(self.joint_limits[:, 0], self.model.actuator_ctrlrange[:, 0]),
            np.minimum(self.joint_limits[:, 1], self.model.actuator_ctrlrange[:, 1]),
        ))
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.ee_site = self.model.site("gripperframe").id
        self.target_site = self.model.site("reach_target").id
        self.ball_body = self.model.body("ball").id
        self.ball_geom = self.model.geom("ball_geom").id
        ball_joint = self.model.joint("ball_free").id
        self.ball_qpos_adr = int(self.model.jnt_qposadr[ball_joint])
        self.ball_dof_adr = int(self.model.jnt_dofadr[ball_joint])
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (15,), dtype=np.float32)
        self.ball_surface = "table"
        self.steps = 0
        self.hold_steps = 0
        self._renderer = None
        self.model_manifest_digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()

    @property
    def task_config(self) -> dict:
        return {
            "task": "SO101SurfaceBallReach-v2", "observation_version": 2,
            "home": self.home.tolist(), "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(), "max_steps": self.max_steps,
            "frame_skip": self.frame_skip, "dt": self.dt,
            "physics_timestep_s": float(self.model.opt.timestep),
            "success_distance_m": self.success_distance, "hold_required": self.hold_required,
            "max_joint_speed_rad_s": self.max_joint_speed, "reset_jitter_rad": 0.04,
            "minimum_initial_goal_distance_m": 0.04,
            "maximum_hold_joint_speed_rad_s": 0.3,
            "collision_penetration_m": 0.004,
            "reward": "-5*d + 2*(exp(-(d/0.04)^2)-1) - 0.002*sum(qvel[:3]^2) + 10*success - 10*failure",
            "goal_sampling": "equal-probability table/floor surface placement; zero initial ball velocity",
            "table_size_m": list(TABLE_SIZE), "table_center_xy_m": list(TABLE_CENTER_XY),
            "table_thickness_m": TABLE_THICKNESS, "robot_base_position_m": [0, 0, TABLE_SIZE[2]],
            "table_spawn": {"center_xy_m": list(TABLE_SPAWN_CENTER), "radius_m": list(TABLE_SPAWN_RADIUS), "angle_rad": list(TABLE_SPAWN_ANGLE)},
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        surface = (options or {}).get("ball_surface")
        if surface is not None and surface not in ("table", "floor"):
            raise ValueError("ball_surface must be 'table' or 'floor'.")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home
        self.data.qpos[:3] += self.np_random.uniform(-0.04, 0.04, 3)
        self.data.ctrl[:] = self.data.qpos[:6]
        self.ball_surface = surface or ("table" if self.np_random.integers(2) == 0 else "floor")
        if self.ball_surface == "table":
            radius = self.np_random.uniform(*TABLE_SPAWN_RADIUS)
            angle = self.np_random.uniform(*TABLE_SPAWN_ANGLE)
            position = [TABLE_SPAWN_CENTER[0] + radius * np.cos(angle),
                        TABLE_SPAWN_CENTER[1] + radius * np.sin(angle), TABLE_SIZE[2] + BALL_RADIUS]
        else:
            side = self.np_random.choice([-1, 1])
            position = [self.np_random.uniform(*FLOOR_SPAWN_X),
                        side * self.np_random.uniform(*FLOOR_SPAWN_ABS_Y), BALL_RADIUS]
        # The centre starts one radius above its support, never randomly aloft.
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3] = position
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold_steps = 0
        return self._observation(), self._info(False)

    def _info(self, success: bool, failure: bool = False) -> dict:
        return {"is_success": bool(success), "distance": float(np.linalg.norm(self.goal - self.ee_position)),
                "hold_steps": self.hold_steps, "collision": bool(failure), "goal": self.goal,
                "ball_spawn_surface": self.ball_surface,
                "ball_position": self.goal,
                "ball_linear_velocity": self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3].copy()}

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

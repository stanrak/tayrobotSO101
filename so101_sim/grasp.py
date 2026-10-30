"""Stage-one ball grasping with policy-controlled wrists and gripper.

The legacy reach environment remains available separately. This task requires
physical jaw contacts and a lifted, stable ball; proximity alone is insufficient.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from gymnasium import spaces
import mujoco
import numpy as np

from .env import BALL_MASS, BALL_RADIUS, TABLE_SIZE, SO101ReachEnv


class SO101GraspEnv(SO101ReachEnv):
    """Learn to clamp, lift, and hold the free ball without a scripted controller."""

    lift_clearance = 0.02
    maximum_ball_speed = 0.03
    maximum_ball_angular_speed = 1.0
    maximum_relative_speed = 0.03
    maximum_hold_joint_speed = 0.3
    minimum_jaw_force = 0.005
    maximum_jaw_normal_cosine = -0.25
    pin_force_threshold = 1.0
    pin_support_excess_threshold = 0.5
    pin_maximum_ball_speed = 0.05
    pin_required_seconds = 0.3
    step_cost_scale = 0.01

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)
        # Retain the verified reach envelope while making both wrists and the
        # existing gripper independently controllable within their pinned limits.
        self.grasp_action_low = np.concatenate((self.action_low, self.ctrl_limits[3:, 0]))
        self.grasp_action_high = np.concatenate((self.action_high, self.ctrl_limits[3:, 1]))
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (39,), dtype=np.float32)
        self._pin_physics_steps = 0
        self._pinning_failure = False
        self._pinning_failure_seconds = 0.0
        self._failure_reason = None
        self._step_peak_downward_force = 0.0
        self._base_source_sha256 = hashlib.sha256(
            Path(__file__).with_name("env.py").read_bytes()
        ).hexdigest()
        self.fixed_jaw_geoms = self._pad_geoms(
            "fixed_jaw_", "wrist_roll_follower_so101_gripper_part0_v1",
            excluded={"fixed_jaw_box1", "fixed_jaw_box2"},
        )
        self.moving_jaw_geoms = self._pad_geoms(
            "moving_jaw_", "moving_jaw_so101_gripper_part0_v1",
            "moving_jaw_so101_gripper_part1_v1", excluded={"moving_jaw_box1"},
        )
        self.robot_geoms = frozenset(
            index for index in range(self.model.ngeom)
            if self._descends_from(int(self.model.geom_bodyid[index]), self.model.body("base").id)
        )
        self.support_geoms = frozenset(
            index for index in range(self.model.ngeom)
            if index not in self.robot_geoms and index != self.ball_geom
        )

    def _descends_from(self, body: int, ancestor: int) -> bool:
        while body:
            if body == ancestor:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def _pad_geoms(self, prefix: str, *mesh_names: str, excluded: set[str]) -> frozenset[int]:
        result = set()
        for index in range(self.model.ngeom):
            geom = self.model.geom(index)
            mesh_id = int(geom.dataid[0])
            mesh = self.model.mesh(mesh_id).name if mesh_id >= 0 else ""
            if ((geom.name.startswith(prefix) and geom.name not in excluded)
                    or mesh in mesh_names):
                # A matching visual mesh must never supply grasp evidence.
                if int(geom.contype[0]) or int(geom.conaffinity[0]):
                    result.add(index)
        if not result:
            raise RuntimeError("The pinned jaw collision geometry has changed.")
        return frozenset(result)

    @property
    def task_config(self) -> dict:
        config = super().task_config
        config.pop("success_distance_m", None)
        config.update({
            "task": "SO101BallGraspLiftHold-v1", "observation_version": 1,
            "base_environment_source_sha256": self._base_source_sha256,
            "action_low": self.grasp_action_low.tolist(),
            "action_high": self.grasp_action_high.tolist(),
            "reset_jitter_rad": 0.0,
            "success": "both jaw pads carrying opposed normal force, no support, ball bottom 20mm above table, low world/relative speeds, 10 consecutive steps",
            "approach_distance_diagnostic_m": self.success_distance,
            "lift_clearance_m": self.lift_clearance,
            "maximum_ball_linear_speed_m_s": self.maximum_ball_speed,
            "maximum_ball_angular_speed_rad_s": self.maximum_ball_angular_speed,
            "maximum_ball_ee_relative_speed_m_s": self.maximum_relative_speed,
            "maximum_hold_joint_speed_rad_s": self.maximum_hold_joint_speed,
            "minimum_jaw_normal_force_n": self.minimum_jaw_force,
            "maximum_opposing_jaw_normal_cosine": self.maximum_jaw_normal_cosine,
            "pinning": {
                "downward_robot_force_n": self.pin_force_threshold,
                "excess_support_upward_force_n": self.pin_support_excess_threshold,
                "maximum_ball_speed_m_s": self.pin_maximum_ball_speed,
                "required_seconds": self.pin_required_seconds,
                "sampling": "every physics substep; consecutive supported downward pressing",
            },
            "grasp_pad_geom_ids": {
                "fixed": sorted(self.fixed_jaw_geoms), "moving": sorted(self.moving_jaw_geoms),
            },
            "threshold_status": "initial engineering thresholds; not material/force calibration or learned-performance evidence",
            "step_cost_scale": self.step_cost_scale,
            "reward": "0.01*(-1 -2*d -0.002*sum(qvel[:6]^2) +0.5*opposed_jaws +0.5*opposed_jaws*unsupported*clip(clearance/20mm,0,1) -2*clip(downward_supported_force/1N,0,5)) +10*success -10*failure",
            "observation": "normalized six q; six qvel*0.1; normalized six ctrl; ball centre*3; ball-ee*5; ball world linear and angular velocity; ball-ee relative world linear velocity; fixed/moving/opposed pad contact and support flags; hold and pin fractions",
            "collision_failure": "legacy robot/self/support penetration >4mm, nonfinite state, or sustained force-based ball pinning",
            "ball_lost": "ball centre more than 30mm below desktop",
            "goal_sampling": config["goal_sampling"] + "; reach witness is not proof of a grasp or a collision-free grasp trajectory",
        })
        return config

    def action_from_joint_positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.clip(2 * (q[:6] - self.grasp_action_low)
                       / (self.grasp_action_high - self.grasp_action_low) - 1, -1, 1).astype(np.float32)

    def neutral_action(self) -> np.ndarray:
        return self.action_from_joint_positions(self.data.ctrl[:6])

    def _velocities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ball_velocity = np.zeros(6)
        ee_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.ball_body, ball_velocity, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.ee_site, ee_velocity, 0)
        return ball_velocity[3:], ball_velocity[:3], ball_velocity[3:] - ee_velocity[3:]

    def _contacts(self) -> dict:
        state = {"fixed_jaw_contact": False, "moving_jaw_contact": False,
                 "opposed_jaw_contacts": False, "jaw_normal_cosine": 0.0,
                 "ball_supported": False, "robot_ball_contact": False,
                 "fixed_jaw_normal_force_n": 0.0, "moving_jaw_normal_force_n": 0.0,
                 "fixed_jaw_normal_world_n": np.zeros(3), "moving_jaw_normal_world_n": np.zeros(3),
                 "robot_ball_force_world_n": np.zeros(3),
                 "support_force_world_n": np.zeros(3)}
        for index, contact in enumerate(self.data.contact):
            if self.ball_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom1 if contact.geom2 == self.ball_geom else contact.geom2
            local_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, local_force)
            # MuJoCo reports force from geom1 toward geom2 in contact axes.
            world_force = contact.frame.reshape(3, 3).T @ local_force[:3]
            if contact.geom1 == self.ball_geom:
                world_force = -world_force
            if other in self.support_geoms and contact.dist <= 1e-6:
                state["ball_supported"] = True
                state["support_force_world_n"] += world_force
            if other in self.robot_geoms:
                state["robot_ball_contact"] = True
                state["robot_ball_force_world_n"] += world_force
            for geoms, key in ((self.fixed_jaw_geoms, "fixed"), (self.moving_jaw_geoms, "moving")):
                if other in geoms:
                    state[f"{key}_jaw_normal_force_n"] += max(0.0, float(local_force[0]))
                    direction = contact.frame[:3] if contact.geom2 == self.ball_geom else -contact.frame[:3]
                    state[f"{key}_jaw_normal_world_n"] += max(0.0, float(local_force[0])) * direction
        for key in ("fixed", "moving"):
            state[f"{key}_jaw_contact"] = state[f"{key}_jaw_normal_force_n"] >= self.minimum_jaw_force
        normals = [state[f"{key}_jaw_normal_world_n"] for key in ("fixed", "moving")]
        magnitude = np.linalg.norm(normals[0]) * np.linalg.norm(normals[1])
        if magnitude > 1e-12:
            state["jaw_normal_cosine"] = float(np.clip(np.dot(*normals) / magnitude, -1, 1))
        state["opposed_jaw_contacts"] = bool(state["fixed_jaw_contact"] and state["moving_jaw_contact"]
                                             and state["jaw_normal_cosine"] <= self.maximum_jaw_normal_cosine)
        return state

    @staticmethod
    def _downward_force(contacts: dict) -> float:
        return max(0.0, -float(contacts["robot_ball_force_world_n"][2]))

    def _pinning(self, contacts: dict, linear_velocity: np.ndarray) -> bool:
        excess_support = float(contacts["support_force_world_n"][2]) - BALL_MASS * 9.81
        return bool(contacts["ball_supported"] and contacts["robot_ball_contact"]
                    and self._downward_force(contacts) >= self.pin_force_threshold
                    and excess_support >= self.pin_support_excess_threshold
                    and np.linalg.norm(linear_velocity) <= self.pin_maximum_ball_speed)

    def _record_pinning(self, contacts: dict, linear_velocity: np.ndarray) -> None:
        self._pin_physics_steps = self._pin_physics_steps + 1 if self._pinning(contacts, linear_velocity) else 0
        seconds = self._pin_physics_steps * float(self.model.opt.timestep)
        if seconds >= self.pin_required_seconds - 1e-12:
            self._pinning_failure = True
            self._pinning_failure_seconds = max(self._pinning_failure_seconds, seconds)

    def _grasp_stable(self, contacts: dict, velocities: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
        linear, angular, relative = velocities
        return bool(contacts["fixed_jaw_contact"] and contacts["moving_jaw_contact"]
                    and contacts["opposed_jaw_contacts"]
                    and not contacts["ball_supported"]
                    and self.goal[2] - BALL_RADIUS - TABLE_SIZE[2] >= self.lift_clearance
                    and np.linalg.norm(linear) <= self.maximum_ball_speed
                    and np.linalg.norm(angular) <= self.maximum_ball_angular_speed
                    and np.linalg.norm(relative) <= self.maximum_relative_speed
                    and np.linalg.norm(self.data.qvel[:6]) < self.maximum_hold_joint_speed)

    def _observation(self) -> np.ndarray:
        contacts = self._contacts()
        linear, angular, relative = self._velocities()
        q = 2 * (self.data.qpos[:6] - self.grasp_action_low) / (self.grasp_action_high - self.grasp_action_low) - 1
        return np.concatenate((
            q, self.data.qvel[:6] * 0.1, self.neutral_action(), self.goal * 3,
            (self.goal - self.ee_position) * 5, linear, angular, relative,
            [contacts["fixed_jaw_contact"], contacts["moving_jaw_contact"], contacts["opposed_jaw_contacts"], contacts["ball_supported"],
             min(1.0, self.hold_steps / self.hold_required),
             min(1.0, max(self._pin_physics_steps * self.model.opt.timestep,
                          self._pinning_failure_seconds) / self.pin_required_seconds)],
        )).astype(np.float32)

    def _info(self, success: bool, failure: bool = False) -> dict:
        info = super()._info(success, failure)
        contacts = self._contacts()
        linear, angular, relative = self._velocities()
        info.update(contacts)
        info.update({
            "lift_clearance_m": float(self.goal[2] - BALL_RADIUS - TABLE_SIZE[2]),
            "ball_angular_velocity": angular.copy(),
            "ball_ee_relative_velocity": relative.copy(),
            "ball_linear_velocity": linear.copy(),
            "pinning": bool(self._pinning_failure),
            "failure_reason": self._failure_reason,
            "pinning_seconds": max(self._pin_physics_steps * float(self.model.opt.timestep),
                                   self._pinning_failure_seconds),
            "downward_robot_force_n": self._downward_force(contacts),
            "peak_downward_robot_force_n": self._step_peak_downward_force,
            "termination_reason": "success" if success else self._failure_reason,
        })
        # A force-based pinning failure is distinct from a geometry collision.
        info["collision"] = self._failure_reason == "collision"
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._pin_physics_steps = 0
        self._pinning_failure = False
        self._pinning_failure_seconds = 0.0
        self._failure_reason = None
        self._step_peak_downward_force = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain six finite numbers.")
        requested = self.grasp_action_low + (np.clip(action, -1, 1) + 1) * 0.5 * (
            self.grasp_action_high - self.grasp_action_low)
        self._step_peak_downward_force = 0.0
        self._pinning_failure = False
        self._pinning_failure_seconds = 0.0
        self._failure_reason = None
        for _ in range(self.frame_skip):
            delta = np.clip(requested - self.data.ctrl,
                            -self.max_joint_speed * self.model.opt.timestep,
                            self.max_joint_speed * self.model.opt.timestep)
            self.data.ctrl[:] = np.clip(self.data.ctrl + delta, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
            mujoco.mj_step(self.model, self.data)
            contacts = self._contacts()
            linear, _, _ = self._velocities()
            self._step_peak_downward_force = max(self._step_peak_downward_force, self._downward_force(contacts))
            self._record_pinning(contacts, linear)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.target_steps += 1
        contacts = self._contacts()
        velocities = self._velocities()
        stable = self._grasp_stable(contacts, velocities)
        self.hold_steps = self.hold_steps + 1 if stable else 0
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self._failure_reason = "numerical"
        elif self._pinning_failure:
            self._failure_reason = "pinning"
        elif self.goal[2] < TABLE_SIZE[2] - 0.03:
            self._failure_reason = "ball_lost"
        elif self._collision(self.data):
            self._failure_reason = "collision"
        failed = self._failure_reason is not None
        success = self.hold_steps >= self.hold_required and not failed
        distance = float(np.linalg.norm(self.goal - self.ee_position))
        bilateral = contacts["opposed_jaw_contacts"]
        lift = np.clip((self.goal[2] - BALL_RADIUS - TABLE_SIZE[2]) / self.lift_clearance, 0, 1)
        # All nonterminal rewards remain nonpositive. A supported press earns no
        # lift term, and a sustained downward press terminates the attempt.
        pressing = self._downward_force(contacts) if contacts["ball_supported"] else 0.0
        step_reward = (-1 - 2 * distance - 0.002 * np.square(self.data.qvel[:6]).sum()
                       + 0.5 * bilateral + 0.5 * bilateral * (not contacts["ball_supported"]) * lift
                       - 2 * np.clip(pressing / self.pin_force_threshold, 0, 5))
        reward = self.step_cost_scale * step_reward + 10 * success - 10 * failed
        timed_out = self.steps >= self.max_steps or self.target_steps >= self.target_min_steps
        info = self._info(success, failed)
        if timed_out and not (success or failed):
            info["termination_reason"] = "time_limit"
        return self._observation(), float(reward), bool(success or failed), bool(timed_out and not (success or failed)), info

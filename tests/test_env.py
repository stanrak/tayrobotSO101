"""Checks for the physical task contract, independent of learned behavior."""

import unittest

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from so101_sim.env import SO101ReachEnv


class ReachEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = SO101ReachEnv()
        self.addCleanup(self.env.close)

    def _place_ball(self, position, velocity=(0, 0, 0)):
        """Arrange a physics fixture without generating robot control actions."""
        address = self.env.ball_qpos_adr
        self.env.data.qpos[address:address + 3] = position
        self.env.data.qvel[self.env.ball_dof_adr:] = 0
        self.env.data.qvel[self.env.ball_dof_adr:self.env.ball_dof_adr + 3] = velocity
        mujoco.mj_forward(self.env.model, self.env.data)

    def _physics_for(self, seconds):
        count = round(seconds / self.env.model.opt.timestep)
        self.assertAlmostEqual(count * self.env.model.opt.timestep, seconds)
        mujoco.mj_step(self.env.model, self.env.data, nstep=count)
        mujoco.mj_forward(self.env.model, self.env.data)

    def _ball_contacts(self):
        return [contact for contact in self.env.data.contact
                if self.env.ball_geom in (contact.geom1, contact.geom2)]

    def test_gymnasium_contract(self):
        check_env(self.env)

    def test_reset_targets_require_movement(self):
        surfaces = set()
        positions = {"table": set(), "floor": set()}
        for seed in range(100):
            obs, info = self.env.reset(seed=seed)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertGreaterEqual(info["distance"], 0.04)
            self.assertFalse(info["is_success"])
            self.assertFalse(self.env._collision(self.env.data))
            self.assertTrue(all(contact.dist >= -1e-8 for contact in self.env.data.contact))
            surface = info["ball_spawn_surface"]
            surfaces.add(surface)
            positions[surface].add(tuple(self.env.goal[:2]))
            support_height = 0.75 if surface == "table" else 0.0
            self.assertAlmostEqual(self.env.goal[2] - 0.015, support_height)
            if surface == "table":
                self.assertTrue(-0.2 + 0.015 < self.env.goal[0] < 1.0 - 0.015)
                self.assertLess(abs(self.env.goal[1]), 0.35 - 0.015)
            else:
                self.assertGreater(abs(self.env.goal[1]), 0.35 + 0.015)
            np.testing.assert_array_equal(self.env.data.qvel[self.env.ball_dof_adr:], np.zeros(6))
            np.testing.assert_array_equal(self.env.data.qpos[self.env.ball_qpos_adr + 3:], [1, 0, 0, 0])
        self.assertEqual(surfaces, {"table", "floor"})
        self.assertTrue(all(len(points) > 20 for points in positions.values()))

    def test_surface_override_and_invalid_surface(self):
        for surface in ("table", "floor"):
            first, info = self.env.reset(seed=42, options={"ball_surface": surface})
            self.assertEqual(info["ball_spawn_surface"], surface)
            repeated, _ = self.env.reset(seed=42, options={"ball_surface": surface})
            np.testing.assert_array_equal(first, repeated)
        with self.assertRaisesRegex(ValueError, "ball_surface"):
            self.env.reset(options={"ball_surface": "air"})

    def test_workspace_physical_parameters_are_fixed(self):
        model = self.env.model
        self.assertEqual((model.nu, model.nq, model.nv), (6, 13, 12))
        self.assertEqual(self.env.joint_limits.shape, (6, 2))
        self.assertEqual(self.env.ctrl_limits.shape, (6, 2))
        self.assertEqual(model.joint("ball_free").type, mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
        self.assertAlmostEqual(model.opt.timestep, 0.001)
        self.assertEqual(model.opt.iterations, 50)
        self.assertEqual(self.env.frame_skip, 50)
        self.assertAlmostEqual(self.env.dt, 0.05)
        np.testing.assert_allclose(model.body("base").pos, [0, 0, 0.75])
        tabletop = model.geom("table_top")
        np.testing.assert_allclose(tabletop.size * 2, [1.2, 0.7, 0.03])
        self.assertAlmostEqual(tabletop.pos[2] + tabletop.size[2], 0.75)
        self.assertEqual(sum(model.geom(i).name.startswith("table_leg_") for i in range(model.ngeom)), 4)
        for index, (x, y) in enumerate(((-0.14, -0.29), (-0.14, 0.29), (0.94, -0.29), (0.94, 0.29))):
            leg = model.geom(f"table_leg_{index}")
            np.testing.assert_allclose(leg.size * 2, [0.05, 0.05, 0.72])
            np.testing.assert_allclose(leg.pos, [x, y, 0.36])
            self.assertAlmostEqual(leg.pos[2] - leg.size[2], 0)
        ball = model.geom("ball_geom")
        self.assertAlmostEqual(ball.size[0], 0.015)
        self.assertAlmostEqual(model.body_mass[self.env.ball_body], 0.02)
        np.testing.assert_allclose(model.body_inertia[self.env.ball_body], [0.4 * 0.02 * 0.015 ** 2] * 3)
        for name in ("ball_geom", "floor", "table_top", *(f"table_leg_{index}" for index in range(4))):
            geom = model.geom(name)
            self.assertEqual(geom.condim, 6)
            np.testing.assert_allclose(geom.friction, [0.6, 0.005, 0.0005])
            np.testing.assert_allclose(geom.solref, [0.002, 1])
        config = self.env.task_config
        self.assertEqual(config["task"], "SO101SurfaceBallReach-v2")
        self.assertEqual(config["observation_version"], 2)
        self.assertFalse(config["floor_goals_reachable"])

    def test_gravity_drop_and_support_reaction(self):
        for surface, xy, support_height, support_name in (
            ("table", [0.65, 0], 0.75, "table_top"),
            ("floor", [0.65, 0.55], 0.0, "floor"),
        ):
            with self.subTest(surface=surface):
                self.env.reset(seed=0, options={"ball_surface": surface})
                initial_z = support_height + 0.215
                self._place_ball([*xy, initial_z])
                self._physics_for(0.1)
                self.assertLess(self.env.goal[2], initial_z - 0.04)
                self.assertAlmostEqual(self.env.data.qvel[self.env.ball_dof_adr + 2], -0.981, places=6)
                self._physics_for(1.9)
                self.assertAlmostEqual(self.env.goal[2], support_height + 0.015, delta=0.001)
                self.assertLess(np.linalg.norm(self.env.data.qvel[self.env.ball_dof_adr:]), 1e-5)
                self.assertFalse(self.env._collision(self.env.data))
                support_id = self.env.model.geom(support_name).id
                reaction = 0.0
                for index, contact in enumerate(self.env.data.contact):
                    if {contact.geom1, contact.geom2} == {support_id, self.env.ball_geom}:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.env.model, self.env.data, index, force)
                        reaction += force[0]
                self.assertAlmostEqual(reaction, 0.02 * 9.81, delta=0.002)

    def test_fast_floor_impacts_have_limited_penetration(self):
        # Different drop heights vary the collision's phase within a physics
        # step, catching excessive tunnelling hidden by one lucky trajectory.
        for height in (0.70, 0.75, 0.90):
            with self.subTest(drop_height_m=height):
                self.env.reset(seed=0, options={"ball_surface": "floor"})
                self._place_ball([0.65, 0.55, height + 0.015])
                maximum_penetration = 0.0
                for _ in range(round(1.0 / self.env.model.opt.timestep)):
                    mujoco.mj_step(self.env.model, self.env.data)
                    ball_z = self.env.data.qpos[self.env.ball_qpos_adr + 2]
                    maximum_penetration = max(maximum_penetration, 0.015 - ball_z)
                mujoco.mj_forward(self.env.model, self.env.data)
                self.assertLess(maximum_penetration, 0.0035)
                self.assertAlmostEqual(self.env.goal[2], 0.015, delta=0.0001)

    def test_surface_friction_slows_sliding_and_rolling_ball(self):
        self.env.reset(seed=0, options={"ball_surface": "table"})
        self._place_ball([0.45, 0, 0.765], velocity=[0.15, 0, 0])
        self._physics_for(0.1)
        self.assertGreater(self.env.goal[0], 0.45)
        self.assertLess(self.env.data.qvel[self.env.ball_dof_adr], 0.15)
        self.assertGreater(abs(self.env.data.qvel[self.env.ball_dof_adr + 4]), 0.1)
        self._physics_for(1.9)
        self.assertLess(np.linalg.norm(self.env.data.qvel[self.env.ball_dof_adr:]), 1e-5)
        self.assertAlmostEqual(self.env.goal[2], 0.765, delta=0.001)
        self.assertTrue(self._ball_contacts())

    def test_ball_leaving_table_falls_to_floor(self):
        self.env.reset(seed=0, options={"ball_surface": "table"})
        self._place_ball([0.97, 0, 0.765], velocity=[0.5, 0, 0])
        self._physics_for(0.4)
        self.assertGreater(self.env.goal[0], 1.0)
        self.assertLess(self.env.goal[2], 0.5)
        self._physics_for(1.6)
        self.assertAlmostEqual(self.env.goal[2], 0.015, delta=0.001)
        floor_id = self.env.model.geom("floor").id
        self.assertTrue(any(floor_id in (contact.geom1, contact.geom2) for contact in self._ball_contacts()))
        self.assertFalse(self.env._collision(self.env.data))

    def test_observation_reward_and_info_follow_moving_ball(self):
        self.env.reset(seed=0, options={"ball_surface": "table"})
        self._place_ball([0.65, 0, 0.965])
        initial = self.env.goal
        observation, reward, _, _, info = self.env.step(self.env.neutral_action())
        self.assertLess(self.env.goal[2], initial[2])
        np.testing.assert_array_equal(info["goal"], self.env.goal)
        np.testing.assert_array_equal(info["ball_position"], self.env.goal)
        np.testing.assert_allclose(self.env.data.site_xpos[self.env.target_site], self.env.goal)
        np.testing.assert_allclose(observation[9:12], self.env.goal * 3, rtol=1e-6)
        np.testing.assert_allclose(observation[12:15], (self.env.goal - self.env.ee_position) * 5, rtol=1e-6)
        distance = np.linalg.norm(self.env.goal - self.env.ee_position)
        self.assertAlmostEqual(info["distance"], distance)
        expected = -5 * distance + 2 * (np.exp(-(distance / 0.04) ** 2) - 1)
        expected -= 0.002 * np.square(self.env.data.qvel[:3]).sum()
        self.assertAlmostEqual(reward, expected)

    def test_robot_table_penetration_still_terminates(self):
        self.env.reset(seed=0, options={"ball_surface": "floor"})
        # Deliberately colliding state isolates failure detection from learning.
        self.env.data.qpos[:6] = [0, 1, 1, -0.4, 0, 0.5]
        self.env.data.ctrl[:] = self.env.data.qpos[:6]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertTrue(self.env._collision(self.env.data))
        _, _, terminated, truncated, info = self.env.step(self.env.neutral_action())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertFalse(info["is_success"])

    def test_seed_reproduces_target_and_physics(self):
        action = np.array([0.1, -0.3, 0.7], dtype=np.float32)
        first, _ = self.env.reset(seed=123)
        trajectory = [self.env.step(action)[0] for _ in range(15)]
        second, _ = self.env.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        for expected in trajectory:
            np.testing.assert_array_equal(expected, self.env.step(action)[0])

    def test_commands_respect_limits_and_slew(self):
        self.env.reset(seed=4)
        before = self.env.data.ctrl.copy()
        self.env.step(np.array([100.0, -100.0, 100.0]))
        self.assertTrue(np.all(np.abs(self.env.data.ctrl - before) <= self.env.max_joint_speed * self.env.dt + 1e-9))
        self.assertTrue(np.all(self.env.data.ctrl >= self.env.ctrl_limits[:, 0]))
        self.assertTrue(np.all(self.env.data.ctrl <= self.env.ctrl_limits[:, 1]))
        np.testing.assert_array_equal(self.env.data.ctrl[3:], self.env.home[3:])

    def test_invalid_actions_rejected(self):
        self.env.reset(seed=2)
        for action in ([1, 2], [0, np.nan, 0], [0, np.inf, 0]):
            with self.assertRaises(ValueError):
                self.env.step(action)

    def test_nonterminal_reward_cannot_reward_waiting(self):
        self.env.reset(seed=7)
        self.env.action_space.seed(7)
        for _ in range(30):
            _, reward, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
            if not terminated:
                self.assertLessEqual(reward, 0.0)

    def test_timeout_is_truncation(self):
        # A real floor placement is outside this table-mounted arm's reach.
        self.env.reset(seed=0, options={"ball_surface": "floor"})
        action = self.env.neutral_action()
        for _ in range(self.env.max_steps):
            _, _, terminated, truncated, info = self.env.step(action)
            self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["is_success"])

    def test_success_requires_full_hold(self):
        self.env.reset(seed=0)
        action = self.env.neutral_action()
        for _ in range(30):
            self.env.step(action)
        # A stationary ball fixture isolates the hold detector. This does not
        # generate robot actions, and applies only inside this unit test.
        self.env.model.geom_contype[self.env.ball_geom] = 0
        self.env.model.geom_conaffinity[self.env.ball_geom] = 0
        self._place_ball(self.env.ee_position)
        self.env.data.xfrc_applied[self.env.ball_body, 2] = 0.02 * 9.81
        self.env.hold_steps = 0
        self.env.steps = 0
        for index in range(self.env.hold_required):
            _, _, terminated, truncated, info = self.env.step(action)
            self.assertEqual(info["is_success"], index == self.env.hold_required - 1)
            self.assertEqual(terminated, info["is_success"])
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()

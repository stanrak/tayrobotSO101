"""Checks for the physical task contract, independent of learned behavior."""

import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from so101_sim.env import SO101ReachEnv


class ReachEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SO101ReachEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_gymnasium_contract(self):
        check_env(self.env)

    def test_reset_targets_require_movement(self):
        for seed in range(100):
            obs, info = self.env.reset(seed=seed)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertGreaterEqual(info["distance"], 0.04)
            self.assertFalse(info["is_success"])
            self.assertEqual(self.env.data.ncon, 0)
            self.assertGreater(self.env.goal[2], 0.08)

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
        self.env.reset(seed=0)
        # An impossible goal isolates the time-limit contract from task success.
        self.env.goal = np.array([5.0, 5.0, 5.0])
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
        # A stationary goal fixture tests the hold detector, not a task controller.
        self.env.goal = self.env.ee_position
        self.env.hold_steps = 0
        self.env.steps = 0
        for index in range(self.env.hold_required):
            _, _, terminated, truncated, info = self.env.step(action)
            self.assertEqual(info["is_success"], index == self.env.hold_required - 1)
            self.assertEqual(terminated, info["is_success"])
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()

"""Passive, bounded rollout control measurements from the fixed raw observation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_OBSERVATION_LAYOUT = (
    "normalized q[:3], qvel[:3]*0.1, normalized ctrl[:3], "
    "current ball centre*3, (ball-ee)*5"
)


class _Moments:
    """Keep only fixed-size sufficient statistics, never rollout samples."""

    def __init__(self, size: int):
        self.count = 0
        self.total = np.zeros(size)
        self.absolute_total = np.zeros(size)
        self.minimum = np.full(size, np.inf)
        self.maximum = np.full(size, -np.inf)
        self.absolute_maximum = np.zeros(size)

    def add(self, value: np.ndarray) -> None:
        self.count += 1
        self.total += value
        self.absolute_total += np.abs(value)
        self.minimum = np.minimum(self.minimum, value)
        self.maximum = np.maximum(self.maximum, value)
        self.absolute_maximum = np.maximum(self.absolute_maximum, np.abs(value))

    def metrics(self, name: str, *, vector: bool = False) -> dict[str, float]:
        if not self.count:
            return {}
        result = {}
        for index in range(len(self.total)):
            prefix = f"joint_{index}_{name}" if vector else name
            result.update({
                f"{prefix}_mean": float(self.total[index] / self.count),
                f"{prefix}_min": float(self.minimum[index]),
                f"{prefix}_max": float(self.maximum[index]),
                f"{prefix}_abs_mean": float(self.absolute_total[index] / self.count),
                f"{prefix}_abs_max": float(self.absolute_maximum[index]),
            })
        return result


class RolloutControlDiagnostics:
    """Decode only the known v3 raw 15D layout, without accessing an environment.

    ``actions`` are the collector's existing policy actions and
    ``executed_actions`` its existing clipped actions. Position/velocity metrics
    are reconstructed from float32 observations, so they are observation-derived
    estimates in radians, radians/second, metres, and metres/second. They are not
    direct actuator-force measurements. On a done transition only the VecEnv's
    terminal observation is used; a reset observation is never substituted.
    """

    def __init__(self, signature: Any):
        self.enabled = False
        try:
            task = signature["task_config"]
            low = np.asarray(task["action_low"], dtype=np.float64)
            high = np.asarray(task["action_high"], dtype=np.float64)
            dt = float(task["dt"])
            max_speed = float(task["max_joint_speed_rad_s"])
            hold_speed = float(task["maximum_hold_joint_speed_rad_s"])
            distance = float(task["success_distance_m"])
            self.enabled = bool(
                task["task"] == "SO101ReachableSurfaceBall-v3"
                and task["observation_version"] == 2
                and task["observation"] == _OBSERVATION_LAYOUT
                and signature["observation_shape"] == [15]
                and signature["action_shape"] == [3]
                and np.array_equal(signature["action_low"], [-1, -1, -1])
                and np.array_equal(signature["action_high"], [1, 1, 1])
                and not signature.get("observation_normalization", False)
                and not task.get("observation_normalization", False)
                and low.shape == high.shape == (3,)
                and np.all(np.isfinite(low)) and np.all(np.isfinite(high))
                and np.all(high > low)
                and np.isfinite(dt) and dt > 0
                and np.isfinite(max_speed) and max_speed > 0
                and np.isfinite(hold_speed) and hold_speed > 0
                and distance == 0.015
            )
            if self.enabled:
                self._low = low.copy()
                self._span = high - low
                self._slew_step = max_speed * dt
                self._hold_speed = hold_speed
                self._goal_distance = distance
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        self.reset()

    def reset(self) -> None:
        self._counts = dict.fromkeys((
            "observed_samples", "skipped_samples", "terminal_samples",
            "missing_terminal_observation_samples", "ball_speed_samples",
            "near_15mm_samples", "eligible_hold_samples", "goal_entry_samples",
        ), 0)
        self._vectors = {name: _Moments(3) for name in (
            "raw_action", "executed_action", "requested_target_rad",
            "observed_ctrl_rad", "observed_q_rad", "requested_step_rad",
            "observed_ctrl_step_rad", "tracking_error_rad",
            "request_residual_rad", "joint_speed_rad_s",
            "raw_action_lower_excess", "raw_action_upper_excess",
        )}
        self._scalars = {name: _Moments(1) for name in (
            "joint_speed_norm_rad_s", "ball_speed_m_s", "distance_m",
            "near_15mm_joint_speed_norm_rad_s", "goal_entry_joint_speed_norm_rad_s",
        )}
        self._lower_count = np.zeros(3)
        self._upper_count = np.zeros(3)
        self._slew_count = np.zeros(3)
        self._observed_slew_count = np.zeros(3)

    @staticmethod
    def _batch(value: Any, width: int) -> np.ndarray | None:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
        if array.ndim == 1 and array.shape == (width,):
            return array.reshape(1, width)
        return array if array.ndim == 2 and array.shape[1] == width else None

    def observe(
        self, previous_observations: Any, actions: Any, executed_actions: Any,
        new_observations: Any, dones: Any, infos: Any,
    ) -> None:
        """Read one existing vector transition; skip incomplete/malformed rows."""
        if not self.enabled:
            return
        previous = self._batch(previous_observations, 15)
        raw = self._batch(actions, 3)
        executed = self._batch(executed_actions, 3)
        new = self._batch(new_observations, 15)
        try:
            done_array = np.asarray(dones)
            if done_array.ndim == 0:
                done_array = done_array.reshape(1)
            if done_array.ndim != 1:
                done_array = None
        except (TypeError, ValueError):
            done_array = None
        info_list = infos if isinstance(infos, (list, tuple)) else []
        sizes = [len(array) for array in (previous, raw, executed, new, done_array) if array is not None]
        count = max([len(info_list), *sizes], default=1)
        for index in range(count):
            if any(array is None or index >= len(array) for array in (previous, raw, executed, done_array)):
                self._counts["skipped_samples"] += 1
                continue
            done_value = done_array[index]
            if not isinstance(done_value, (bool, np.bool_, int, np.integer)) or done_value not in (0, 1):
                self._counts["skipped_samples"] += 1
                continue
            done = bool(done_value)
            info = info_list[index] if index < len(info_list) and isinstance(info_list[index], Mapping) else {}
            if done:
                terminal = self._batch(info.get("terminal_observation"), 15)
                if terminal is None or len(terminal) != 1:
                    self._counts["missing_terminal_observation_samples"] += 1
                    self._counts["skipped_samples"] += 1
                    continue
                post = terminal[0]
            elif new is not None and index < len(new):
                post = new[index]
            else:
                self._counts["skipped_samples"] += 1
                continue
            pre = previous[index]
            if not all(np.all(np.isfinite(value)) for value in (pre, post, raw[index], executed[index])):
                self._counts["skipped_samples"] += 1
                continue
            self._observe_row(pre, raw[index], executed[index], post, done, info)

    def _observe_row(self, pre, raw, executed, post, done, info) -> None:
        pre_ctrl = self._low + (pre[6:9] + 1) * 0.5 * self._span
        post_ctrl = self._low + (post[6:9] + 1) * 0.5 * self._span
        post_q = self._low + (post[:3] + 1) * 0.5 * self._span
        requested = self._low + (np.clip(executed, -1, 1) + 1) * 0.5 * self._span
        requested_step = requested - pre_ctrl
        observed_step = post_ctrl - pre_ctrl
        qvel = post[3:6] * 10
        joint_speed = float(np.linalg.norm(qvel))
        pre_distance = float(np.linalg.norm(pre[12:15] / 5))
        post_distance = float(np.linalg.norm(post[12:15] / 5))
        self._counts["observed_samples"] += 1
        self._counts["terminal_samples"] += int(done)
        values = {
            "raw_action": raw, "executed_action": executed,
            "requested_target_rad": requested, "observed_ctrl_rad": post_ctrl,
            "observed_q_rad": post_q, "requested_step_rad": requested_step,
            "observed_ctrl_step_rad": observed_step,
            "tracking_error_rad": post_ctrl - post_q,
            "request_residual_rad": requested - post_ctrl,
            "joint_speed_rad_s": qvel,
            "raw_action_lower_excess": np.maximum(-1 - raw, 0),
            "raw_action_upper_excess": np.maximum(raw - 1, 0),
        }
        for name, value in values.items():
            self._vectors[name].add(value)
        self._lower_count += raw < -1
        self._upper_count += raw > 1
        # The tolerance covers reconstruction/float32 roundoff at the exact
        # slew boundary; this is demand exceeding a control-period budget.
        self._slew_count += np.abs(requested_step) > self._slew_step + 1e-6
        # Full-period observed target movement is separate from the requested
        # demand above and from physical joint/servo-force saturation. Keep a
        # positive threshold even for a synthetic budget below the tolerance.
        observed_tolerance = min(1e-6, self._slew_step * 0.5)
        self._observed_slew_count += np.abs(observed_step) >= self._slew_step - observed_tolerance
        self._scalars["joint_speed_norm_rad_s"].add(np.array([joint_speed]))
        self._scalars["distance_m"].add(np.array([post_distance]))
        try:
            ball_velocity = np.asarray(info.get("ball_linear_velocity"), dtype=np.float64)
            if ball_velocity.shape == (3,) and np.all(np.isfinite(ball_velocity)):
                self._scalars["ball_speed_m_s"].add(np.array([np.linalg.norm(ball_velocity)]))
                self._counts["ball_speed_samples"] += 1
        except (TypeError, ValueError, OverflowError):
            pass
        if post_distance < self._goal_distance:
            self._counts["near_15mm_samples"] += 1
            self._counts["eligible_hold_samples"] += int(joint_speed < self._hold_speed)
            self._scalars["near_15mm_joint_speed_norm_rad_s"].add(np.array([joint_speed]))
            if pre_distance >= self._goal_distance:
                self._counts["goal_entry_samples"] += 1
                self._scalars["goal_entry_joint_speed_norm_rad_s"].add(np.array([joint_speed]))

    def metrics(self) -> dict[str, float]:
        """Return fresh scalar metrics; empty conditional groups have counts only."""
        result = {"enabled": float(self.enabled), **{key: float(value) for key, value in self._counts.items()}}
        for name, moments in self._vectors.items():
            result.update(moments.metrics(name, vector=True))
        for name, moments in self._scalars.items():
            result.update(moments.metrics(name))
        count = self._counts["observed_samples"]
        if count:
            for joint in range(3):
                result[f"joint_{joint}_raw_action_lower_clip_fraction"] = float(self._lower_count[joint] / count)
                result[f"joint_{joint}_raw_action_upper_clip_fraction"] = float(self._upper_count[joint] / count)
                result[f"joint_{joint}_slew_limit_fraction"] = float(self._slew_count[joint] / count)
                result[f"joint_{joint}_observed_slew_limit_fraction"] = float(self._observed_slew_count[joint] / count)
            result["near_15mm_fraction"] = self._counts["near_15mm_samples"] / count
            result["eligible_hold_fraction"] = self._counts["eligible_hold_samples"] / count
        near = self._counts["near_15mm_samples"]
        if near:
            result["near_15mm_eligible_hold_fraction"] = self._counts["eligible_hold_samples"] / near
        return {f"diagnostics/control_{key}": value for key, value in result.items()}

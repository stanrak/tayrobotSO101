"""Learn visual search and physical grasping from the original wrist camera.

All six actions retain the physical grasp task's controller from the first
step. RGB perception supplies actor features; simulator ball state is used
only to score physical contacts, rewards and outcomes, never as its position
or velocity input. No scan trajectory, inverse kinematics or teacher is used.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from gymnasium import spaces
import mujoco
import numpy as np

from .env import BALL_RADIUS
from .grasp import SO101GraspEnv
from .vision import BallDetection, detect_green_ball


def estimate_ball_world(
    detection: BallDetection, *, width: int, height: int, fovy: float,
    camera_position: np.ndarray, camera_rotation: np.ndarray,
) -> np.ndarray | None:
    """Approximate a known-radius sphere centre from RGB size and calibration.

    This pinhole size estimate is biased off-axis and by imperfect silhouettes.
    It is not simulator depth, exact object state or a calibrated depth sensor.
    Camera pose is forward kinematics from the robot's joint configuration.
    """
    if not detection.visible or detection.center_px is None or detection.bbox_px is None:
        return None
    if width < 3 or height < 3 or not math.isfinite(fovy) or not 0 < fovy < 180:
        raise ValueError("Invalid pinhole camera dimensions or field of view.")
    position = np.asarray(camera_position, dtype=np.float64)
    rotation = np.asarray(camera_rotation, dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3) or not (
        np.all(np.isfinite(position)) and np.all(np.isfinite(rotation))
    ):
        raise ValueError("Camera pose must contain finite position and rotation arrays.")
    _, _, box_width, box_height = detection.bbox_px
    radius_pixels = (box_width + box_height - 2) / 4.0
    if radius_pixels < 2.0:
        return None
    focal = height / (2 * math.tan(math.radians(fovy) / 2))
    # The small-angle size estimate is corrected for the central sphere's
    # tangent cone; off-axis/occlusion errors remain and are documented above.
    depth = BALL_RADIUS * math.sqrt(1 + (focal / radius_pixels) ** 2)
    u, v = detection.center_px
    camera_point = np.array([
        (u - (width - 1) / 2) * depth / focal,
        -(v - (height - 1) / 2) * depth / focal,
        -depth,
    ])
    estimate = position + rotation @ camera_point
    return estimate if np.all(np.isfinite(estimate)) else None


class SO101VisualGraspEnv(SO101GraspEnv):
    """A separate 38D RGB-feature policy task with unchanged six-joint actions."""

    camera_name = "wrist_cam"
    camera_width = 160
    camera_height = 120
    vision_memory_seconds = 2.0
    acquisition_frames_required = 3
    minimum_visual_quality = 0.55
    visual_reward_weights = {"visual_acquisition": 0.007, "visual_centering": 0.003,
                             "visual_recent": 0.005}

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)
        self.observation_space = spaces.Box(-np.inf, np.inf, (38,), dtype=np.float32)
        self.camera_id = self.model.camera(self.camera_name).id
        self._vision_renderer = None
        self._clear_vision()

    def _clear_vision(self) -> None:
        self._vision_state_key = None
        self._vision_rgb = None
        self._vision_detection = BallDetection(False)
        self._visual_estimate = None
        self._visual_estimate_time = None
        self._acquisition_frames = 0
        self._visual_acquired = False

    @property
    def task_config(self) -> dict:
        config = super().task_config
        config.update({
            "task": "SO101VisualBallGraspLiftHold-v1", "observation_version": 1,
            "grasp_environment_source_sha256": hashlib.sha256(
                Path(__file__).with_name("grasp.py").read_bytes()).hexdigest(),
            "visual_environment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "vision_source_sha256": hashlib.sha256(
                Path(__file__).with_name("vision.py").read_bytes()).hexdigest(),
            "observation": "38D: six normalized q, six qvel*0.1, six normalized controls; current RGB visible/u/v/bbox width/height/quality; last RGB-derived approximate world centre*3 and centre-minus-end-effector FK*5; estimate age/2s and validity; two virtual jaw contact flags; hold/pin fractions; acquisition fraction and acquired flag. No exact ball position or velocity.",
            "camera": {
                "name": self.camera_name, "width": self.camera_width, "height": self.camera_height,
                "frame_rate_hz": 1.0 / self.dt,
                "fovy_degrees": float(self.model.cam_fovy[self.camera_id]),
                "local_position_m": self.model.cam_pos[self.camera_id].tolist(),
                "local_quaternion_wxyz": self.model.cam_quat[self.camera_id].tolist(),
                "pose": "original pinned wrist camera; no reorientation or target tracking",
                "depth_estimation": "RGB-only known-radius pinhole size approximation; biased off-axis/under occlusion; camera extrinsics from robot FK; no rendered depth/segmentation/object pose",
                "memory_seconds": self.vision_memory_seconds,
                "memory": "last estimate is frozen world position, explicitly stale when current visible=0; invalid position and relative features are zero after expiry; no motion extrapolation",
                "minimum_quality": self.minimum_visual_quality,
                "acquisition_control_frames": self.acquisition_frames_required,
                "reset_image_counts_for_acquisition": False,
            },
            "actor_contact_feedback": "two virtual force-bearing jaw-contact flags; hold/pin progress retained as task feedback; simulator contact scoring is not physical sensor calibration",
            "visual_reward_weights": dict(self.visual_reward_weights),
            "reward": "grasp-v2 bounded physical costs plus nonpositive RGB acquisition/centering/recent-vision deficits; before visual acquisition approach/clamp/lift/hold costs remain full missing costs; maximum total nonterminal magnitude 0.095; terminal +10/-10 unchanged; simulator geometry may score rewards but never supplies actor ball coordinates",
            "success": config["success"] + "; requires three consecutive quality RGB control frames earlier in this attempt; scoring gate only, all six actions always active",
        })
        return config

    def _render_wrist_rgb(self) -> np.ndarray:
        if self._vision_renderer is None:
            self._vision_renderer = mujoco.Renderer(
                self.model, height=self.camera_height, width=self.camera_width,
            )
        self._vision_renderer.update_scene(self.data, camera=self.camera_name)
        return self._vision_renderer.render().copy()

    def _update_vision(self) -> None:
        key = (float(self.data.time), self.steps)
        if key == self._vision_state_key:
            return
        rgb = self._render_wrist_rgb()
        if rgb.shape != (self.camera_height, self.camera_width, 3):
            raise ValueError("Wrist RGB image dimensions differ from the visual task contract.")
        detection = detect_green_ball(rgb)
        estimate = None
        if detection.visible and detection.confidence >= self.minimum_visual_quality:
            estimate = estimate_ball_world(
                detection, width=self.camera_width, height=self.camera_height,
                fovy=float(self.model.cam_fovy[self.camera_id]),
                camera_position=self.data.cam_xpos[self.camera_id],
                camera_rotation=self.data.cam_xmat[self.camera_id].reshape(3, 3),
            )
        self._vision_rgb = rgb.copy()
        self._vision_detection = detection
        if estimate is not None:
            self._visual_estimate = estimate
            self._visual_estimate_time = float(self.data.time)
        # The reset frame is visible to the actor, but acquisition requires
        # three actual control samples (0.15 simulated seconds at 20 Hz).
        if self.steps > 0:
            self._acquisition_frames = self._acquisition_frames + 1 if estimate is not None else 0
            self._acquisition_frames = min(self.acquisition_frames_required, self._acquisition_frames)
            self._visual_acquired |= self._acquisition_frames >= self.acquisition_frames_required
        self._vision_state_key = key

    def _memory(self) -> tuple[float, bool]:
        if self._visual_estimate_time is None:
            return self.vision_memory_seconds, False
        age = max(0.0, float(self.data.time) - self._visual_estimate_time)
        return age, bool(self._visual_estimate is not None and age <= self.vision_memory_seconds)

    @property
    def current_wrist_rgb(self) -> np.ndarray:
        self._update_vision()
        return self._vision_rgb.copy()

    @property
    def vision_detection(self) -> BallDetection:
        self._update_vision()
        return self._vision_detection

    @property
    def visual_acquired(self) -> bool:
        return self._visual_acquired

    @property
    def visual_position_status(self) -> dict:
        """Expose the same explicitly aged RGB estimate for viewer diagnostics."""
        self._update_vision()
        age, valid = self._memory()
        return {"position": self._visual_estimate.copy() if valid else None,
                "age_seconds": age, "valid": valid}

    def _observation(self) -> np.ndarray:
        self._update_vision()
        detection = self._vision_detection
        pixels = np.zeros(6)
        if detection.visible:
            u, v = detection.center_px
            _, _, width, height = detection.bbox_px
            pixels[:] = [1.0, 2 * u / (self.camera_width - 1) - 1,
                         2 * v / (self.camera_height - 1) - 1,
                         width / self.camera_width, height / self.camera_height,
                         detection.confidence]
        age, valid = self._memory()
        estimates = np.concatenate((self._visual_estimate * 3,
                                    (self._visual_estimate - self.ee_position) * 5)) if valid else np.zeros(6)
        contacts = self._contacts()
        q = 2 * (self.data.qpos[:6] - self.grasp_action_low) / (self.grasp_action_high - self.grasp_action_low) - 1
        return np.concatenate((
            q, self.data.qvel[:6] * 0.1, self.neutral_action(), pixels, estimates,
            [min(1.0, age / self.vision_memory_seconds), float(valid),
             contacts["fixed_jaw_contact"], contacts["moving_jaw_contact"],
             min(1.0, self.hold_steps / self.hold_required),
             min(1.0, max(self._pin_physics_steps * self.model.opt.timestep,
                          self._pinning_failure_seconds) / self.pin_required_seconds),
             self._acquisition_frames / self.acquisition_frames_required, float(self._visual_acquired)],
        )).astype(np.float32)

    def _grasp_stable(self, contacts: dict, velocities: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
        self._update_vision()
        return bool(self._visual_acquired and super()._grasp_stable(contacts, velocities))

    def _reward_terms(self, contacts: dict, gaps: np.ndarray) -> dict[str, float]:
        self._update_vision()
        terms = super()._reward_terms(contacts, gaps)
        if not self._visual_acquired:
            for name in ("approach", "clamp", "lift", "hold"):
                terms[name] = -self.reward_weights[name]
        detection = self._vision_detection
        centering = 0.0
        if detection.visible and detection.confidence >= self.minimum_visual_quality:
            u, v = detection.center_px
            offset = np.array([2 * u / (self.camera_width - 1) - 1,
                               2 * v / (self.camera_height - 1) - 1])
            centering = detection.confidence * (1 - min(1.0, float(np.linalg.norm(offset)) / math.sqrt(2)))
        age, valid = self._memory()
        deficits = {
            "visual_acquisition": 0.0 if self._visual_acquired else 1 - self._acquisition_frames / self.acquisition_frames_required,
            "visual_centering": 1 - centering,
            "visual_recent": 1 - max(0.0, 1 - age / self.vision_memory_seconds) if valid else 1.0,
        }
        terms.update({name: -self.visual_reward_weights[name] * float(np.clip(value, 0, 1))
                      for name, value in deficits.items()})
        return terms

    def _info(self, success: bool, failure: bool = False) -> dict:
        self._update_vision()
        info = super()._info(success, failure)
        age, valid = self._memory()
        info.update({
            "vision_visible": self._vision_detection.visible,
            "vision_confidence": self._vision_detection.confidence,
            "vision_detection_reason": self._vision_detection.reason,
            "vision_memory_valid": valid, "vision_estimate_age_seconds": age,
            "vision_estimated_position": self._visual_estimate.copy() if valid else None,
            "vision_acquisition_frames": self._acquisition_frames,
            "vision_acquired": self._visual_acquired,
        })
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._clear_vision()
        return super().reset(seed=seed, options=options)

    def close(self) -> None:
        try:
            if self._vision_renderer is not None:
                self._vision_renderer.close()
                self._vision_renderer = None
        finally:
            super().close()

"""A wrist RGB inset that observes snapshots without controlling the robot."""

from __future__ import annotations

import copy
import math
import time

import mujoco
import numpy as np

from .vision import BallTracker


class WristCameraInset:
    """Render the named mounted camera on its owning thread, at a bounded rate.

    The camera owns model/data copies and reads no ball coordinates, depth,
    segmentation or policy state. Rendering and RGB detection issue no actions.
    """

    def __init__(self, model, *, width=320, height=240, fps=10.0):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Camera fps must be finite and positive.")
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if self.camera_id < 0:
            raise ValueError("The model has no named wrist_cam camera.")
        self.width, self.height = width, height
        self.interval = 1.0 / fps
        self.next_frame = -math.inf
        self.last_simulation_time = -math.inf
        self.renderer = None
        self.tracker = BallTracker()
        self.last_track = None
        self.last_image = None
        self.closed = False

    def update(self, viewer, source_data, *, policy_rgb=None, policy_detection=None,
               policy_acquired=None, policy_position=None) -> bool:
        """Snapshot under the viewer lock; render and publish outside the lock."""
        if self.closed:
            return False
        now = time.monotonic()
        simulation_time = float(source_data.time)
        if now < self.next_frame and simulation_time >= self.last_simulation_time:
            return False
        if policy_rgb is None:
            with viewer.lock():
                mujoco.mj_copyData(self.data, self.model, source_data)
            mujoco.mj_forward(self.model, self.data)
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.renderer.update_scene(self.data, camera=self.camera_id)
            rgb = self.renderer.render()
        else:
            # A visual policy's exact cached image is authoritative. Rendering
            # another resolution could disagree with the actor's detections.
            rgb = np.asarray(policy_rgb)
        track = self.tracker.update(rgb, simulation_time)
        detection = policy_detection if policy_detection is not None else track.detection
        image = self._annotate(rgb, detection)
        viewport = getattr(viewer, "viewport", mujoco.MjrRect(0, 0, 640, 480))
        if viewport is None:
            # Native viewer destruction can race with a frame in progress.
            return False
        scale = min(1.0, max(1, viewport.width - 24) / self.width,
                    max(1, viewport.height - 24) / self.height)
        display_width = max(1, int(self.width * scale))
        display_height = max(1, int(self.height * scale))
        rows = np.minimum((np.arange(display_height) * rgb.shape[0] / display_height).astype(int), rgb.shape[0] - 1)
        columns = np.minimum((np.arange(display_width) * rgb.shape[1] / display_width).astype(int), rgb.shape[1] - 1)
        display_image = np.ascontiguousarray(image[rows[:, None], columns])
        # Handle.set_images expects top-origin RGB and performs the GL flip.
        viewer.set_images((mujoco.MjrRect(viewport.left + 12, viewport.bottom + 12,
                                        display_width, display_height), display_image))
        if detection.visible:
            u, v = detection.center_px
            status = f"Ball pixel: ({u:.1f}, {v:.1f}) | score: {detection.confidence:.2f}"
        else:
            status = f"Ball: NOT VISIBLE ({detection.reason})"
        source = "policy camera" if policy_rgb is not None else "observer only"
        policy_status = ("RGB features | " + ("ACQUIRED" if policy_acquired else "SEARCHING")) if policy_rgb is not None else "camera is not a policy input"
        position_status = ""
        if policy_position is not None:
            if policy_position["valid"]:
                position = policy_position["position"]
                measured = "observed" if policy_position["age_seconds"] <= 1e-9 else "last seen"
                position_status = (f"\nRGB estimate ~ ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m; "
                                   f"{measured}, age {policy_position['age_seconds']:.2f}s")
            else:
                position_status = "\nRGB position estimate unavailable"
        viewer.set_texts((
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"Wrist RGB | {source}\n" + status,
            f"wrist_cam | {rgb.shape[1]}x{rgb.shape[0]} | {policy_status}\n"
            f"Image time: {simulation_time:.2f} sim s | display up to {1/self.interval:g} Hz" + position_status,
        ))
        self.last_track, self.last_image = track, image
        self.last_simulation_time = simulation_time
        self.next_frame = time.monotonic() + self.interval
        return True

    @staticmethod
    def _annotate(rgb, detection):
        image = np.array(rgb, dtype=np.uint8, copy=True)
        if detection.visible:
            x, y, width, height = detection.bbox_px
            x2, y2 = x + width - 1, y + height - 1
            image[y:y + 2, x:x2 + 1] = [255, 230, 0]
            image[max(y, y2 - 1):y2 + 1, x:x2 + 1] = [255, 230, 0]
            image[y:y2 + 1, x:x + 2] = [255, 230, 0]
            image[y:y2 + 1, max(x, x2 - 1):x2 + 1] = [255, 230, 0]
            u, v = (int(round(value)) for value in detection.center_px)
            image[max(0, v - 4):v + 5, u] = [255, 255, 255]
            image[v, max(0, u - 4):u + 5] = [255, 255, 255]
        return image

    def close(self):
        self.closed = True
        renderer, self.renderer = self.renderer, None
        if renderer is not None:
            renderer.close()

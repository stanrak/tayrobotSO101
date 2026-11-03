"""Bounded RGB-only ball observation, independent of physics and policy actions.

The green colour/round silhouette heuristic is a simulation baseline. Its score
is a shape quality indicator, not a calibrated probability or a 3D position.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MAX_IMAGE_PIXELS = 1024 * 1024
MAX_COMPONENT_RUNS = 16384
MIN_BALL_PIXELS = 16


@dataclass(frozen=True)
class BallDetection:
    """A current image observation; absent targets never carry stale pixels."""

    visible: bool
    center_px: tuple[float, float] | None = None
    bbox_px: tuple[int, int, int, int] | None = None  # x, y, width, height
    confidence: float = 0.0
    reason: str = "not_visible"


@dataclass(frozen=True)
class BallTrack:
    """Timestamped current detection and bounded counters for this time epoch."""

    detection: BallDetection
    simulation_time: float
    epoch: int
    frames_seen: int
    visible_frames: int
    lost_frames: int


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int, float, float]] | None:
    """Label 8-connected horizontal runs with bounded union-find storage."""
    parents: list[int] = []
    runs: list[tuple[int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y, row in enumerate(mask):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        if len(runs) + len(starts) > MAX_COMPONENT_RUNS:
            return None
        current: list[tuple[int, int, int]] = []
        first_previous = 0
        for start_value, end_value in zip(starts, ends, strict=True):
            start, end = int(start_value), int(end_value)
            index = len(parents)
            parents.append(index)
            runs.append((y, start, end))
            while first_previous < len(previous) and previous[first_previous][1] < start - 1:
                first_previous += 1
            candidate = first_previous
            while candidate < len(previous) and previous[candidate][0] <= end + 1:
                other = root(previous[candidate][2])
                own = root(index)
                if other != own:
                    parents[other] = own
                candidate += 1
            current.append((start, end, index))
        previous = current

    # Each item is area, min x/y, max x/y, sum x/y. No pixel history is kept.
    combined: dict[int, list[float | int]] = {}
    for index, (y, start, end) in enumerate(runs):
        key = root(index)
        count = end - start + 1
        sum_x = (start + end) * count / 2
        if key not in combined:
            combined[key] = [count, start, y, end, y, sum_x, y * count]
        else:
            item = combined[key]
            item[0] += count
            item[1] = min(item[1], start)
            item[2] = min(item[2], y)
            item[3] = max(item[3], end)
            item[4] = max(item[4], y)
            item[5] += sum_x
            item[6] += y * count
    return [tuple(item) for item in combined.values()]


def detect_green_ball(rgb: np.ndarray) -> BallDetection:
    """Find one unclipped round green blob using only a uint8 RGB image.

    Multiple plausible blobs are ambiguous. Large green regions, thin shapes,
    clipped candidates and excessively complex masks are reported as absent.
    The detector never reads segmentation, depth, camera pose or ball state.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image with shape (height, width, 3).")
    height, width = rgb.shape[:2]
    if height < 3 or width < 3 or height * width > MAX_IMAGE_PIXELS:
        raise ValueError("RGB image dimensions must be >=3 and at most 1048576 pixels.")
    # Integer inequalities avoid uint8 overflow and an extra HSV dependency.
    colour = rgb.astype(np.int32)
    red, green, blue = colour[..., 0], colour[..., 1], colour[..., 2]
    mask = (green >= 50) & (green * 100 > red * 135) & (green * 100 > blue * 120)
    components = _components(mask)
    if components is None:
        return BallDetection(False, reason="too_complex")
    candidates: list[BallDetection] = []
    clipped = False
    non_round = False
    for area, min_x, min_y, max_x, max_y, sum_x, sum_y in components:
        if area < MIN_BALL_PIXELS:
            continue
        box_width, box_height = max_x - min_x + 1, max_y - min_y + 1
        aspect = min(box_width, box_height) / max(box_width, box_height)
        fill = area / (box_width * box_height)
        if area > height * width * 0.3 or aspect < 0.65 or not 0.55 <= fill <= 0.90:
            non_round = True
            continue
        if min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1:
            clipped = True
            continue
        shape_quality = aspect * max(0.0, 1.0 - abs(fill - math.pi / 4) / 0.4)
        candidates.append(BallDetection(
            True, (float(sum_x / area), float(sum_y / area)),
            (int(min_x), int(min_y), int(box_width), int(box_height)),
            float(np.clip(shape_quality, 0.0, 1.0)), "detected",
        ))
    if len(candidates) > 1 or (candidates and clipped):
        return BallDetection(False, reason="ambiguous")
    if candidates:
        return candidates[0]
    reason = "border_clipped" if clipped else "shape_rejected" if non_round else "not_visible"
    return BallDetection(False, reason=reason)


class BallTracker:
    """Observe every supplied frame without predicting a missing target."""

    def __init__(self) -> None:
        self.epoch = 0
        self.last_time: float | None = None
        self.frames_seen = 0
        self.visible_frames = 0
        self.lost_frames = 0

    def update(self, rgb: np.ndarray, simulation_time: float) -> BallTrack:
        simulation_time = float(simulation_time)
        if not math.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError("simulation_time must be finite and nonnegative.")
        detection = detect_green_ball(rgb)
        if self.last_time is not None and simulation_time < self.last_time:
            self.epoch += 1
            self.frames_seen = self.visible_frames = self.lost_frames = 0
        self.last_time = simulation_time
        self.frames_seen += 1
        self.visible_frames += int(detection.visible)
        self.lost_frames = 0 if detection.visible else self.lost_frames + 1
        return BallTrack(detection, simulation_time, self.epoch, self.frames_seen,
                         self.visible_frames, self.lost_frames)

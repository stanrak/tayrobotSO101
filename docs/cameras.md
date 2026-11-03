# Wrist camera, visual search and grasping

[Training](training.md) · [Grasping](grasping.md) · [Manual control](getting-started.md)

The robot has an original named MuJoCo camera, `wrist_cam`, attached to its gripper. Its position, orientation, 61-degree field of view and the exact six-joint home pose remain unchanged. The camera moves with the arm. It is not automatically pointed at the ball.

The original camera may not see the ball at home. The visual task starts there anyway: all six joints are available from the first policy action so the robot can learn to bring the ball into view. There is no programmed scanning motion, inverse-kinematics controller or automatic gripper closure.

## Watch the camera

For manual control with a camera inset:

```powershell
.\.venv\Scripts\python.exe -m so101_sim --wrist-camera
```

The inset appears in the lower-left corner of the same viewer. A yellow box and white cross identify a visible green ball. Missing, clipped or ambiguous detections show `NOT VISIBLE` without a stale target marker. The text includes the image's simulation timestamp.

For a fresh visual-learning run with the inset:

```powershell
.\.venv\Scripts\python.exe -m so101_sim.train --task vision --timesteps 300000 --envs 1 --viewer --wrist-camera --seed 42
```

The training CLI defaults to `--task vision`. One environment is a useful initial interactive configuration because each visual environment owns a renderer and processes an RGB frame on every 20 Hz control step. The inset displays at most ten times per wall-clock second and reuses the policy's exact cached 160×120 image and detection. Enlarging the inset does not change policy resolution. Graphics and image processing add work; this feature does not establish faster training.

`SEARCHING` means the policy has not yet acquired the target during this attempt. `ACQUIRED` means it has observed three consecutive sufficiently clear control frames. The display does not drive the robot. In manual mode or `--task state`, the camera is an observer only; the state task still receives simulator ball coordinates.

## What the visual policy receives

`SO101VisualBallGraspLiftHold-v1` uses an MLP with 38 features, derived from camera perception and robot feedback:

| Input | Values |
| --- | ---: |
| Six joint positions, six joint velocities and six actuator commands | 18 |
| Current RGB visibility, normalized pixel centre, box width/height and shape quality | 6 |
| Last image-derived approximate world centre and its offset from current end-effector FK | 6 |
| Estimate age and memory validity | 2 |
| Two virtual jaw-contact flags | 2 |
| Physical hold and pin progress | 2 |
| Visual acquisition progress and acquired-this-attempt flag | 2 |

The actor does not receive exact simulator ball coordinates or velocities. RGB is processed by a bounded green-colour and round-shape detector; the policy learns joint actions from its features. This is a perception baseline for the known green ball, rather than an end-to-end image network or a general object recognizer. Shape quality is a heuristic, not a calibrated probability. Virtual contact feedback is simulation task feedback, not evidence that physical tactile sensors are installed.

Three consecutive quality RGB control samples establish acquisition, taking at least 0.15 simulated seconds. The reset image does not count. Before acquisition, the approach/clamp/lift/hold terms retain their full missing-stage costs, and blind grasping cannot count as success. Once acquired, those costs can decrease through physical grasp progress. Acquisition, visual centering and recent visual evidence also affect reward.

This sequencing changes rewards and success scoring. It never replaces policy actions, locks joints into a scan, moves the camera independently or supplies a target joint pose. The policy must learn the ordering. Ball physics, collision/pinning rules, exact-home resets, five-minute target budget and six-actuator slew limits remain active. Nonterminal raw costs are bounded by 0.095 per control step, with terminal success/failure still +10/−10. The trainer retains its 0.01 reward multiplier and existing PPO settings.

## From pixels to a position estimate

A pixel identifies a viewing ray, so a single uncalibrated RGB image does not uniquely give metric 3D position. The current simulation baseline uses the known 30 mm sphere diameter, observed apparent size and the camera's fixed calibration to estimate depth. Robot joint forward kinematics transforms that estimate into world coordinates. It does not read rendered depth, segmentation or the true ball pose.

This monocular size estimate is approximate. Off-axis projection, occlusion, small silhouettes and inaccurate calibration can bias it. The detector rejects some unusable silhouettes, but passing its checks is not a guarantee of accurate depth. The camera model and projection conventions are described in the official [MuJoCo camera reference](https://mujoco.readthedocs.io/en/stable/modeling.html#cameras) and [OpenCV calibration reference](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html); OpenCV is not an installed dependency of this feature.

Each visible frame can update the estimate. During a brief disappearance the observation retains the last estimated world position for at most two simulated seconds, with explicit age and validity. It is a frozen last observation, not a new measurement or a motion prediction. After expiry, position features become zero with validity false. The current visibility flag remains false throughout occlusion, and there is no hidden simulator-coordinate fallback.

Continuous processing therefore does not mean the ball is visible continuously. The original wrist view can lose the target outside its field of view or behind the jaws. Learning must handle reacquisition and short-term memory. A future real RGB-D sensor could provide depth if actual hardware supports it; simulator depth would need to be labelled as simulated depth. Intersecting rays with the tabletop can help before lifting, but cannot track a lifted ball correctly.

## Compatibility and verification limits

Vision and state tasks have different actor inputs and checkpoint signatures. Start a fresh visual policy; do not resume a state or historical reach checkpoint into it. Use `--task state` explicitly for a compatible state-policy experiment. Standalone evaluation has a matching `--task` option; use the task recorded with the checkpoint.

Tests verify image-only observation construction, missing-target handling, acquisition scoring, camera/home preservation, all-six-joint control and renderer lifecycle. Static visibility fixtures establish that the original camera can see the ball at some poses. They do not demonstrate a safe search trajectory, a learned search policy or successful pickup. Actual learning quality needs separately measured training and matching complete evaluations.

# Fixed physical workspace and task contract

Read the [instruction index](instruction.md), [training contract](training.md), and [workflow](workflow.md). Implement the composed workspace in `so101_sim/env.py`; keep pinned upstream robot assets and their manifest hashes unchanged.

## Desk and ball physics

| Property | Required value |
| --- | --- |
| Desk size | `1.2 x 0.7 x 0.75 m` |
| Desk top | `0.03 m` thick; four collision-enabled legs |
| Desk world center x/y | `[0.4, 0] m` |
| Robot base mount | `[0, 0, 0.75] m` |
| Ball | Free physical body; radius `0.015 m`, mass `0.02 kg` |
| Gravity | `[0, 0, -9.81] m/s^2` |
| Ball/support contact | `condim=6`, friction `[0.6, 0.005, 0.0005]`, solref `[0.002, 1]` |
| Physics | Timestep `0.001 s`, up to `50` solver iterations |
| Policy control | `frame_skip=50`, giving `20 Hz` |

These are nominal, uncalibrated material parameters. Do not claim an exact real-material match.

## Reachable goal sampling

- Default resets place a stationary ball directly on the table within verified robot reach.
- Sample scratch forward-kinematics poses inside existing action limits, offset the ball center radially outward by 6-10 mm on the tabletop, and reject candidates farther than 14 mm from the end-effector or with contact penetration greater than 1 micrometre.
- Keep at least 40 mm initial robot-to-ball distance and 20 mm between consecutive automatic/manual-N target spawns.
- Seeded resets reproduce independently of the previous target.
- Never expose generating joint angles in observations, info, or controls, or use those angles as robot actions. The scratch reachability check is not a controller or teacher.
- The table-mounted arm cannot reach the floor. Exclude floor placements from default learning/manual sampling.
- `options={"ball_surface": "floor"}` is an explicit physics diagnostic only; keep its samples outside the desk footprint and legs.
- Do not silently move the robot, add a solver, migrate controllers, or expand the learning task to compensate for floor reachability.
- Never restore floating targets or pin/teleport the ball during simulation steps. Goal observations and distance follow the ball's physical motion.

## Target and attempt cadence

- Each target receives 1200 actual control steps per environment, equivalent to 60 simulated seconds at 20 Hz, across unseeded automatic resets and PPO rollouts.
- Early success/collision resets restore the same spawn and retain the cumulative target count.
- Truncate at the cumulative target boundary; resample on the following reset. Each attempt also has its own 1200-step limit.
- Seeded reset starts a fresh reproducible target. `options={"new_target": True}` and manual N explicitly override the dwell.
- Target and attempt budgets do not authorize training or guarantee learning. They are distinct from total trainer transitions and the startup trial's one-update limit.

## Contact and checkpoint semantics

- Permit ball support and robot/ball contacts.
- Robot self-contact and contact with the table/floor fail on penetration deeper than 4 mm.
- Keep the six robot joints/actuators separate from the ball's seven qpos and six qvel coordinates.
- Preserve geometry, physics, reachable placement, target cadence, contact filtering, and live goal tracking under their physical contract tests.
- Task `SO101ReachableSurfaceBall-v3` and checkpoint environment signatures identify the changed surface-ball semantics. Historical checkpoints remain incompatible; do not migrate them automatically.
- Further object/dynamics changes require a user request authorizing those specific changes.

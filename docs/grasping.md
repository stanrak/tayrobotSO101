# Stage one: grasp, lift and hold

[Environment](environment.md) · [Training](training.md) · [Development](development.md)

The default training and checkpoint evaluation environment is `SO101BallGraspLiftHold-v1`, implemented by `SO101GraspEnv` in [grasp.py](../so101_sim/grasp.py). The aim is to grip the free ball between the two jaws, lift it clear of the tabletop and hold it steadily. Implemented success checks do not demonstrate that an untrained or historical policy can accomplish this sequence.

## Control and observations

All six actions lie in `[-1, 1]` and request joint positions. The first three retain the existing reach envelope; wrist flex, wrist roll and gripper use the intersections of the pinned joint/actuator limits. Commands slew at no more than 0.8 rad/s at the 1 kHz physics rate. There is no automatic closure, lifting motion, solver, teacher or grasp attachment.

The observation contains 39 float32 values:

| Group | Count |
| --- | ---: |
| Six normalized joint positions | 6 |
| Six joint velocities scaled by 0.1 | 6 |
| Six normalized actuator commands | 6 |
| Ball world position scaled by 3 | 3 |
| Ball minus gripperframe position scaled by 5 | 3 |
| Ball world linear velocity | 3 |
| Ball world angular velocity | 3 |
| Ball minus gripperframe world linear velocity | 3 |
| Fixed pad contact, moving pad contact, opposed pad normals, support flag | 4 |
| Consecutive stable-hold and supported-pin fractions | 2 |

The state uses simulator measurements, including contact and ball position. Camera images are not inputs to this policy. The retained reach task has a different 15-value observation and three actions.

## Physical success

For 10 consecutive control steps (0.5 simulated seconds), all conditions must hold:

- Both actual jaw pad groups carry at least 0.005 N normal force each. Camera housings and jaw bases do not count as pad evidence. The force-weighted pad normals must oppose each other (cosine at most -0.25), rejecting a ball merely resting on two upward-facing pads.
- The bottom of the ball is at least 20 mm above the tabletop, without table, floor or other static support contact.
- Ball linear speed is at most 0.03 m/s and angular speed at most 1 rad/s.
- Ball-to-gripperframe relative linear speed is at most 0.03 m/s.
- The norm of all six joint velocities is below 0.3 rad/s and no failure is present.

Mere proximity, a ball pinned against the desk, single-jaw contact and a briefly tossed ball do not satisfy this definition. These thresholds are initial engineering choices, not calibrated force/material limits.

## Supported pressing and retries

The environment measures robot-on-ball contact force in world axes at every physics substep. A stationary or slowly moving supported ball is classified as pinned when downward robot force reaches 1 N and the upward support reaction exceeds the ball weight by 0.5 N, continuously for 0.3 simulated seconds. Brief legitimate contact while closing the jaws is allowed. The detector ends the failed attempt; it does not move the arm out of the jam or supply recovery actions.

Geometry penetration deeper than 4 mm still fails. A ball falling more than 30 mm below desktop height is lost; nonfinite robot/ball state also fails. Failure reasons distinguish pinning, collision, ball loss and numerical failure. Automatic episode reset restores the robot to exact home and reuses the same spawn until the target budget expires.

Targets last 6,000 control steps per environment, or five simulated minutes, across retries and PPO updates. Attempts retain their 1,200-step limit. At target expiry the following reset samples another reachable tabletop location; seeded resets and explicit new-target requests override the timer. The ball remains free under gravity/contact during each attempt. Manual mode uses the same five-minute interval while preserving all-six-joint keyboard control.

## Reward and diagnostics

The raw reward combines a nonpositive time/distance/velocity cost with partial credit for opposed jaw contact and unsupported lift, a supported downward-force penalty, a success bonus and a failure penalty. The entire nonterminal shaping term is scaled by 0.01 inside the task, while terminal success/failure remain +10/-10. This balances ordinary waiting costs against early failure; it is separate from the unchanged trainer RewardScale multiplier. Nonterminal reward stays nonpositive; proximity alone supplies no grasp success. Exact coefficients and thresholds are saved in `task_config`. Reward shaping is a starting design and needs controlled learning evidence before claiming improvement.

Generic PPO action/critic diagnostics operate on all six actions. Passive grasp logs record bilateral-contact fraction, opposed-contact fraction, unsupported-lift fraction, maximum ball clearance, maximum downward robot force and pinning failures. A lifted sample alone is not necessarily a bilateral grasp. The older observation-decoded three-joint control diagnostics deliberately disable themselves for this new layout. Evaluation records preserve pinning reasons and maximum lift/force separately from actual success.

## Compatibility and verification limits

This is a separate task and observation/action contract. Checkpoint signatures include the grasp implementation and the shared base environment source hash. Earlier reach checkpoints are rejected; they are not converted or reused automatically. Manual startup remains separate from training.

The existing scratch forward-kinematics sampler proves contact-safe reach proximity inside the chosen action envelope. It does not certify a complete collision-free grasp/lift trajectory for every spawn. Contact fixtures can verify that the unchanged jaws physically support a free ball, but an arranged already-grasped fixture does not prove a policy can acquire it from the table. Grasp acquisition, coverage over sampled positions, reward tuning and generalization remain empirical learning questions.

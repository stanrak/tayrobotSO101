"""Reproducible held-out evaluation and optional MuJoCo policy playback."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable

import numpy as np
import mujoco
from stable_baselines3 import PPO
import torch

from .grasp import SO101GraspEnv
from .outputs import OutputError, current_checkpoint, evaluation_output


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")


def write_json(path: Path, value: Any) -> None:
    """Write ordinary JSON, rejecting NaN rather than hiding invalid metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in PACKAGES}


def metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".metadata.json")


def environment_signature(env: SO101GraspEnv) -> dict[str, Any]:
    """Describe task semantics and immutable assets used by a checkpoint."""
    config = env.task_config
    if callable(config):
        config = config()
    # Round-trip to ordinary JSON types, including a stable representation of lists.
    config = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    manifest_path = ROOT / "assets" / "model_manifest.json"
    manifest_bytes = manifest_path.read_text(encoding="utf-8").encode("utf-8")
    manifest = json.loads(manifest_bytes)
    asset_dir = ROOT / "assets" / manifest["model"]
    for relative, expected in manifest["sha256"].items():
        asset_path = asset_dir / relative
        actual = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Model asset integrity check failed: {asset_path}")
    return {
        "task_config": config,
        "model_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_repository": manifest["repository"],
        "model_commit": manifest["commit"],
        "environment_source_sha256": hashlib.sha256(
            Path(inspect.getfile(type(env))).read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest(),
        "observation_shape": list(env.observation_space.shape),
        "observation_dtype": str(env.observation_space.dtype),
        "action_shape": list(env.action_space.shape),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
    }


def save_checkpoint(
    model: PPO,
    checkpoint: Path,
    signature: dict[str, Any],
    training: dict[str, Any],
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    write_json(
        metadata_path(checkpoint),
        {
            "schema_version": 1,
            "algorithm": "PPO",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
            "environment": signature,
            "training": training,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )


def load_checked_policy(checkpoint: Path, env: SO101GraspEnv) -> tuple[PPO, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix != ".zip":
        checkpoint = checkpoint.with_suffix(".zip")
    metadata = json.loads(metadata_path(checkpoint).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("algorithm") != "PPO":
        raise ValueError("Unsupported checkpoint metadata schema or algorithm.")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata.get("checkpoint_sha256"):
        raise ValueError("Checkpoint contents do not match its metadata SHA256.")
    current = environment_signature(env)
    if metadata.get("environment") != current:
        expected = metadata.get("environment", {})
        differences = sorted(key for key in set(expected) | set(current) if expected.get(key) != current.get(key))
        raise ValueError("Checkpoint environment mismatch: " + ", ".join(differences))
    current_versions = package_versions()
    if metadata.get("package_versions") != current_versions:
        raise ValueError(
            "Checkpoint package versions differ. Use the recorded requirements: "
            + json.dumps(metadata.get("package_versions"), sort_keys=True)
        )
    return PPO.load(str(checkpoint), env=env, device="cpu"), metadata


def _success_interval(successes: int, count: int) -> list[float]:
    """Wilson 95% interval, including meaningful bounds for 0% and 100%."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def _sync_playback_view(viewer, model, data, state: np.ndarray) -> None:
    """Publish policy physics while ignoring native reset, slider and option edits."""
    with viewer.lock():
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        options = copy.copy(model.opt)
        viewer.sync()
        for name in dir(options):
            if not name.startswith("_"):
                setattr(model.opt, name, getattr(options, name))
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        viewer.sync(state_only=True)


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _episode_results(
    policy: str,
    requested_episodes: int,
    rows: list[dict[str, Any]],
    *,
    partial_episode: dict[str, Any] | None = None,
    control_steps: int = 0,
) -> dict[str, Any]:
    """Summarize completed attempts only; unfinished attempts stay separate."""
    summary = None
    if rows:
        successes = sum(row["is_success"] for row in rows)
        summary = {
            "episodes": len(rows),
            "successes": successes,
            "collision_episodes": sum(row["collision"] for row in rows),
            "pinning_episodes": sum(row["pinning"] for row in rows),
            "time_limit_episodes": sum(row["truncated"] for row in rows),
            "success_rate": successes / len(rows),
            "success_rate_95pct_wilson_interval": _success_interval(successes, len(rows)),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in rows])),
            "median_final_distance_m": float(np.median([row["final_distance_m"] for row in rows])),
            "mean_minimum_distance_m": float(np.mean([row["minimum_distance_m"] for row in rows])),
            "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_final_hold_steps": float(np.mean([row["final_hold_steps"] for row in rows])),
            "mean_maximum_hold_steps": float(np.mean([row["maximum_hold_steps"] for row in rows])),
            "mean_maximum_lift_clearance_m": float(np.mean([row["maximum_lift_clearance_m"] for row in rows])),
        }
    complete = len(rows) == requested_episodes
    return {
        "policy": policy,
        "summary": summary,
        "episodes": rows,
        "complete": complete,
        "requested_episodes": requested_episodes,
        "completed_episodes": len(rows),
        "stopped_reason": None if complete else "wall_time_budget",
        "partial_episode": partial_episode,
        "control_steps": control_steps,
    }


def run_episodes(
    env: SO101GraspEnv,
    seeds: Iterable[int],
    *,
    model: PPO | None = None,
    policy: str = "trained",
    render: bool = False,
    progress: Callable[[int, int], None] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Roll out a policy; all baselines receive identical seeded initial conditions.

    Progress receives completed episodes and observed control steps, including
    steps from the current episode, without collecting additional transitions.
    An optional absolute ``time.monotonic()`` deadline stops between operations;
    only naturally completed episodes contribute to summary metrics.
    """
    if policy not in ("trained", "no_motion", "random"):
        raise ValueError(f"Unknown evaluation policy: {policy}")
    if policy == "trained" and model is None:
        raise ValueError("The trained policy requires a loaded PPO model.")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("At least one evaluation episode is required.")
    if _deadline_expired(deadline):
        return _episode_results(policy, len(seeds), [])
    if render:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(env.model, env.data)
    else:
        viewer_context = nullcontext(None)
    rows = []
    control_steps = 0
    partial_episode = None
    with viewer_context as viewer:
        if progress is not None:
            progress(0, 0)
        viewer_state = None
        if viewer is not None:
            viewer_state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
            with viewer.lock():
                env.configure_camera(viewer.cam)
        for episode_seed in seeds:
            if _deadline_expired(deadline):
                break
            with viewer.lock() if viewer is not None else nullcontext():
                observation, info = env.reset(seed=int(episode_seed))
                stationary_action = np.asarray(env.neutral_action(), dtype=np.float32).copy()
            generator = np.random.default_rng(int(episode_seed) + 1_000_000)
            episode_return = 0.0
            minimum_distance = float(info["distance"])
            initial_distance = minimum_distance
            maximum_hold = 0
            maximum_clearance = 0.0
            maximum_downward_force = 0.0
            done = False
            steps = 0
            terminated = truncated = False
            while not done:
                if _deadline_expired(deadline):
                    break
                step_start = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    raise RuntimeError("Playback window closed before evaluation completed.")
                if policy == "trained":
                    action, _ = model.predict(observation, deterministic=True)
                elif policy == "no_motion":
                    action = stationary_action
                else:
                    action = generator.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                if _deadline_expired(deadline):
                    break
                with viewer.lock() if viewer is not None else nullcontext():
                    observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                episode_return += float(reward)
                minimum_distance = min(minimum_distance, float(info["distance"]))
                maximum_hold = max(maximum_hold, int(info["hold_steps"]))
                maximum_clearance = max(maximum_clearance, float(info.get("lift_clearance_m", 0.0)))
                maximum_downward_force = max(maximum_downward_force, float(info.get("peak_downward_robot_force_n", 0.0)))
                done = bool(terminated or truncated)
                control_steps += 1
                if progress is not None:
                    progress(len(rows), control_steps)
                if viewer is not None:
                    _sync_playback_view(viewer, env.model, env.data, viewer_state)
                    time.sleep(max(0.0, env.dt - (time.perf_counter() - step_start)))
            row = {
                "seed": int(episode_seed),
                "return": episode_return,
                "steps": steps,
                "initial_distance_m": initial_distance,
                "final_distance_m": float(info["distance"]),
                "minimum_distance_m": minimum_distance,
                "final_hold_steps": int(info.get("hold_steps", 0)),
                "maximum_hold_steps": maximum_hold,
                "is_success": bool(info.get("is_success", False)),
                "collision": bool(info.get("collision", False)),
                "pinning": bool(info.get("pinning", False)),
                "maximum_lift_clearance_m": maximum_clearance,
                "maximum_downward_robot_force_n": maximum_downward_force,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason") or ("success" if info.get("is_success", False) else "collision" if info.get("collision", False) else "time_limit" if truncated else "other_failure"),
            }
            if not done:
                row.update(
                    is_success=False, terminated=False, truncated=False,
                    termination_reason="wall_time_budget",
                )
                partial_episode = row
                break
            rows.append(row)
            if progress is not None:
                progress(len(rows), control_steps)
    return _episode_results(
        policy, len(seeds), rows, partial_episode=partial_episode,
        control_steps=control_steps,
    )


class _EvaluationProgress:
    """Report bounded progress, including heartbeats within long episodes."""

    def __init__(self, policy: str, episodes: int, completed_before: int, total: int, started: float):
        self.policy = policy
        self.episodes = episodes
        self.completed_before = completed_before
        self.total = total
        self.started = started
        self.next_report = started

    def __call__(self, completed: int, control_steps: int) -> None:
        now = time.perf_counter()
        if control_steps != 0 and completed != self.episodes and now < self.next_report:
            return
        print(
            f"Evaluation {self.policy}: {completed}/{self.episodes} episodes, "
            f"{control_steps:,} control steps; "
            f"overall {self.completed_before + completed}/{self.total} episodes; "
            f"elapsed {now - self.started:.1f}s",
            flush=True,
        )
        self.next_report = now + 30.0


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    episodes: int = 100,
    seed: int = 20_000,
    render: bool = False,
    policies: tuple[str, ...] = ("trained", "no_motion", "random"),
    deadline: float | None = None,
) -> dict[str, Any]:
    """Evaluate selected policies, sharing an optional absolute monotonic deadline.

    The unbounded default remains a complete three-policy held-out evaluation.
    Budget exhaustion leaves unavailable summaries null, including when it
    happens before environment construction or checkpoint validation.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("Evaluation policies must be nonempty and unique.")
    if any(policy not in ("trained", "no_motion", "random") for policy in policies):
        raise ValueError("Unknown evaluation policy.")
    seeds = list(range(seed, seed + episodes))
    results = {policy: _episode_results(policy, episodes, []) for policy in policies}
    metadata = None
    env = None
    try:
        if not _deadline_expired(deadline):
            torch.set_num_threads(1)
            env = SO101GraspEnv()
        if env is not None and not _deadline_expired(deadline):
            print(f"Loading evaluation checkpoint: {checkpoint}", flush=True)
            model, metadata = load_checked_policy(checkpoint, env)
            total_episodes = len(policies) * episodes
            print(
                f"Checkpoint evaluation: {episodes} episodes per policy x {len(policies)} policies "
                f"= {total_episodes} episodes; at most {total_episodes * env.max_steps:,} control steps. "
                "Progress is reported every 30 seconds and at each policy's completion.",
                flush=True,
            )
            started = time.perf_counter()
            completed_before = 0
            for policy in policies:
                if _deadline_expired(deadline):
                    break
                results[policy] = run_episodes(
                    env, seeds, model=model if policy == "trained" else None,
                    policy=policy, render=render and policy == "trained",
                    progress=_EvaluationProgress(policy, episodes, completed_before, total_episodes, started),
                    deadline=deadline,
                )
                completed_before += results[policy]["completed_episodes"]
        complete = all(result["complete"] for result in results.values())
        return {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": metadata["checkpoint_sha256"] if metadata is not None else None,
            "environment": metadata["environment"] if metadata is not None else None,
            "checkpoint_validation_completed": metadata is not None,
            "package_versions": package_versions(),
            "evaluation_seed_start": seed,
            "evaluation_seed_end": seed + episodes - 1,
            "deterministic_trained_policy": True,
            "rendered_policy_playback": render,
            "viewer_physics_edits_ignored": render,
            "success_definition": "Both jaw pads carry force; ball bottom is at least 20mm above the tabletop without support; world/relative motion is slow for 10 consecutive control steps, with no failure.",
            "policies_requested": list(policies),
            "policies_completed": [policy for policy, result in results.items() if result["complete"]],
            "complete": complete,
            "requested_episodes": episodes * len(policies),
            "completed_episodes": sum(result["completed_episodes"] for result in results.values()),
            "control_steps": sum(result["control_steps"] for result in results.values()),
            "stopped_reason": None if complete else "wall_time_budget",
            "results": results,
        }
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to inspect (default: latest managed experiment, with legacy local-artifacts/runs/current fallback).")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--output", type=Path, help="Optional new report file; existing files are rejected. Default: a unique evaluation directory next to the checkpoint.")
    parser.add_argument("--render", action="store_true", help="Play trained-policy episodes at real time in MuJoCo.")
    args = parser.parse_args()
    try:
        with evaluation_output(args.output, checkpoint=args.checkpoint, seed=args.seed) as output:
            checkpoint = (args.checkpoint or current_checkpoint()).resolve()
            if checkpoint.suffix != ".zip":
                checkpoint = checkpoint.with_suffix(".zip")
            report = evaluate_checkpoint(checkpoint, episodes=args.episodes, seed=args.seed, render=args.render)
            report["checkpoint"] = str(checkpoint)
            write_json(output, report)
    except OutputError as exc:
        parser.error(str(exc))
    for policy, result in report["results"].items():
        summary = result["summary"]
        print(f"{policy}: success={summary['successes']}/{summary['episodes']} "
              f"({summary['success_rate']:.1%}), final distance={summary['mean_final_distance_m'] * 1000:.1f} mm, "
              f"return={summary['mean_return']:.2f}")
    print(f"Evaluation report: {output.resolve()}")


if __name__ == "__main__":
    main()

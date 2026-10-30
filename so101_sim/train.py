"""Train PPO continuously, with optional step limits or a first-update startup trial."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import HumanOutputFormat, Logger, configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from .grasp import SO101GraspEnv
from .diagnostics import RolloutControlDiagnostics
from .evaluate import environment_signature, evaluate_checkpoint, run_episodes, save_checkpoint, write_json
from .learning import DiagnosticPPO as PPO, RewardScale
from .lineage import compare_parent_validation, load_parent_validation
from .outputs import OutputError, current_checkpoint, training_output
from .resume import ResumeError, inspect_resume_checkpoint, load_resume_policy
from .vector_env import ManagedSubprocVecEnv as SubprocVecEnv


DEFAULT_TRIAL_TIMESTEPS = 500
LEARNING_DEFAULTS = {
    "log_std_init": -2.0, "learning_rate": 0.0003, "n_epochs": 10,
    "target_kl": None, "use_sde": False, "sde_sample_freq": -1,
}


def _configure_training_logger(output: Path) -> Logger:
    logger = configure(str(output), ["stdout", "csv"])
    for output_format in logger.output_formats:
        if isinstance(output_format, HumanOutputFormat):
            # SB3's default 36-character display truncates distinct diagnostic
            # names to the same key and raises before the CSV writer can run.
            output_format.max_length = 128
    return logger


def _inherit_resume_settings(args, checkpoint, provided: set[str]) -> None:
    for key, default in LEARNING_DEFAULTS.items():
        saved = checkpoint.training.get(key, default)
        if key in provided and getattr(args, key) != saved:
            raise ResumeError(f"Resume must preserve {key}={saved!r}; omit the conflicting option or start a fresh experiment.")
        setattr(args, key, saved)
    saved_scale = checkpoint.training["reward_scale"]
    if args.reward_scale != saved_scale:
        raise ResumeError(
            f"Reward scale mismatch: checkpoint uses {saved_scale}, requested {args.reward_scale}. "
            f"Use --reward-scale {saved_scale} to retain its critic/optimizer units; conversion is not supported."
        )
    fixed = {
        "n_steps": 256, "batch_size": 256, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "policy_net_arch": [64, 64], "observation_normalization": False,
        "reward_normalization": False,
    }
    differences = [key for key, value in fixed.items() if checkpoint.training.get(key) != value]
    if differences:
        raise ResumeError("Checkpoint training settings are unsupported by this trainer: " + ", ".join(differences))


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


class _OptimizerStepCounter:
    """Count completed optimizer calls and reject an unauthorized trial update."""

    def __init__(self, optimizer: Any, *, limit: int | None = None):
        self.optimizer = optimizer
        self.limit = limit
        self.count = 0
        self.original_step = optimizer.step

    def step(self, *args: Any, **kwargs: Any) -> Any:
        if self.limit is not None and self.count >= self.limit:
            raise RuntimeError("Startup trial attempted a second optimizer step; only the first network update is allowed.")
        result = self.original_step(*args, **kwargs)
        self.count += 1
        return result

    def __enter__(self) -> _OptimizerStepCounter:
        self.optimizer.step = self.step
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.optimizer.step = self.original_step


class _TrainingStepLimit(BaseCallback):
    """Stop at an exact transition budget while allowing complete rollout updates."""

    def __init__(self, limit: int, rollout_size: int, *, starting_timesteps: int = 0):
        super().__init__()
        self.limit = starting_timesteps + limit
        self.session_limit = limit
        self.rollout_size = rollout_size

    def _on_step(self) -> bool:
        if self.num_timesteps < self.limit:
            return True
        if self.num_timesteps > self.limit:
            raise RuntimeError("Training exceeded its environment step limit.")
        # SB3 checks callbacks before adding the latest transition to its buffer.
        # A complete final rollout must return True so PPO can optimize it;
        # otherwise discard the incomplete rollout and stop without overshooting.
        return self.session_limit % self.rollout_size == 0


class FixedSeedEvaluation(BaseCallback):
    """Select checkpoints using validation goals, independent from final held-out goals."""

    def __init__(
        self,
        output: Path,
        signature: dict[str, Any],
        training_config: dict[str, Any],
        *,
        eval_frequency: int = 100_000,
        eval_episodes: int = 20,
        start_time: float,
        progress=lambda: {},
        starting_timesteps: int = 0,
        parent_validation: dict[str, Any] | None = None,
        validation_time_budget: float | None = None,
        final_time_budget: float | None = None,
    ):
        super().__init__()
        self.output = output
        self.signature = signature
        self.training_config = training_config
        self.eval_frequency = eval_frequency
        self.eval_seeds = list(range(10_000, 10_000 + eval_episodes))
        self.starting_timesteps = starting_timesteps
        self.next_evaluation = starting_timesteps + eval_frequency
        self.start_time = start_time
        self.progress = progress
        self.validation_time_budget = validation_time_budget
        self.final_time_budget = final_time_budget
        self.final_deadline = None
        self.validation_reports = 0
        self.validation_partial_steps = 0
        self.validation_seconds = 0.0
        self.validation_episodes = 0
        self.validation_steps = 0
        self.rollout_distances = []
        self.rollout_max_hold = 0
        self.grasp_samples = 0
        self.grasp_bilateral_samples = 0
        self.grasp_opposed_samples = 0
        self.grasp_lifted_samples = 0
        self.grasp_max_clearance = 0.0
        self.grasp_max_downward_force = 0.0
        self.grasp_pinning_failures = 0
        self.best_score: tuple[float, float, float] | None = None
        self.best_validation_summary: dict[str, Any] | None = None
        self.best_validation_timesteps: int | None = None
        self.best_checkpoint_sha256: str | None = None
        self.parent_validation = parent_validation
        self.control_diagnostics = RolloutControlDiagnostics(signature)
        self.control_metric_keys: set[str] = set()
        self.eval_env = SO101GraspEnv()
        self.last_evaluation = -1

    def training_metadata(self) -> dict[str, Any]:
        metadata = {
            **self.training_config,
            **self.progress(),
            "actual_timesteps": self.num_timesteps - self.starting_timesteps,
            "cumulative_timesteps": self.num_timesteps,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "validation_seeds": self.eval_seeds,
        }
        if self.parent_validation is not None:
            comparison = compare_parent_validation(self.parent_validation, self.best_validation_summary)
            metadata["parent_validation_comparison"] = {
                "file": "parent_comparison.json", "status": comparison["status"],
                "recommendation": comparison["recommendation"],
                "scope": "session_best_vs_selected_parent_validation",
            }
        return metadata

    def _publish_parent_comparison(self) -> None:
        if self.parent_validation is None:
            return
        comparison = compare_parent_validation(self.parent_validation, self.best_validation_summary)
        checkpoint = self.output / "best_model.zip"
        comparison["child_best"] = None if self.best_validation_summary is None else {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": self.best_checkpoint_sha256,
            "session_timesteps": self.best_validation_timesteps,
            "cumulative_timesteps": self.starting_timesteps + self.best_validation_timesteps,
            "validation_seeds": self.eval_seeds,
        }
        write_json(self.output / "parent_comparison.json", comparison)
        print(f"Session-best validation versus selected parent: {comparison['status']} "
              f"(recommendation: {comparison['recommendation']}).", flush=True)

    def evaluate(self, *, deadline: float | None = None) -> dict[str, Any]:
        if self.validation_time_budget is not None:
            budget = self.validation_time_budget
            if deadline is not None and self.final_time_budget is not None:
                # Allow the final fixed batch more time to finish while keeping
                # room for held-out work inside the shared completion budget.
                budget = min(120.0, self.final_time_budget)
            own_deadline = time.monotonic() + budget
            deadline = own_deadline if deadline is None else min(deadline, own_deadline)
        started = time.perf_counter()
        try:
            options = {} if deadline is None else {"deadline": deadline}
            report = run_episodes(self.eval_env, self.eval_seeds, model=self.model, **options)
        finally:
            evaluation_seconds = time.perf_counter() - started
            self.validation_seconds += evaluation_seconds
        self.validation_episodes += len(report["episodes"])
        self.validation_steps += sum(episode["steps"] for episode in report["episodes"])
        partial_steps = int((report.get("partial_episode") or {}).get("steps", 0))
        self.validation_partial_steps += partial_steps
        self.validation_reports += 1
        complete = bool(report.get("complete", True))
        summary = report.get("summary") or {}
        # Keep every observed partial batch, but only equal-size complete batches
        # may select a checkpoint or enter the exact-parent comparison.
        write_json(self.output / f"validation_{self.num_timesteps}_{self.validation_reports}.json", report)
        summary_keys = ("episodes", "successes", "collision_episodes", "pinning_episodes",
                        "time_limit_episodes", "success_rate", "mean_return", "mean_final_distance_m",
                        "median_final_distance_m", "mean_minimum_distance_m", "mean_episode_steps",
                        "mean_final_hold_steps", "mean_maximum_hold_steps", "mean_maximum_lift_clearance_m")
        row = {
            "timesteps": self.num_timesteps - self.starting_timesteps,
            "cumulative_timesteps": self.num_timesteps,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "evaluation_seconds": evaluation_seconds,
            "validation_seconds": self.validation_seconds,
            "validation_control_steps": self.validation_steps,
            "complete": complete, "requested_episodes": len(self.eval_seeds),
            "completed_episodes": len(report["episodes"]),
            "stopped_reason": report.get("stopped_reason"), "partial_control_steps": partial_steps,
            **{key: summary.get(key) for key in summary_keys},
        }
        csv_path = self.output / "evaluations.csv"
        needs_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        eligible = complete and summary and summary.get("episodes") == len(self.eval_seeds)
        if eligible:
            score = (summary["success_rate"], -summary["mean_final_distance_m"], summary["mean_return"])
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_validation_summary = dict(summary)
                self.best_validation_timesteps = self.num_timesteps - self.starting_timesteps
                save_checkpoint(self.model, self.output / "best_model.zip", self.signature, self.training_metadata())
                checkpoint = self.output / "best_model.zip"
                self.best_checkpoint_sha256 = (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                               if checkpoint.is_file() else None)
                write_json(self.output / "best_validation.json", {
                    "timesteps": self.num_timesteps - self.starting_timesteps,
                    "cumulative_timesteps": self.num_timesteps, **report,
                    **({"checkpoint_sha256": self.best_checkpoint_sha256}
                       if self.best_checkpoint_sha256 is not None else {}),
                })
            self._publish_parent_comparison()
        self.last_evaluation = self.num_timesteps
        for name in ("success_rate", "mean_final_distance_m", "mean_return",
                     "mean_maximum_lift_clearance_m", "mean_maximum_hold_steps"):
            if name not in summary:
                for attribute in ("name_to_value", "name_to_count", "name_to_excluded"):
                    pending = getattr(self.logger, attribute, None)
                    if isinstance(pending, dict):
                        pending.pop(f"validation/{name}", None)
            if name in summary:
                self.logger.record(f"validation/{name}", summary[name])
        self.logger.record("validation/complete", int(complete))
        self.logger.record("time/validation_seconds", self.validation_seconds)
        self.logger.record("validation/control_steps", self.validation_steps)
        self.logger.record("validation/partial_control_steps", self.validation_partial_steps)
        outcome = (f"{summary['successes']}/{summary['episodes']} successes, "
                   f"mean final distance {summary['mean_final_distance_m'] * 1000:.1f} mm") if summary else "no completed episodes"
        print(f"Validation at {self.num_timesteps - self.starting_timesteps:,} session steps: "
              f"{len(report['episodes'])}/{len(self.eval_seeds)} episodes, "
              f"{'complete' if complete else 'time budget reached; not ranked'}; {outcome}", flush=True)
        return report

    def _on_step(self) -> bool:
        self.control_diagnostics.observe(
            getattr(self.model, "_last_obs", None), self.locals.get("actions"),
            self.locals.get("clipped_actions"), self.locals.get("new_obs"),
            self.locals.get("dones"), self.locals.get("infos"),
        )
        for info in self.locals.get("infos", []):
            distance = info.get("distance")
            if distance is not None and math.isfinite(distance):
                self.rollout_distances.append(float(distance))
            self.rollout_max_hold = max(self.rollout_max_hold, int(info.get("hold_steps", 0)))
            if "fixed_jaw_contact" in info and "moving_jaw_contact" in info:
                self.grasp_samples += 1
                self.grasp_bilateral_samples += int(info["fixed_jaw_contact"] and info["moving_jaw_contact"])
                self.grasp_opposed_samples += int(info.get("opposed_jaw_contacts", False))
                clearance = float(info.get("lift_clearance_m", 0.0))
                self.grasp_lifted_samples += int(clearance >= 0.02 and not info.get("ball_supported", True))
                self.grasp_max_clearance = max(self.grasp_max_clearance, clearance)
                self.grasp_max_downward_force = max(
                    self.grasp_max_downward_force, float(info.get("peak_downward_robot_force_n", 0.0)),
                )
                self.grasp_pinning_failures += int(info.get("pinning", False))
        if self.num_timesteps >= self.next_evaluation:
            limit = self.training_config.get("environment_step_limit")
            if (self.final_time_budget is not None and limit is not None
                    and self.num_timesteps - self.starting_timesteps >= limit
                    and self.final_deadline is None):
                self.final_deadline = time.monotonic() + self.final_time_budget
            self.evaluate(deadline=self.final_deadline)
            self.next_evaluation = self.num_timesteps + self.eval_frequency
        return True

    def _on_rollout_start(self) -> None:
        self.rollout_distances = []
        self.rollout_max_hold = 0
        self.grasp_samples = self.grasp_bilateral_samples = self.grasp_lifted_samples = 0
        self.grasp_opposed_samples = 0
        self.grasp_max_clearance = self.grasp_max_downward_force = 0.0
        self.grasp_pinning_failures = 0
        self.control_diagnostics.reset()

    def _on_rollout_end(self) -> None:
        # SB3 dumps every few rollouts. Omitted conditional metrics must not
        # retain an earlier rollout's near-goal values in its pending logger.
        for attribute in ("name_to_value", "name_to_count", "name_to_excluded"):
            pending = getattr(self.logger, attribute, None)
            if isinstance(pending, dict):
                for name in self.control_metric_keys:
                    pending.pop(name, None)
        control_metrics = self.control_diagnostics.metrics()
        self.control_metric_keys = set(control_metrics)
        for name, value in control_metrics.items():
            self.logger.record(name, value)
        self.logger.record("time/session_timesteps", self.num_timesteps - self.starting_timesteps)
        self.logger.record("time/cumulative_timesteps", self.num_timesteps)
        if self.rollout_distances:
            count = len(self.rollout_distances)
            self.logger.record("rollout/mean_distance_m", sum(self.rollout_distances) / count)
            self.logger.record("rollout/minimum_distance_m", min(self.rollout_distances))
            for threshold in (15, 40, 80):
                fraction = sum(d < threshold / 1000 for d in self.rollout_distances) / count
                self.logger.record(f"rollout/within_{threshold}mm_fraction", fraction)
        self.logger.record("rollout/maximum_hold_steps", self.rollout_max_hold)
        if self.grasp_samples:
            self.logger.record("grasp/observed_samples", self.grasp_samples)
            self.logger.record("grasp/bilateral_contact_fraction", self.grasp_bilateral_samples / self.grasp_samples)
            self.logger.record("grasp/opposed_contact_fraction", self.grasp_opposed_samples / self.grasp_samples)
            self.logger.record("grasp/unsupported_lift_fraction", self.grasp_lifted_samples / self.grasp_samples)
            self.logger.record("grasp/maximum_clearance_m", self.grasp_max_clearance)
            self.logger.record("grasp/maximum_downward_robot_force_n", self.grasp_max_downward_force)
            self.logger.record("grasp/pinning_failures", self.grasp_pinning_failures)

    def _on_training_end(self) -> None:
        if self.final_time_budget is not None and self.final_deadline is None:
            self.final_deadline = time.monotonic() + self.final_time_budget
        try:
            if self.last_evaluation != self.num_timesteps:
                self.evaluate(deadline=self.final_deadline)
        finally:
            self.eval_env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--trial", action="store_true",
        help="Run only through the first optimizer step (default rollout: 500 environment steps).",
    )
    mode.add_argument(
        "--full-training", action="store_true",
        help="Compatibility alias for the default full training mode.",
    )
    parser.add_argument(
        "--timesteps", type=int,
        help="New training transitions in this session (default: unlimited); with --resume these are additional steps. With --trial: rollout budget before the first network update (at least 2; default: 500).",
    )
    parser.add_argument("--output", type=Path, help="Optional custom empty directory. Default: a new experiment directory under local-artifacts/runs/experiments for every invocation.")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="Continue a full-training checkpoint, or 'latest'. --timesteps counts additional transitions; the default remains a fresh policy.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", "--render", action="store_true", help="Watch actual training in a live MuJoCo viewer (environment 0).")
    parser.add_argument("--viewer-speed", type=_finite_float, default=1.0, help="Viewer simulation speed multiplier (default: 1, real time).")
    parser.add_argument("--envs", type=int, help="Parallel environments (startup trial: 1 only; full training default: 8).")
    parser.add_argument("--eval-frequency", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, help="Validation episodes per round (quick: 5; full: 20).")
    parser.add_argument("--heldout-episodes", type=int, help="Held-out episodes per selected policy (quick: 5; full: 100).")
    parser.add_argument("--evaluation-profile", choices=("quick", "full"), default="quick", help="Quick development checks (default), or full evaluation with both baselines.")
    parser.add_argument("--eval-time-budget", type=_finite_float, help="Shared wall-clock seconds for final validation and held-out (quick: 300; full: uncapped).")
    parser.add_argument("--log-std-init", type=_finite_float, default=argparse.SUPPRESS, help="Finite initial Gaussian policy log standard deviation (fresh default: -2; resume: inherited).")
    parser.add_argument("--reward-scale", type=_finite_float, default=0.01, help="Positive training-only reward multiplier; Monitor and evaluations use raw rewards (default: 0.01).")
    parser.add_argument("--learning-rate", type=_finite_float, default=argparse.SUPPRESS, help="Fresh default: 0.0003; resume inherits the checkpoint value.")
    parser.add_argument("--n-epochs", type=int, default=argparse.SUPPRESS, help="PPO optimization epochs (fresh default: 10; resume: inherited); trials always use one.")
    parser.add_argument("--target-kl", type=_finite_float, default=argparse.SUPPRESS, help="Optional positive PPO KL stopping threshold (fresh default: disabled; resume: inherited).")
    parser.add_argument("--use-sde", action="store_true", default=argparse.SUPPRESS, help="Use state-dependent exploration (fresh default: off; resume: inherited).")
    parser.add_argument("--sde-sample-freq", type=int, default=argparse.SUPPRESS, help="Resample gSDE noise every N control steps, or -1 once per rollout (resume: inherited).")
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy", help="Environment backend; subproc is opt-in for headless full training.")
    args = parser.parse_args(argv)
    quick = args.evaluation_profile == "quick"
    args.eval_episodes = args.eval_episodes if args.eval_episodes is not None else (5 if quick else 20)
    args.heldout_episodes = args.heldout_episodes if args.heldout_episodes is not None else (5 if quick else 100)
    args.eval_time_budget = args.eval_time_budget if args.eval_time_budget is not None else (300.0 if quick else None)
    if args.eval_time_budget is not None and args.eval_time_budget <= 0:
        parser.error("eval-time-budget must be positive")
    args.validation_time_budget = min(60.0, args.eval_time_budget) if quick else None
    provided_learning = {key for key in LEARNING_DEFAULTS if hasattr(args, key)}
    for key, default in LEARNING_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    args.resume_checkpoint = None
    args.resume_signature = None
    trial = args.trial
    args.full_training = not trial
    if trial and args.timesteps is None:
        args.timesteps = DEFAULT_TRIAL_TIMESTEPS
    explicit_envs = args.envs is not None
    args.envs = args.envs if args.envs is not None else (1 if trial else 8)
    if args.viewer_speed <= 0:
        parser.error("viewer-speed must be positive")
    if trial and args.resume is not None:
        parser.error("--resume is unavailable with --trial; startup trials always begin with a fresh policy")
    if args.reward_scale <= 0 or args.learning_rate <= 0 or args.n_epochs < 1:
        parser.error("reward-scale, learning-rate and n-epochs must be positive")
    if args.target_kl is not None and args.target_kl <= 0:
        parser.error("target-kl must be positive")
    if args.sde_sample_freq != -1 and args.sde_sample_freq < 1:
        parser.error("sde-sample-freq must be -1 or a positive number of control steps")
    if args.vec_env == "subproc" and (trial or args.viewer):
        parser.error("subproc supports headless full training only; use dummy for --trial or --viewer")
    if trial and args.target_kl is not None:
        parser.error("target-kl is unavailable in a startup trial, which must complete exactly one optimizer step")
    if (args.timesteps is not None and args.timesteps < 1) or min(args.envs, args.eval_frequency, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("timesteps, envs and evaluation counts must be positive")
    if trial and args.timesteps < 2:
        parser.error("A startup trial requires at least 2 environment steps before its first network update.")
    if trial and args.envs != 1:
        parser.error("A startup trial uses exactly one environment so its total step budget cannot be multiplied by --envs.")
    if not trial and args.timesteps is not None and args.timesteps % args.envs:
        if explicit_envs:
            parser.error("timesteps must be divisible by envs to stop at the exact total step limit.")
        args.envs = 1
        print("Using one environment to respect the exact --timesteps limit.", flush=True)
    try:
        if args.resume is not None:
            # Resolve 'latest' before creating the new run replaces its pointer.
            checkpoint_path = current_checkpoint() if args.resume == "latest" else Path(args.resume)
            probe = SO101GraspEnv()
            try:
                args.resume_signature = environment_signature(probe)
                args.resume_checkpoint = inspect_resume_checkpoint(checkpoint_path, args.resume_signature)
            finally:
                probe.close()
            _inherit_resume_settings(args, args.resume_checkpoint, provided_learning)
        if not args.use_sde and args.sde_sample_freq != -1:
            parser.error("sde-sample-freq requires --use-sde")
        with training_output(
            args.output, mode="startup_trial" if trial else "full_training", seed=args.seed,
        ) as output:
            _run(args, output)
    except (OutputError, ResumeError) as exc:
        parser.error(str(exc))


def _run(args, output: Path) -> None:
    trial = not args.full_training
    resume = args.resume_checkpoint
    starting = resume.counters if resume is not None else {
        "timesteps": 0, "optimizer_steps": 0, "update_count": 0, "optimization_epochs": 0,
    }
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "algorithm": "PPO",
        "mode": "startup_trial" if trial else "full_training",
        "requested_timesteps": args.timesteps,
        "requested_timesteps_unit": "new training transitions in this session",
        "environment_step_limit": args.timesteps,
        "cumulative_environment_step_limit": (starting["timesteps"] + args.timesteps
                                               if args.timesteps is not None else None),
        "continuous": not trial and args.timesteps is None,
        "stop_criterion": "first_optimizer_step" if trial else (
            "manual_interrupt" if args.timesteps is None else "total_environment_steps"
        ),
        "selected_rollout_budget": args.timesteps if trial else None,
        "trial_stop_criterion": "first_optimizer_step" if trial else None,
        "max_gradient_updates": 1 if trial else None,
        "seed": args.seed,
        "num_envs": args.envs,
        "device": "cpu",
        "torch_threads": 1,
        # A startup trial collects one task-specific rollout and optimizes its
        # entire batch once, so the first optimizer call ends the trial.
        "n_steps": args.timesteps if trial else 256,
        "batch_size": args.timesteps if trial else 256,
        "n_epochs": 1 if trial else args.n_epochs,
        "learning_rate": args.learning_rate,
        "target_kl": args.target_kl,
        "use_sde": args.use_sde,
        "sde_sample_freq": args.sde_sample_freq,
        "vec_env_backend": args.vec_env,
        "reward_scale": args.reward_scale,
        "reward_scale_scope": "training only; Monitor and evaluation report raw environment rewards",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_net_arch": [64, 64],
        "log_std_init": args.log_std_init,
        "observation_normalization": False,
        "reward_normalization": False,
        "evaluation_frequency": None if trial else args.eval_frequency,
        "validation_episodes": 0 if trial else args.eval_episodes,
        "heldout_episodes": 0 if trial else args.heldout_episodes,
        "heldout_seed_start": None if trial else 20_000,
        "evaluation_profile": args.evaluation_profile,
        "evaluation_time_budget_seconds": args.eval_time_budget,
        "validation_time_budget_seconds": args.validation_time_budget,
        "final_validation_time_budget_seconds": (min(120.0, args.eval_time_budget)
                                                  if args.evaluation_profile == "quick" else None),
        "heldout_policies": [] if trial else (["trained"] if args.evaluation_profile == "quick" else ["trained", "no_motion", "random"]),
        "checkpoint_loaded": False,
        "parent_checkpoint": str(resume.path) if resume is not None else None,
        "parent_checkpoint_sha256": resume.metadata["checkpoint_sha256"] if resume is not None else None,
        "parent_run_id": resume.path.parent.name if resume is not None else None,
        **{f"starting_{key}": value for key, value in starting.items()},
        "live_viewer": args.viewer,
        "viewer_speed": args.viewer_speed if args.viewer else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if resume is not None:
        signature = args.resume_signature
    else:
        probe = SO101GraspEnv()
        try:
            signature = environment_signature(probe)
        finally:
            probe.close()
    parent_validation = None
    if resume is not None:
        parent_validation = load_parent_validation(
            resume.path, resume.metadata, list(range(10_000, 10_000 + args.eval_episodes)),
        )
        config["parent_comparison_file"] = "parent_comparison.json"
        write_json(output / "parent_comparison.json", compare_parent_validation(parent_validation, None))
    write_json(
        output / "training.json",
        {"status": "trial_running" if trial else "running", "mode": config["mode"],
         "config": config, "environment": signature, "actual_timesteps": 0,
         "update_count": 0, "optimizer_steps": 0, "gradient_update_count": 0,
         **{f"cumulative_{key}": value for key, value in starting.items()}},
    )

    def make_env(index: int):
        def factory():
            monitored = Monitor(
                SO101GraspEnv(),
                filename=str(output / f"monitor_{index}.csv"),
                info_keywords=("is_success", "distance", "hold_steps"),
            )
            return RewardScale(monitored, config["reward_scale"])
        return factory

    factories = [make_env(index) for index in range(args.envs)]
    vector_env = None
    start_time = time.perf_counter()
    callback = None
    evaluation_callback = None
    model = None
    optimizer_counter = None
    view_callback = None
    heldout_seconds = 0.0

    def progress() -> dict[str, Any]:
        epochs = int(model._n_updates) if model is not None else starting["optimization_epochs"]
        timesteps = int(model.num_timesteps) if model is not None else starting["timesteps"]
        updates = int(model.rollout_update_count) if model is not None else starting["update_count"]
        optimizer_steps = optimizer_counter.count if optimizer_counter is not None else 0
        return {
            "actual_timesteps": timesteps - starting["timesteps"],
            "cumulative_timesteps": timesteps,
            "starting_timesteps": starting["timesteps"],
            "session_environment_step_limit": args.timesteps,
            "update_count": updates - starting["update_count"],
            "cumulative_update_count": updates,
            "update_count_unit": "completed PPO rollout updates",
            "optimization_epochs": epochs - starting["optimization_epochs"],
            "cumulative_optimization_epochs": epochs,
            "optimization_epochs_unit": "SB3 epochs entered, including a KL-stopped partial epoch",
            "optimizer_steps": optimizer_steps,
            "cumulative_optimizer_steps": starting["optimizer_steps"] + optimizer_steps,
            "gradient_update_count": optimizer_steps,
            "gradient_update_count_unit": "completed optimizer.step calls",
            "phase_seconds": {
                "rollout_including_callbacks": float(model.rollout_seconds) if model is not None else 0.0,
                "optimization": float(model.optimization_seconds) if model is not None else 0.0,
                "diagnostics": float(model.diagnostics_seconds) if model is not None else 0.0,
                "validation": evaluation_callback.validation_seconds if evaluation_callback is not None else 0.0,
                "heldout": heldout_seconds,
            },
            "validation_episodes": evaluation_callback.validation_episodes if evaluation_callback is not None else 0,
            "validation_control_steps": evaluation_callback.validation_steps if evaluation_callback is not None else 0,
            "validation_partial_control_steps": evaluation_callback.validation_partial_steps if evaluation_callback is not None else 0,
            "validation_counts_scope": "completed episodes; partial episode steps recorded separately",
        }

    try:
        vector_env = (SubprocVecEnv(factories, start_method="spawn")
                      if args.vec_env == "subproc" else DummyVecEnv(factories))
        vector_env.seed(args.seed)
        if not trial:
            callback = FixedSeedEvaluation(
                output,
                signature,
                config,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                start_time=start_time,
                progress=progress,
                starting_timesteps=starting["timesteps"],
                parent_validation=parent_validation,
                validation_time_budget=args.validation_time_budget,
                final_time_budget=args.eval_time_budget,
            )
            evaluation_callback = callback
        if args.viewer:
            from .training_view import TrainingViewer

            view_callback = TrainingViewer(speed=args.viewer_speed, progress=progress)
            callback = CallbackList([view_callback, callback]) if callback is not None else view_callback
        if not trial and args.timesteps is not None:
            step_limit = _TrainingStepLimit(
                args.timesteps, config["n_steps"] * args.envs, starting_timesteps=starting["timesteps"],
            )
            callbacks = callback.callbacks if isinstance(callback, CallbackList) else [callback]
            callback = CallbackList([step_limit, *callbacks])
        if resume is not None:
            model = load_resume_policy(resume, vector_env, seed=args.seed)
            config["checkpoint_loaded"] = True
            write_json(output / "training.json", {
                "status": "running", "mode": config["mode"], "config": config,
                "environment": signature, **progress(),
            })
            print(f"Resumed {resume.path} at {starting['timesteps']:,} cumulative steps; "
                  "policy weights and optimizer restored. New environment states use the session seed.", flush=True)
        else:
            model = PPO(
                "MlpPolicy",
                vector_env,
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                max_grad_norm=config["max_grad_norm"],
                target_kl=config["target_kl"],
                use_sde=config["use_sde"],
                sde_sample_freq=config["sde_sample_freq"],
                policy_kwargs={"net_arch": config["policy_net_arch"], "log_std_init": config["log_std_init"]},
                device="cpu",
                seed=args.seed,
                verbose=1,
            )
        model.set_logger(_configure_training_logger(output))
        if not trial:
            target = "continuously" if args.timesteps is None else f"for {args.timesteps:,} new environment steps"
            print(f"Training {target}. Press Ctrl+C or close the viewer to stop and save final_model.zip.", flush=True)
        with _OptimizerStepCounter(model.policy.optimizer, limit=1 if trial else None) as optimizer_counter:
            model.learn(
                total_timesteps=math.inf if args.timesteps is None else args.timesteps,
                callback=callback,
                log_interval=1 if trial else 5,
                **({"reset_num_timesteps": False} if resume is not None else {}),
            )
        if trial:
            if optimizer_counter.count != 1:
                raise RuntimeError("Startup trial ended before its first optimizer step completed.")
            # No validation callback, playback, baselines or held-out episodes run
            # in this branch. These transitions are the entire startup trial.
            metadata = {
                **config, **progress(),
                "elapsed_seconds": time.perf_counter() - start_time,
                "trial_stop_reason": "first_optimizer_step_completed",
                "task_performance_evaluated": False,
            }
            save_checkpoint(model, output / "trial_model.zip", signature, metadata)
            write_json(
                output / "training.json",
                {
                    "status": "trial_completed",
                    "mode": config["mode"],
                    "trial_status": "first_network_update_completed",
                    "trial_stop_criterion": config["trial_stop_criterion"],
                    "trial_stop_reason": "first_optimizer_step_completed",
                    "selected_rollout_budget": config["selected_rollout_budget"],
                    "config": config,
                    "environment": signature,
                    **progress(),
                    "training_seconds": time.perf_counter() - start_time,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": "trial_model.zip",
                    "task_performance_evaluated": False,
                },
            )
            print(f"Startup trial finished: {model.num_timesteps} total environment steps, "
                  f"{optimizer_counter.count} network update (optimizer.step).")
            print(f"Task performance has not been evaluated. Startup trial files: {output}")
            return
        save_checkpoint(model, output / "final_model.zip", signature, evaluation_callback.training_metadata())
        training_seconds = time.perf_counter() - start_time
        best_available = evaluation_callback.best_validation_summary is not None
        heldout_checkpoint = "best_model.zip" if best_available else "final_model.zip"
        final_deadline = evaluation_callback.final_deadline
        if not isinstance(final_deadline, (int, float)):
            final_deadline = time.monotonic() + args.eval_time_budget if args.eval_time_budget is not None else None
        write_json(output / "training.json", {
            "status": "evaluating", "mode": config["mode"], "config": config,
            "environment": signature, **progress(), "training_seconds": training_seconds,
            "heldout_checkpoint": heldout_checkpoint, "task_performance_evaluated": False,
        })
        print(f"Training finished. Final checkpoint saved to {output / 'final_model.zip'}.", flush=True)
        print(f"Starting held-out evaluation: {len(config['heldout_policies'])} policies x {args.heldout_episodes:,} episodes "
              f"({len(config['heldout_policies']) * args.heldout_episodes:,} total), {args.evaluation_profile} profile. "
              "Press Ctrl+C to stop evaluation; saved checkpoints are retained.", flush=True)
        heldout_started = time.perf_counter()
        try:
            evaluation_options = {}
            if args.evaluation_profile == "quick":
                evaluation_options["policies"] = ("trained",)
            if final_deadline is not None:
                evaluation_options["deadline"] = final_deadline
            report = evaluate_checkpoint(output / heldout_checkpoint, episodes=args.heldout_episodes,
                                         seed=20_000, **evaluation_options)
        finally:
            heldout_seconds = time.perf_counter() - heldout_started
        report["checkpoint"] = heldout_checkpoint
        report["evaluation_profile"] = args.evaluation_profile
        report["evidence_scope"] = "development diagnostic" if args.evaluation_profile == "quick" else "full held-out evaluation"
        report["checkpoint_selection"] = "validation_best" if best_available else "final_without_complete_validation"
        write_json(output / "evaluation.json", report)
        write_json(
            output / "training.json",
            {
                "status": "completed",
                "mode": config["mode"],
                "config": config,
                "environment": signature,
                **progress(),
                "training_seconds": training_seconds,
                "total_seconds_including_heldout_evaluation": time.perf_counter() - start_time,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "heldout_checkpoint": heldout_checkpoint,
                "heldout_summaries": {name: result["summary"] for name, result in report["results"].items()},
                "evaluation_complete": report.get("complete", True),
                "evaluation_stop_reason": report.get("stopped_reason"),
                "task_performance_evaluated": report.get("complete", True),
            },
        )
        for name, result in report["results"].items():
            summary = result["summary"]
            if summary is None:
                print(f"Held-out {name}: budget reached before any episode completed; no score assigned.")
            else:
                print(f"Held-out {name}: {summary['successes']}/{summary['episodes']} successes, "
                      f"mean final distance {summary['mean_final_distance_m'] * 1000:.1f} mm"
                      f" ({'complete' if result.get('complete', True) else 'partial evidence'})")
        print(f"Training and held-out evaluation saved to {output}")
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        checkpoint = None
        if interrupted and not trial and model is not None:
            save_checkpoint(
                model, output / "final_model.zip", signature,
                {**config, **progress(), "stop_reason": "user_interrupt",
                 "elapsed_seconds": time.perf_counter() - start_time},
            )
            checkpoint = "final_model.zip"
            print(f"Training interrupted; current model saved to {output / checkpoint}", flush=True)
        write_json(
            output / "training.json",
            {
                "status": "interrupted" if interrupted else "failed",
                "mode": config["mode"],
                **({"trial_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"} if trial else {}),
                "config": config,
                "environment": signature,
                **progress(),
                "stop_reason": "user_interrupt" if interrupted else "error",
                **({"checkpoint": checkpoint} if checkpoint is not None else {}),
                "task_performance_evaluated": False,
                "elapsed_seconds": time.perf_counter() - start_time,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        # Every resource must close even if a worker or viewer cleanup fails.
        with ExitStack() as cleanup:
            if evaluation_callback is not None:
                cleanup.callback(evaluation_callback.eval_env.close)
            if view_callback is not None:
                cleanup.callback(view_callback.close)
            logger = getattr(model, "logger", None)
            if logger is not None:
                cleanup.callback(logger.close)
            if vector_env is not None:
                cleanup.callback(vector_env.close)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training stopped.", flush=True)

"""Check the PPO startup pipeline through its first optimizer step; full training is opt-in."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from .env import SO101ReachEnv
from .evaluate import environment_signature, evaluate_checkpoint, run_episodes, save_checkpoint, write_json


DEFAULT_TRIAL_TIMESTEPS = 500


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


class FixedSeedEvaluation(BaseCallback):
    """Select checkpoints using validation goals, independent from final held-out goals."""

    def __init__(
        self,
        output: Path,
        signature: dict[str, Any],
        training_config: dict[str, Any],
        *,
        eval_frequency: int = 25_000,
        eval_episodes: int = 20,
        start_time: float,
    ):
        super().__init__()
        self.output = output
        self.signature = signature
        self.training_config = training_config
        self.eval_frequency = eval_frequency
        self.eval_seeds = list(range(10_000, 10_000 + eval_episodes))
        self.next_evaluation = eval_frequency
        self.start_time = start_time
        self.best_score: tuple[float, float, float] | None = None
        self.eval_env = SO101ReachEnv()
        self.last_evaluation = -1

    def training_metadata(self) -> dict[str, Any]:
        return {
            **self.training_config,
            "actual_timesteps": self.num_timesteps,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "validation_seeds": self.eval_seeds,
        }

    def evaluate(self) -> dict[str, Any]:
        report = run_episodes(self.eval_env, self.eval_seeds, model=self.model)
        summary = report["summary"]
        row = {
            "timesteps": self.num_timesteps,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            **{key: value for key, value in summary.items() if not isinstance(value, list)},
        }
        csv_path = self.output / "evaluations.csv"
        needs_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        score = (
            summary["success_rate"],
            -summary["mean_final_distance_m"],
            summary["mean_return"],
        )
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            save_checkpoint(self.model, self.output / "best_model.zip", self.signature, self.training_metadata())
            write_json(self.output / "best_validation.json", {"timesteps": self.num_timesteps, **report})
        self.last_evaluation = self.num_timesteps
        self.logger.record("validation/success_rate", summary["success_rate"])
        self.logger.record("validation/mean_final_distance_m", summary["mean_final_distance_m"])
        self.logger.record("validation/mean_return", summary["mean_return"])
        print(f"Validation at {self.num_timesteps:,} steps: "
              f"{summary['successes']}/{summary['episodes']} successes, "
              f"mean final distance {summary['mean_final_distance_m'] * 1000:.1f} mm", flush=True)
        return report

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation:
            self.evaluate()
            self.next_evaluation = self.num_timesteps + self.eval_frequency
        return True

    def _on_training_end(self) -> None:
        if self.last_evaluation != self.num_timesteps:
            self.evaluate()
        self.eval_env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-training", action="store_true",
        help="Explicitly enable unrestricted training, validation and held-out evaluation.",
    )
    parser.add_argument(
        "--timesteps", type=int,
        help="Startup rollout budget until the first network update (at least 2; default: 500), or total transitions with --full-training (default: 500000).",
    )
    parser.add_argument("--output", type=Path, help="Fresh output directory (default: runs/startup_trial or runs/reach).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs", type=int, help="Parallel environments (startup trial: 1 only; full training default: 8).")
    parser.add_argument("--eval-frequency", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--log-std-init", type=_finite_float, default=-2.0, help="Finite initial Gaussian policy log standard deviation (default: -2).")
    args = parser.parse_args(argv)
    trial = not args.full_training
    args.timesteps = args.timesteps if args.timesteps is not None else (DEFAULT_TRIAL_TIMESTEPS if trial else 500_000)
    args.envs = args.envs if args.envs is not None else (1 if trial else 8)
    args.output = args.output or Path("runs/startup_trial" if trial else "runs/reach")
    if min(args.timesteps, args.envs, args.eval_frequency, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("timesteps, envs and evaluation counts must be positive")
    if trial and args.timesteps < 2:
        parser.error("A startup trial requires at least 2 environment steps before its first network update.")
    if trial and args.envs != 1:
        parser.error("A startup trial uses exactly one environment so its total step budget cannot be multiplied by --envs.")
    output = args.output.resolve()
    if any(output.glob("*.zip")) or (output / "training.json").exists():
        parser.error(f"Output contains an existing run; choose a fresh --output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "algorithm": "PPO",
        "mode": "startup_trial" if trial else "full_training",
        "requested_timesteps": args.timesteps,
        "environment_step_limit": args.timesteps if trial else None,
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
        "n_epochs": 1 if trial else 10,
        "learning_rate": 0.0003,
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
        "checkpoint_loaded": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    probe = SO101ReachEnv()
    try:
        signature = environment_signature(probe)
    finally:
        probe.close()
    write_json(
        output / "training.json",
        {"status": "trial_running" if trial else "running", "mode": config["mode"],
         "config": config, "environment": signature, "actual_timesteps": 0,
         "update_count": 0, "optimizer_steps": 0, "gradient_update_count": 0},
    )

    def make_env(index: int):
        def factory():
            return Monitor(
                SO101ReachEnv(),
                filename=str(output / f"monitor_{index}.csv"),
                info_keywords=("is_success", "distance", "hold_steps"),
            )
        return factory

    vector_env = DummyVecEnv([make_env(index) for index in range(args.envs)])
    vector_env.seed(args.seed)
    start_time = time.perf_counter()
    callback = None
    model = None
    optimizer_counter = None

    def progress() -> dict[str, Any]:
        epochs = int(model._n_updates) if model is not None else 0
        return {
            "actual_timesteps": int(model.num_timesteps) if model is not None else 0,
            # With no target_kl/early stopping, every PPO rollout update executes
            # n_epochs complete optimization epochs, as counted by SB3.
            "update_count": epochs // config["n_epochs"],
            "update_count_unit": "completed PPO rollout updates",
            "optimization_epochs": epochs,
            "optimizer_steps": optimizer_counter.count if optimizer_counter is not None else 0,
            "gradient_update_count": optimizer_counter.count if optimizer_counter is not None else 0,
            "gradient_update_count_unit": "completed optimizer.step calls",
        }

    try:
        if not trial:
            callback = FixedSeedEvaluation(
                output,
                signature,
                config,
                eval_frequency=args.eval_frequency,
                eval_episodes=args.eval_episodes,
                start_time=start_time,
            )
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
            policy_kwargs={"net_arch": config["policy_net_arch"], "log_std_init": config["log_std_init"]},
            device="cpu",
            seed=args.seed,
            verbose=1,
        )
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        with _OptimizerStepCounter(model.policy.optimizer, limit=1 if trial else None) as optimizer_counter:
            model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=1 if trial else 5)
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
        save_checkpoint(model, output / "final_model.zip", signature, callback.training_metadata())
        training_seconds = time.perf_counter() - start_time
        report = evaluate_checkpoint(output / "best_model.zip", episodes=args.heldout_episodes, seed=20_000)
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
                "heldout_checkpoint": "best_model.zip",
                "heldout_summaries": {name: result["summary"] for name, result in report["results"].items()},
            },
        )
        for name, result in report["results"].items():
            summary = result["summary"]
            print(f"Held-out {name}: {summary['successes']}/{summary['episodes']} successes, "
                  f"mean final distance {summary['mean_final_distance_m'] * 1000:.1f} mm")
        print(f"Training and held-out evaluation saved to {output}")
    except BaseException as exc:
        write_json(
            output / "training.json",
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "mode": config["mode"],
                **({"trial_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"} if trial else {}),
                "config": config,
                "environment": signature,
                **progress(),
                "task_performance_evaluated": False,
                "elapsed_seconds": time.perf_counter() - start_time,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        vector_env.close()
        if callback is not None:
            callback.eval_env.close()


if __name__ == "__main__":
    main()

"""Train the first SO-101 reaching policy using PPO on CPU."""

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


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=Path("runs/reach"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval-frequency", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--log-std-init", type=_finite_float, default=-2.0, help="Finite initial Gaussian policy log standard deviation (default: -2).")
    args = parser.parse_args()
    if min(args.timesteps, args.envs, args.eval_frequency, args.eval_episodes, args.heldout_episodes) < 1:
        parser.error("timesteps, envs and evaluation counts must be positive")
    output = args.output.resolve()
    if any(output.glob("*.zip")) or (output / "training.json").exists():
        parser.error(f"Output contains an existing run; choose a fresh --output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "algorithm": "PPO",
        "requested_timesteps": args.timesteps,
        "seed": args.seed,
        "num_envs": args.envs,
        "device": "cpu",
        "torch_threads": 1,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 10,
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
        "evaluation_frequency": args.eval_frequency,
        "validation_episodes": args.eval_episodes,
        "heldout_episodes": args.heldout_episodes,
        "heldout_seed_start": 20_000,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    probe = SO101ReachEnv()
    try:
        signature = environment_signature(probe)
    finally:
        probe.close()
    write_json(output / "training.json", {"status": "running", "config": config, "environment": signature})

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
    callback = FixedSeedEvaluation(
        output,
        signature,
        config,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        start_time=start_time,
    )
    try:
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
        model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=5)
        save_checkpoint(model, output / "final_model.zip", signature, callback.training_metadata())
        training_seconds = time.perf_counter() - start_time
        report = evaluate_checkpoint(output / "best_model.zip", episodes=args.heldout_episodes, seed=20_000)
        write_json(output / "evaluation.json", report)
        write_json(
            output / "training.json",
            {
                "status": "completed",
                "config": config,
                "environment": signature,
                "actual_timesteps": model.num_timesteps,
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
                "config": config,
                "environment": signature,
                "elapsed_seconds": time.perf_counter() - start_time,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        vector_env.close()
        callback.eval_env.close()


if __name__ == "__main__":
    main()

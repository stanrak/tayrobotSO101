"""Validate and restore explicit training checkpoints without changing history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from .evaluate import metadata_path, package_versions
from .learning import DiagnosticPPO


class ResumeError(ValueError):
    """A checkpoint cannot be continued with a verified training state."""


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: Path
    metadata: dict[str, Any]
    training: dict[str, Any]
    counters: dict[str, int]


_COUNTERS = {
    "timesteps": "num_timesteps",
    "optimizer_steps": "diagnostic_optimizer_steps",
    "update_count": "rollout_update_count",
    "optimization_epochs": "_n_updates",
}
_MODEL_SETTINGS = (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
    "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
    "use_sde", "sde_sample_freq",
)


def _json_object(content: str | bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON value: {value}")

    try:
        result = json.loads(content, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ResumeError(f"Invalid {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise ResumeError(f"Invalid {label}: expected a JSON object.")
    return result


def _counter(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ResumeError(f"Missing or invalid {label}; resume requires recorded nonnegative integer counters.")
    return value


def _validate_settings(training: dict[str, Any]) -> None:
    for name in ("n_steps", "batch_size", "n_epochs"):
        if _counter(training.get(name), f"training setting {name}") < 1:
            raise ResumeError(f"Training setting {name} must be positive.")
    for name in (
        "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm", "log_std_init", "reward_scale",
    ):
        value = training.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ResumeError(f"Missing or invalid finite training setting: {name}.")
        if name in ("learning_rate", "reward_scale", "max_grad_norm") and value <= 0:
            raise ResumeError(f"Training setting {name} must be positive.")
    target_kl = training.get("target_kl")
    if target_kl is not None and (
        type(target_kl) not in (int, float) or not math.isfinite(target_kl) or target_kl <= 0
    ):
        raise ResumeError("Training setting target_kl must be null or finite and positive.")
    for name in ("use_sde", "observation_normalization", "reward_normalization"):
        if type(training.get(name)) is not bool:
            raise ResumeError(f"Missing or invalid boolean training setting: {name}.")
    if training["observation_normalization"] or training["reward_normalization"]:
        raise ResumeError("Resume does not support running normalization statistics; use an unnormalized checkpoint.")
    frequency = training.get("sde_sample_freq")
    if type(frequency) is not int or (frequency != -1 and frequency < 1):
        raise ResumeError("Training setting sde_sample_freq must be -1 or a positive integer.")
    architecture = training.get("policy_net_arch")
    if not isinstance(architecture, list) or not architecture or any(type(size) is not int or size < 1 for size in architecture):
        raise ResumeError("Missing or invalid policy_net_arch in training metadata.")


def _verified_bytes(checkpoint: Path, expected_digest: Any) -> bytes:
    try:
        content = checkpoint.read_bytes()
    except OSError as exc:
        raise ResumeError(f"Cannot read resume checkpoint {checkpoint}: {exc}") from exc
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ResumeError("Resume checkpoint contents do not match its metadata SHA256.")
    return content


def _environment_mismatch(saved: Any, current: dict[str, Any]) -> str:
    """Explain compatibility separately from checkpoint/run discovery."""
    def describe(value: Any) -> tuple[Any, Any, Any]:
        if not isinstance(value, dict):
            return "unrecorded", "unrecorded", "unrecorded"
        config = value.get("task_config")
        task = config.get("task", "unrecorded") if isinstance(config, dict) else value.get("task", "unrecorded")
        return task, value.get("observation_shape", "unrecorded"), value.get("action_shape", "unrecorded")

    previous = describe(saved)
    expected = describe(current)
    detail = (f"Checkpoint found, but resume checkpoint environment does not match the current task, assets, or interface. "
              f"Saved: task={previous[0]}, observations={previous[1]}, actions={previous[2]}. "
              f"Current: task={expected[0]}, observations={expected[1]}, actions={expected[2]}. ")
    if previous != expected:
        detail += "Different tasks or observation/action layouts cannot share a resumed policy. "
    else:
        detail += "Task names and shapes agree, but source, task settings, or asset signatures differ. "
    return (detail + "To start fresh on the current task, omit --resume. To continue this checkpoint, "
            "use its matching code and environment; checkpoint migration is not automatic.")


def inspect_resume_checkpoint(checkpoint: Path, signature: dict[str, Any]) -> ResumeCheckpoint:
    """Inspect only JSON/ZIP records; do not deserialize policy objects here."""
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix != ".zip":
        checkpoint = checkpoint.with_suffix(".zip")
    try:
        metadata = _json_object(metadata_path(checkpoint).read_text(encoding="utf-8"), "checkpoint metadata")
    except OSError as exc:
        raise ResumeError(f"Cannot read resume checkpoint metadata: {exc}") from exc
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1 or metadata.get("algorithm") != "PPO":
        raise ResumeError("Unsupported resume checkpoint schema or algorithm; expected PPO schema 1.")
    content = _verified_bytes(checkpoint, metadata.get("checkpoint_sha256"))
    if metadata.get("environment") != signature:
        raise ResumeError(_environment_mismatch(metadata.get("environment"), signature))
    if metadata.get("package_versions") != package_versions():
        raise ResumeError("Resume checkpoint package versions differ; use its recorded environment without migrating it.")
    training = metadata.get("training")
    if not isinstance(training, dict) or training.get("mode") != "full_training":
        raise ResumeError("Only full-training checkpoints can be resumed; startup trials remain fresh.")
    _validate_settings(training)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            required = {"data", "policy.pth", "policy.optimizer.pth"}
            if not required.issubset(archive.namelist()):
                raise ResumeError("Resume checkpoint is missing policy, optimizer, or training data.")
            data = _json_object(archive.read("data"), "checkpoint training data")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ResumeError(f"Invalid resume checkpoint archive: {exc}") from exc
    counters = {name: _counter(data.get(field), f"checkpoint counter {field}") for name, field in _COUNTERS.items()}
    for name, value in counters.items():
        cumulative = f"cumulative_{name}"
        legacy = "actual_timesteps" if name == "timesteps" else name
        if cumulative in training:
            recorded = _counter(training[cumulative], f"metadata {cumulative}")
        elif _counter(training.get(f"starting_{name}", 0), f"metadata starting_{name}") == 0 and legacy in training:
            recorded = _counter(training[legacy], f"metadata {legacy}")
        else:
            continue
        if recorded != value:
            raise ResumeError(f"Checkpoint and metadata disagree about {name}.")
    return ResumeCheckpoint(checkpoint, metadata, training, counters)


def load_resume_policy(record: ResumeCheckpoint, vector_env, *, seed: int) -> DiagnosticPPO:
    """Restore weights/Adam state; reset session metrics, never run an action."""
    # Load the same bytes just verified, rather than reopening a potentially
    # changed path after the hash check. No fallback to a new policy is allowed.
    content = _verified_bytes(record.path, record.metadata.get("checkpoint_sha256"))
    try:
        model = DiagnosticPPO.load(io.BytesIO(content), env=vector_env, device="cpu", force_reset=True, seed=seed)
    except Exception as exc:
        raise ResumeError(f"Could not restore the resume policy and optimizer: {exc}") from exc
    for name, field in _COUNTERS.items():
        if _counter(getattr(model, field, None), f"restored {field}") != record.counters[name]:
            raise ResumeError(f"Restored model counter {field} differs from the inspected checkpoint.")
    for name in _MODEL_SETTINGS:
        actual = getattr(model, name, None)
        if name in ("learning_rate", "clip_range") and callable(actual):
            actual = actual(1.0)
        expected = record.training.get(name)
        if actual != expected:
            raise ResumeError(f"Restored training setting {name} differs from checkpoint metadata.")
    for metadata_key, policy_key in (("policy_net_arch", "net_arch"), ("log_std_init", "log_std_init")):
        if model.policy_kwargs.get(policy_key) != record.training[metadata_key]:
            raise ResumeError(f"Restored policy setting {metadata_key} differs from checkpoint metadata.")
    for name in ("rollout_seconds", "optimization_seconds", "diagnostics_seconds"):
        setattr(model, name, 0.0)
    model.ep_info_buffer = None
    model.ep_success_buffer = None
    model._episode_num = 0
    return model

"""Compare a new session with verified saved validation of its exact parent.

This module reads adjacent artifacts only. It never loads a policy, collects
transitions, changes checkpoint selection, or writes to a historical run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


_PARENT_SCOPE = "exact_selected_parent_saved_validation"
_COMPARISON_SCOPE = "immediate_parent_on_matching_validation_seeds"
_SELECTION_ORDER = ["success_rate", "negative_mean_final_distance_m", "mean_return"]


class _Unavailable(ValueError):
    """A saved artifact cannot establish a comparable parent baseline."""


def _finite_json(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise _Unavailable("Saved artifacts contain non-finite JSON numbers.")


def _object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str):
        raise _Unavailable(f"Saved artifact contains non-finite JSON: {value}.")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _Unavailable("Saved artifact contains duplicate JSON keys.")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_keys,
    )
    if not isinstance(value, dict):
        raise _Unavailable("Saved artifact must contain a JSON object.")
    _finite_json(value)
    return value


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise _Unavailable("Saved validation requires nonnegative integer timestep counters.")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise _Unavailable("Validation scores and episode values must be finite numbers.")
    return float(value)


def _score(summary: Any) -> tuple[float, float, float]:
    if not isinstance(summary, dict):
        raise _Unavailable("Validation summary is missing or malformed.")
    success = _number(summary.get("success_rate"))
    distance = _number(summary.get("mean_final_distance_m"))
    episode_return = _number(summary.get("mean_return"))
    if not 0 <= success <= 1 or distance < 0:
        raise _Unavailable("Validation success rate or distance is outside its valid range.")
    return success, -distance, episode_return


def _training(metadata: Any) -> dict[str, Any]:
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("algorithm") != "PPO"
        or not isinstance(metadata.get("environment"), dict)
        or not metadata["environment"]
    ):
        raise _Unavailable("Checkpoint metadata schema, algorithm, or environment is unsupported.")
    training = metadata.get("training")
    if not isinstance(training, dict) or training.get("mode") != "full_training":
        raise _Unavailable("Parent validation requires a full-training checkpoint.")
    _finite_json(metadata)
    return training


def _match_counters(training: dict[str, Any], report: dict[str, Any]) -> None:
    session = _counter(report.get("timesteps"))
    cumulative = _counter(report.get("cumulative_timesteps"))
    if (
        _counter(training.get("actual_timesteps")) != session
        or _counter(training.get("cumulative_timesteps")) != cumulative
        or cumulative < session
    ):
        raise _Unavailable("Saved validation timestep counters do not match the selected parent checkpoint.")
    if "starting_timesteps" in training and _counter(training["starting_timesteps"]) + session != cumulative:
        raise _Unavailable("Checkpoint starting, session, and cumulative timesteps disagree.")


def _same_number(actual: Any, expected: float) -> bool:
    return math.isclose(_number(actual), expected, rel_tol=1e-12, abs_tol=1e-12)


def _validated_summary(report: dict[str, Any], seeds: list[int]) -> tuple[dict[str, Any], tuple[float, float, float]]:
    if report.get("policy") != "trained":
        raise _Unavailable("Saved parent validation is not a trained-policy report.")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(seeds):
        raise _Unavailable("Saved validation episodes do not match the requested seed count.")
    if any(not isinstance(episode, dict) for episode in episodes):
        raise _Unavailable("Saved validation episode rows are malformed.")
    if any(type(episode.get("seed")) is not int for episode in episodes) or [episode["seed"] for episode in episodes] != seeds:
        raise _Unavailable("Saved validation episode seeds or their order do not match the child validation seeds.")
    if any(type(episode.get("is_success")) is not bool for episode in episodes):
        raise _Unavailable("Saved validation success flags are malformed.")
    returns = [_number(episode.get("return")) for episode in episodes]
    distances = [_number(episode.get("final_distance_m")) for episode in episodes]
    if any(distance < 0 for distance in distances):
        raise _Unavailable("Saved validation episode distances cannot be negative.")
    summary = report.get("summary")
    score = _score(summary)
    count = len(episodes)
    successes = sum(episode["is_success"] for episode in episodes)
    if (
        _counter(summary.get("episodes")) != count
        or _counter(summary.get("successes")) != successes
        or not _same_number(summary["success_rate"], successes / count)
        or not _same_number(summary["mean_return"], math.fsum(returns) / count)
        or not _same_number(summary["mean_final_distance_m"], math.fsum(distances) / count)
    ):
        raise _Unavailable("Saved validation summary scores do not agree with its raw episode rows.")
    return summary, score


def load_parent_validation(checkpoint: Path, metadata: dict, seeds: list[int]) -> dict[str, Any]:
    """Read a saved baseline only when it belongs to the exact selected ZIP.

    A final checkpoint may share a sibling best validation only when its bytes
    and recorded timestep counters equal the validated best checkpoint. Missing
    evidence yields an explicit unavailable result instead of a fresh rollout.
    """
    checkpoint = Path(checkpoint)
    result = {
        "status": "unavailable",
        "reason": None,
        "scope": _PARENT_SCOPE,
        "source": "saved_best_validation",
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": None,
        "validation_path": str(checkpoint.parent / "best_validation.json"),
        "seeds": copy.deepcopy(seeds),
        "summary": None,
        "score": None,
    }
    try:
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise _Unavailable("Child validation seeds must be an explicit nonempty list of integers.")
        checkpoint = checkpoint.resolve()
        result["parent_checkpoint"] = str(checkpoint)
        result["validation_path"] = str(checkpoint.parent / "best_validation.json")
        if checkpoint.suffix != ".zip":
            raise _Unavailable("The selected parent must be a ZIP checkpoint.")
        training = _training(metadata)
        parent_digest = _digest(checkpoint)
        result["parent_checkpoint_sha256"] = parent_digest
        if metadata.get("checkpoint_sha256") != parent_digest:
            raise _Unavailable("The selected parent checkpoint does not match its metadata SHA256.")

        best_path = checkpoint.parent / "best_model.zip"
        best_metadata = _object(best_path.with_suffix(".metadata.json"))
        best_training = _training(best_metadata)
        best_digest = _digest(best_path)
        if best_metadata.get("checkpoint_sha256") != best_digest:
            raise _Unavailable("The sibling best checkpoint does not match its metadata SHA256.")
        if best_digest != parent_digest:
            raise _Unavailable("Saved best validation belongs to a different checkpoint than the selected parent.")
        if best_metadata["environment"] != metadata["environment"]:
            raise _Unavailable("Saved best checkpoint environment differs from the selected parent.")
        if best_metadata.get("package_versions") != metadata.get("package_versions"):
            raise _Unavailable("Saved best checkpoint package versions differ from the selected parent.")
        report = _object(checkpoint.parent / "best_validation.json")
        if "checkpoint_sha256" in report and report["checkpoint_sha256"] != parent_digest:
            raise _Unavailable("Saved validation checkpoint SHA256 differs from the selected parent.")
        _match_counters(training, report)
        _match_counters(best_training, report)
        for record in (training, best_training):
            if "validation_seeds" in record and (
                not isinstance(record["validation_seeds"], list)
                or any(type(seed) is not int for seed in record["validation_seeds"])
                or record["validation_seeds"] != seeds
            ):
                raise _Unavailable("Saved checkpoint validation seeds differ from the child validation seeds.")
        summary, score = _validated_summary(report, seeds)
        result.update(status="available", summary=copy.deepcopy(summary), score=list(score))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result["reason"] = str(exc) or type(exc).__name__
    return result


def compare_parent_validation(parent: dict, child_summary: dict | None) -> dict[str, Any]:
    """Compare session best with its immediate parent using the selector score.

    This does not rank a whole lineage, alter either run's best checkpoint, or
    change which run a latest-run pointer identifies. Equal scores retain the
    parent recommendation; pending and unavailable results make no selection.
    """
    result = {
        "status": "unavailable",
        "reason": parent.get("reason"),
        "scope": _COMPARISON_SCOPE,
        "selection_order": list(_SELECTION_ORDER),
        "parent": copy.deepcopy(parent),
        "child_scope": "new_session_best_validation",
        "child_summary": None,
        "child_score": None,
        "recommendation": "unavailable",
    }
    # Do not return NaN/Infinity in an unavailable result: the caller must still
    # be able to persist the explicit reason as ordinary JSON.
    child_score = None
    if child_summary is not None:
        try:
            child_score = _score(child_summary)
            _finite_json(child_summary)
        except (ValueError, TypeError, OverflowError) as exc:
            result["reason"] = str(exc) or type(exc).__name__
            return result
        result["child_summary"] = copy.deepcopy(child_summary)
        result["child_score"] = list(child_score)
    if parent.get("status") != "available":
        result["reason"] = result["reason"] or "No verified saved validation is available for the exact selected parent."
        return result
    if child_summary is None:
        result.update(status="pending", reason="The child session has no completed validation yet.")
        return result
    try:
        parent_score = _score(parent.get("summary"))
        if list(parent_score) != parent.get("score"):
            raise _Unavailable("Parent summary and its recorded selector score disagree.")
    except (ValueError, TypeError, OverflowError) as exc:
        result["reason"] = str(exc) or type(exc).__name__
        return result
    status = "better" if child_score > parent_score else "worse" if child_score < parent_score else "equal"
    result.update(
        status=status,
        reason=None,
        recommendation="child" if status == "better" else "parent",
    )
    return result

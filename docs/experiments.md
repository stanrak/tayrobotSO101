# Experiment records and evaluation

[Documentation](README.md) · [Project overview](../README.md) · [Training](training.md) · [Environment](environment.md)

Training, validation, and standalone evaluation have separate records. This makes it possible to compare saved results without overwriting an earlier experiment or confusing startup diagnostics with learned performance.

## Where results are stored

Every training or trial invocation creates a directory under:

```text
runs/experiments/<UTC timestamp>_<mode>_seed<seed>_<unique suffix>/
```

Omitting `--output` selects this managed layout. A custom training output must be an empty directory. A run starts a fresh policy unless `--resume` explicitly selects a compatible checkpoint. Both modes leave older experiment directories at their original paths.

| File or directory | Contents |
| --- | --- |
| `run.json` | Run identity, command, seed, timestamps/status, platform/packages, Git revision/state, source hashes, parent lineage when resumed, and observed counts. |
| `source/` | Exact allowlisted source/configuration snapshot recorded at run start. |
| `training.json` | Training settings, environment signature, requested/actual session transitions, starting/cumulative counters, parent lineage, phase timings, and final status. |
| `progress.csv` | PPO, critic, action, task, and timing metrics. |
| `monitor_*.csv.monitor.csv` | Per-environment episode records in original reward units. |
| `best_model.zip` and metadata | Best checkpoint selected by fixed-seed validation. |
| `final_model.zip` and metadata | Policy saved at the end of full training or on a user interruption. |
| `trial_model.zip` and metadata | Policy after the trial's first optimizer step. |
| `evaluations.csv`, `best_validation.json` | Periodic validation summaries and the best validation episode report. |
| `evaluation.json` | Final held-out report when full training completes and evaluation succeeds. |

Failed and interrupted runs retain the observed status and counts. An abrupt process kill can leave a previously recorded running status with partial logs. Missing evaluation metrics mean they were not measured, rather than zero performance.

`runs/latest.json` is an atomic convenience pointer to the most recently started managed training run. It is not the best run across all history. The latest run's checkpoint lookup tries `best_model.zip`, then `trial_model.zip`, then `final_model.zip`. When the pointer is absent, lookup falls back to legacy `runs/current/`; an invalid pointer fails clearly. A recent failed run without a checkpoint can therefore require an explicit checkpoint path.

## Continued-training lineage and counts

`--resume CHECKPOINT` continues optimization from a checked full-training checkpoint. `--resume latest` resolves the existing pointer and checkpoint selection before the new run updates `runs/latest.json`. Because selection prefers `best_model.zip`, the starting count belongs to that selected checkpoint and can be earlier than the parent's final count. An explicit `final_model.zip` path selects the latest saved training state in that parent run instead.

The parent checkpoint's bytes and metadata are retained. Continuation creates a new output directory and records:

| Field | Meaning |
| --- | --- |
| `config.checkpoint_loaded` | Whether this run explicitly restored a checkpoint. |
| `config.parent_checkpoint` / `parent_checkpoint_sha256` / `parent_run_id` | Identity of the selected parent state. |
| `config.starting_timesteps` / `starting_optimizer_steps` / `starting_update_count` / `starting_optimization_epochs` | Cumulative counts at the selected checkpoint. |
| `actual_timesteps` / `optimizer_steps` / `update_count` / `optimization_epochs` | Work observed in the new session only. |
| `cumulative_timesteps` / `cumulative_optimizer_steps` / `cumulative_update_count` / `cumulative_optimization_epochs` | Parent baseline plus observed work in this session. |
| `phase_seconds` | Timing accumulated only in this session. |

For example, a 300,000-transition checkpoint followed by a 200,000-transition session yields `actual_timesteps=200000` and `cumulative_timesteps=500000`. Counts refer to the selected checkpoint, not all independent runs stored on disk. Legacy records without measured cumulative or lineage fields remain blank in comparisons; no values are invented by summing unrelated runs.

Checkpoint ZIPs preserve policy weights and optimizer state, while logs and evaluation scores remain records rather than a replay dataset. A resumed session starts fresh environment trajectories using its seed. State restoration checks hashes, task/environment/assets, package versions, reward scale, optimizer availability, and continuation counters. Unsupported older states fail clearly and are not automatically migrated. See [Training](training.md) for commands and compatible settings.

## Validation and held-out results

Periodic validation uses fixed seeds starting at 10,000, with 20 episodes by default. Checkpoint selection prioritizes success rate, then lower mean final distance, then higher mean return. The same seeds are reused for selection, so the best validation score is not an independent test result.

A normally completed full-training run evaluates its best checkpoint on held-out seeds starting at 20,000, with 100 episodes per policy by default. It compares:

- The trained policy, using deterministic predictions.
- A no-motion baseline using the reset pose's stationary action.
- A random-action baseline receiving identical seeded initial conditions.

Held-out evaluation has additional simulation cost beyond the requested training transition count. A user interruption saves the current full-training policy and skips the final held-out path. Trial mode has no validation, baseline comparisons, checkpoint selection, or held-out task score.

Reports contain success counts/rates and Wilson 95% intervals; collision and timeout counts; episode length and return; final/minimum distance; and hold progress. Per-episode termination reasons help distinguish approach, precision, holding, collision, and timeout problems. Reward values use the original environment units, regardless of training reward scale.

## Evaluate a saved policy

Finish or stop the current training before running standalone evaluation. To evaluate and view the latest available policy:

```powershell
.\.venv\Scripts\python -m so101_sim.evaluate --render --episodes 5
```

For a quantitative report without a viewer:

```powershell
.\.venv\Scripts\python -m so101_sim.evaluate --episodes 100 --seed 20000
```

Use `--checkpoint` to select a specific checkpoint instead of the latest-run pointer. Checkpoints include metadata checks for content SHA256, algorithm/schema, environment signature, model asset integrity, and package versions. An incompatible historical policy is rejected rather than migrated automatically.

Each invocation creates a unique record beside the checkpoint:

```text
<checkpoint directory>/evaluations/<evaluation_id>/
    run.json
    source/
    evaluation.json
```

Reports identify the checkpoint by path and SHA256. Existing reports are retained. Optional `--output` accepts a new report filename and also keeps a canonical report in the per-evaluation record. Rendering shows only the trained policy; baselines are still included in the report. Playback is paced at simulation real time, and closing the playback window before completion stops evaluation with an error.

## Compare saved runs without running policies

```powershell
.\.venv\Scripts\python -m so101_sim.experiments
```

This rebuilds `runs/index.csv` from saved metadata and reports. It does not load a policy or generate new task measurements. The derived table includes legacy and managed runs, learning settings, reward scale, backend, source/environment fingerprints, parent lineage, session/cumulative counts, budgets, phase timings, and measured metrics. Each evaluation row identifies its kind, checkpoint hash, and seeds.

Compare matching task/environment signatures and evaluation seeds at comparable budgets. Validation, held-out, and standalone results are labeled separately. Startup diagnostics have blank task-performance metrics. A result from an older task version cannot serve as a directly comparable success-rate baseline for the physical ball task.

Custom training outputs outside `runs/` are outside the default catalog. To compare directories under another parent, use:

```powershell
.\.venv\Scripts\python -m so101_sim.experiments --root <parent>
```

Replace `<parent>` with the actual path before running the command.

## Concurrent access and historical files

Writer locks protect training, standalone evaluation, and catalog rebuilding from concurrent modification. The existing `runs/.current.lock` name remains compatible with older processes. The operating system releases the lock after the writer exits, including after a crash; deleting the lock file is not a way to safely unlock an active process.

Existing `runs/current/`, `runs/history/`, `checkpoints/`, and `reports/` records remain historical local artifacts. Generated learning GIFs and run outputs are excluded from ordinary Git tracking. New experiment folders include a local ignore marker, including custom output folders. A clean checkout may therefore have no trained checkpoint or historical report. Static simulation images can remain part of the documentation.


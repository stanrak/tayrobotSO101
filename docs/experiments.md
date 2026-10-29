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
| `parent_comparison.json` | For continuation, availability and outcome of the child's validation comparison against the exact selected parent's saved validation. |
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

Cumulative counts equal the selected checkpoint's starting counts plus the observed session work. Counts refer to the selected checkpoint, not all independent runs stored on disk. Legacy records without measured cumulative or lineage fields remain blank in comparisons; no values are invented by summing unrelated runs.

Checkpoint ZIPs preserve policy weights and optimizer state, while logs and evaluation scores remain records rather than a replay dataset. A resumed session starts fresh environment trajectories using its seed. State restoration checks hashes, task/environment/assets, package versions, reward scale, optimizer availability, and continuation counters. Unsupported older states fail clearly and are not automatically migrated. See [Training](training.md) for commands and compatible settings.

## Compare continuation with its selected parent

A resumed run writes `parent_comparison.json` separately from its own checkpoint selection. The baseline must belong to the exact parent checkpoint chosen by `--resume`: the selected ZIP must have the same verified SHA256 as that run's saved best ZIP. Their checkpoint environments must match, and both records' session/cumulative transition counts must agree with the saved validation. Validation episode seeds must match the child's ordered seed list and episode budget; the selected checkpoint also passes the normal continuation compatibility checks. A validation report's own checkpoint hash is checked when present; new reports include it. The comparison reads existing records and does not evaluate the parent or search earlier ancestors for a substitute.

The nested `parent.status` identifies an `available` or `unavailable` baseline. The overall `status` is `pending` until the child has a completed validation, then `better`, `equal`, or `worse`; a missing or mismatched baseline produces `unavailable` with a `reason`. For example, a selected final checkpoint from an interrupted run may have no validation tied to that exact saved state; the parent's best-validation score then cannot stand in for it. Even identical policy weights in different ZIP archives do not establish an exact hash match. Different validation seeds or episode budgets also prevent comparison. Missing evidence does not imply that the parent or child performed poorly.

The comparison uses the same ranking as child checkpoint selection: higher success rate, then lower mean final distance, then higher raw return. `recommendation` names the child for a better score and the parent for an equal or worse score; it is unavailable before a valid comparison. This compares reused validation data and does not provide a new held-out result or statistical proof of improvement. The first completed child validation still creates `best_model.zip`, and later improvements replace it within the child run. Held-out evaluation continues to use the child's best checkpoint. A weaker child receives an explicit comparison result; it does not trigger a rollback, copy the parent checkpoint, or change `runs/latest.json` from the newest run to a stronger older run.

After each child validation, the terminal prints `Session-best validation versus selected parent` with the comparison status and recommendation. Here `better` and `worse` describe the child's session best relative to the selected parent, rather than the latest validation relative to the child's previous round. The JSON file retains the availability reason, both scores, and child-best checkpoint identity/counters for later inspection.

## Validation and held-out results

Periodic validation uses fixed seeds starting at 10,000, with 20 episodes by default. Checkpoint selection prioritizes success rate, then lower mean final distance, then higher mean return. The same seeds are reused for selection, so the best validation score is not an independent test result.

A normally completed full-training run evaluates its best checkpoint on held-out seeds starting at 20,000, with 100 episodes per policy by default. It compares:

- The trained policy, using deterministic predictions.
- A no-motion baseline using the reset pose's stationary action.
- A random-action baseline receiving identical seeded initial conditions.

Held-out evaluation runs these three policies sequentially in one environment, outside the requested training transition count. The default is 300 episodes in total, with up to 360,000 additional control steps. `--heldout-episodes` sets the episode count **per policy**; smaller counts reduce runtime and statistical precision.

`final_model.zip` is saved before this phase. The `Monitor` and `DummyVecEnv` messages after final validation come from loading the selected checkpoint for evaluation; they do not indicate result packaging. The terminal identifies checkpoint loading and each policy's start, reports in-flight progress about every 30 seconds, and prints each policy's completion. Progress includes completed episodes, control steps, and overall elapsed time.

Before held-out evaluation begins, `training.json` records status `evaluating` and the measured training counters. `evaluation.json` is written only after the complete report is available. Ctrl+C saves the current full-training policy and records status `interrupted`; it skips any remaining held-out work. Trial mode has no validation, baseline comparisons, checkpoint selection, or held-out task score.

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

Existing `runs/current/`, `runs/history/`, `checkpoints/`, and `reports/` records remain historical local artifacts. Run outputs, generated learning GIFs, detailed reviews, and their measurements are confidential local artifacts excluded from Git tracking. Detailed reviews belong under `reports/`; public documentation keeps only general technical descriptions, process, and broad progress. New experiment folders include a local ignore marker, including custom output folders. A clean checkout may therefore have no trained checkpoint or historical report. Static simulation images can remain part of the documentation.

## Save a finalized run with DVC

After training and any optional standalone evaluation have finished, save the latest managed run locally:

```powershell
.\.venv\Scripts\python -m so101_sim.data add
```

This uses the separately installed DVC executable to cache the finalized run and create a Git-ignored pointer beside it. Each run has one saved pointer; existing pointers are preserved. The command does not start training or evaluation. See [Data versioning](data-versioning.md) for selecting another run, checking snapshot status, and retaining both the pointers and cache for recovery.


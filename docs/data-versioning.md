# Save completed runs with DVC

[Documentation](README.md) · [Experiments](experiments.md) · [Training](training.md)

DVC keeps a local snapshot of a finalized experiment directory, including its checkpoints, saved metadata, logs, source snapshot, and any completed standalone evaluations already inside it. Each training invocation creates a new run directory, so each saved run has its own DVC pointer.

The project uses the existing `dvc` executable on `PATH`. DVC is optional tooling outside the simulation environment; it is not added to the simulator or training requirements. Check that the same PowerShell session can find it:

```powershell
dvc --version
```

## Save a run after training

After training has finished or stopped, finish any optional standalone evaluations that should belong to the snapshot. Then run:

```powershell
.\.venv\Scripts\python -m so101_sim.data add
```

Omitting the target selects `latest`, using the managed latest-run pointer. To select another managed run, replace `<run_directory>` below with its directory name:

```powershell
.\.venv\Scripts\python -m so101_sim.data add "runs/experiments/<run_directory>"
```

The command accepts only direct run directories under `runs/experiments/`. The run must have valid finalized provenance: a `finished_utc` value and status `completed`, `trial_completed`, `failed`, or `interrupted`. Nested evaluation records must also be finalized. A process killed before finalization can leave a running record that cannot be saved through this command. Custom outputs, legacy directories, and individual checkpoint files are outside this command's scope.

The command holds the existing output writer lock while it validates and saves the run. Training, evaluation, and catalog writers must have exited. It also rejects links and junctions, preventing a run directory from pulling in files elsewhere.

Saving creates `runs/experiments/<run_directory>.dvc` beside the run and adds the referenced content to `.dvc/cache/`. The original run stays at its existing path. The wrapper uses `dvc add --no-relink`, which avoids recreating workspace links from the cache; see the official [DVC add reference](https://doc.dvc.org/command-reference/add).

## Inspect the saved snapshot

```powershell
dvc status "runs/experiments/<run_directory>.dvc"
```

This compares the current files with the saved pointer. Run it after writers have stopped. Adding a standalone evaluation after saving changes the directory and can make its status modified. The wrapper refuses an existing `.dvc` pointer so it cannot replace the earlier snapshot. Finish evaluations before saving, and let the next training invocation create a new run directory for the next snapshot.

The snapshot contains only files the trainer already saved. It does not create a rollout archive, replay buffer, or missing observation/action dataset. Saving a run neither starts training or evaluation nor adds an automatic training pipeline.

## Local storage and recovery

The project DVC configuration uses `core.no_scm = true`, disables analytics, and selects a copy cache. DVC operates independently of Git. Its usual Git revision comparisons and automatic Git ignore updates are unavailable in this mode, as described in the official [DVC init reference](https://doc.dvc.org/command-reference/init).

DVC runtime state uses the Git-ignored `.dvc/site-cache/` directory inside this project. Run commands from the repository root so this relative setting resolves consistently; the project data command sets that working directory itself.

| Location | Keep for |
| --- | --- |
| `runs/experiments/<run_directory>/` | Original experiment files in their existing location. |
| `runs/experiments/<run_directory>.dvc` | Pointer identifying the saved directory content. |
| `.dvc/cache/` | Cached content needed to recover the snapshot. |

Both the `.dvc` pointers and their referenced cache content are needed for recovery. A Git checkout alone contains neither. Keep them together in a private backup; a cache on the same disk does not protect against losing that disk. Restore into a separate working copy before using recovered files with existing historical records.

Run results, pointer metadata, cache content, and local remote settings remain confidential and Git-ignored. Only the generic DVC configuration and usage instructions belong in Git. There is no configured remote and saving a run does not upload it.

## Optional backup location

To use a private backup disk later, choose its actual path and configure it locally. For example:

```powershell
dvc remote add --local -d backup 'E:\SO101-DVC'
dvc push "runs/experiments/<run_directory>.dvc"
```

These are separate manual operations. `--local` keeps the location in Git-ignored `.dvc/config.local`; DVC supports a local directory as a remote. See the official [DVC remote add reference](https://doc.dvc.org/command-reference/remote/add). A push copies the referenced content to that location; separately retain the `.dvc` pointers and local configuration in the private backup.

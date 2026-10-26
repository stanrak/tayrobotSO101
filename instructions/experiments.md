# Experiment records and comparisons

Read the [instruction index](instruction.md), [training contract](training.md), and [workflow](workflow.md). Output organization does not grant permission to run training or evaluation and does not require a new dependency or tracking service.

## Isolated output directories

- Each trial or full-training invocation automatically creates a unique `runs/experiments/<UTC timestamp>_<mode>_seed<seed>_<unique suffix>/` directory.
- Omit `--output` for ordinary commands. A custom training output must be empty.
- Never reuse, move, or delete older experiment directories. Preserve `runs/current`, `runs/history`, and other legacy records at their original paths and bytes.
- Each standalone evaluation has its own `evaluations/<evaluation_id>/` record beside the checkpoint. Do not overwrite previous reports or custom output files.
- Respect output writer locks, including `runs/.current.lock` for compatibility with older active processes. Do not delete or bypass locks, or mutate an active run.

## Provenance and checkpoints

- Record `run.json` provenance: command, seed, timestamps/status, Python/platform/package versions, Git revision/dirty state, and source hashes.
- Retain an exact allowlisted source/config snapshot alongside training settings, environment signatures, checkpoints, metrics, actual transitions, and actual optimizer counts.
- Keep snapshot selection explicit. Include relevant agent-maintained instructions when reorganizing their paths and retain existing allowlisted sources. Never add private files, credentials, arbitrary repository content, or otherwise unapproved user-owned guidance to snapshots.
- Preserve partial, failed, and interrupted runs accurately. Do not report planned counts as observed counts.
- Do not automatically load policies or migrate historical checkpoints. Keep existing asset, environment-signature, package-version, and checkpoint-integrity checks.
- An explicit full-training `--resume` may restore a compatible checkpoint under the [training continuation contract](training.md). Always create a new output record and keep parent files unchanged; record parent checkpoint path, SHA256, run identifier, and starting counters.
- Preserve session and cumulative counts separately. Top-level actual transition/optimizer/update/epoch counts remain session values; `cumulative_*` includes the selected parent baseline. Do not derive a lineage by summing independent runs or filling missing legacy measurements.
- `runs/latest.json` is an atomic convenience pointer to the latest managed training run. Default checkpoint lookup follows it; fall back to legacy `runs/current` only when the pointer is absent. Invalid pointers must fail clearly.

## Saved-data comparisons

- Maintain the derived `runs/index.csv` catalog from saved metadata, including legacy runs. `python -m so101_sim.experiments` rebuilds it without loading or rolling out policies.
- Label held-out, validation, and standalone evaluation metrics separately. Leave unmeasured fields blank.
- Never assign learned task-performance scores to a startup diagnostic or rank it as a trained policy.
- Compare matching task/environment signatures, evaluation seeds, and budgets. Retain checkpoint identities/hashes, reward scale, backend, learning settings, optimizer counts, and timing scopes needed to interpret a comparison.
- For resumed runs, retain parent lineage and starting/session/cumulative counters in the catalog and finalized provenance. Phase timings describe the new session only. Checkpoint selection can choose a parent best policy earlier than its final model; use the selected checkpoint's counters rather than the parent run's final totals.
- Distinguish best-validation checkpoint metrics, the last recorded validation, and actual final/held-out evaluation. Do not present the last validation as a new evaluation of the final checkpoint.

## Git and historical artifacts

- Keep run results out of Git: `runs/`, historical `checkpoints/` and `reports/`, and generated learning GIFs remain ignored.
- Every new experiment must contain a local ignore marker, including custom output folders.
- Remove previously tracked result artifacts from the Git index without deleting local bytes. Ordinary runs do not rewrite Git history.
- Static simulation documentation images may remain versioned.
- Preserve locks and historical bytes during catalog/documentation work. Do not force-add results or use a reorganization as a reason to delete them.

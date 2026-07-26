# Provenance-complete P0 experiment

This pipeline is for new paper-facing evidence. It does not upgrade historical
results merely by adding manifest rows.

## What is enforced

- clean exact Git commit and immutable 40–64 hex checkpoint revision;
- exact scenario-byte fingerprint, independent rollout seed, repeat index, and
  nondeterminism settings for every episode;
- lossless float32 full hidden activation, action, EEF pose, joint pose/velocity,
  task phase, scoped force trace, outcome, and complete episode metadata;
- nominal swept geometry for links 5–7, hand, gripper, and fingers, measured as a
  union of sampled MuJoCo geom AABBs;
- predeclared intrusion/boundary/transition/clear signed-distance intervals;
- at least 3 train, 3 calibration, and 5 held-out scenarios across at least two
  tasks, with fingerprint-disjoint splits and at least two repeats;
- train-only fitting, calibration-only threshold choice, and held-out reporting;
  grouped cross-validation and bootstrap use task/scenario rather than frames;
- negative-only calibration remains usable for the preregistered empirical FPR
  bound, but records an undefined calibration TPR; a held-out split without both
  frame classes makes the dissociation claim unsupported rather than triggering
  post-hoc horizon or scenario changes;
- probe comparisons against time, EEF pose, joint pose, action, task phase, and
  robot-state baselines;
- held-out online comparison against vanilla, always retreat, fixed step-3
  retreat, robot-state guard, and wall-presence-only guard, plus a genuinely
  online calibration-threshold risk–coverage sweep.

The corridor implementation is conservative: it unions world AABBs of all distal
collision geoms sampled along the nominal no-wall rollout and the bounding box that
bridges each geom between adjacent policy steps. This is a full-arm quantity but not
an exact continuous-time mesh union; the representation is saved in provenance and
should be described that way in the paper.

## Frozen design and authoring history

The tracked repository now contains the required two-task, 3/3/5 split under
`scenarios_p0/{train,calibration,heldout}/` and its committed
`configs/p0_core.json`.  The exact-revision gate selected LIBERO-Spatial task 2
alongside anchor task 0; the frozen scenario/config commit is `bc67488`.

The authoring commands below are historical/recovery instructions.  Do **not** rerun
them for the frozen P0 experiment; a new authoring attempt must use new output paths
and cannot silently replace these scenario bytes.

```bash
export CB_CHECKPOINT_REVISION=962318cec55ac10993ff0f5f43eda9a270b4c873
bash setup/submit_p0_authoring.sh all
```

The gate evaluates t1--t9 five times. The authoring job deterministically chooses
the highest-success non-anchor task (lowest task ID breaks a tie), pairs it with t0,
finds successful nominal states, identifies the moved black bowl, and creates all
11 scenarios plus `configs/p0_core.json`. Each authored scenario stores a nominal
geometry seed that must still complete the original task when capture replays it.

The completed authoring attempt was inspected with:

```bash
cat results/p0_runs/p0_authoring/authoring_report.json
find results/p0_runs/p0_authoring/previews -name '*.png' | sort
CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh preflight
```

The resulting server-authored scenario bytes and config were committed with:

```bash
git add configs/p0_core.json scenarios_p0
git commit -m "Freeze P0 scenarios and experiment config"
git push -u origin HEAD
```

See `docs/P0_HANDOFF_20260726.md` for the successful job IDs, audit outcome, SSH
workflow, and the exact next commands for the paper-facing run.

## Before submitting the paper-facing run

Resolve the cached checkpoint commit on a networked login node before the GPU job.
The compute job uses `local_files_only=True` and will fail instead of silently
downloading or falling back to `main`.

Run the fail-closed preflight first:

```bash
CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh preflight
```

Then submit all three dependent stages:

```bash
CB_P0_CONFIG=configs/p0_core.json bash setup/submit_p0.sh all
```

The config's `output_dir` receives capture/probe artifacts. The online guard output
defaults to `<output_dir>_online_guard`. Every stage refuses to overwrite an existing
directory. A run is usable only when its `complete.json` exists. Raw P0 directories
under `results/p0_runs/` are ignored because they contain large activations; after review,
copy only the final summary to a separately named tracked path and register it in
`results/manifest.json`.

For a staged/debug run, use `capture`, `analyze`, or `guard` separately and set
`CB_CAPTURE_DIR`/`CB_GUARD_OUT` explicitly. Do not use an underpowered smoke run in a
paper claim.

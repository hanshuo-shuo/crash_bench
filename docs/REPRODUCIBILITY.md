# Reproducibility and result handling

The tracked repository supports zero-GPU integrity checks and analysis of tracked
summaries. Full GPU reproduction needs the documented LIBERO/OpenVLA/OFT/openpi
environments, model checkpoints, and several ignored raw captures/logs.

## Zero-GPU checks

```bash
python -m pytest \
  tests/test_glass_recovery_eval.py::test_zero_gpu_v2_integration_collection_training_latch_cohort_eval_analysis -q
python -m pytest tests -q
python scripts/audit_repo.py
git diff --check
```

`audit_repo.py` validates document vocabulary, scenario fingerprints, manifest
paths, claim-ledger result references, immutable E14 semantics, E15 smoke
fail-closed inputs, and the semantic evidence contract for any future learned-
recovery promotion. The targeted integration creates synthetic schema-v2 data,
trains a tiny head on CPU, checks the runtime latch, selects a sealed fake
cohort, runs a fake environment, and performs source-cluster analysis. Neither
command reproduces GPU rollouts or recovers ignored activation/video assets.

## GPU evaluation discipline

Use the sbatch file listed in `EXPERIMENT_INDEX.md`, write a new output path, and
then add/update its manifest entry. Never overwrite a frozen result in place.
The scenario root and fingerprints, checkpoint, commit, repeat count, thresholds,
raw-log location, and script/sbatch must be recorded before a result is used in a
claim.

Tracked summaries are not substitutes for ignored raw assets:

- Slurm logs (`*.log`), rollout videos, intermediate renders, and hidden-state
  captures are ignored.
- `results/selfreport*/hidden.npz` and `meta.json` are needed to regenerate
  probe/shield analyses but are not committed.
- E13's promoted wall and glass matrices retain all 180 episode rows, exact
  effective instructions, checkpoint identity, commit, Slurm job IDs, and 20
  scenario fingerprints. The combined JSON and Markdown are derived from those
  tracked matrices; videos and Slurm logs remain ignored.
- E14 rendered frames, hidden states, branch arrays, and checkpoints remain in
  the ignored Quest run root. The promoted summary
  `results/glass_recovery_acceptance_smoke_20260809.json` records its commit,
  checkpoint revision, Slurm jobs, accepted state/scene hashes, attempt counts,
  and SHA256 checksums for the ignored placement and collection manifests.
- E14 is immutable v1 history. The current smoke path is E15/v2; reproducing the
  historical E14 job requires checking out its recorded commit rather than
  feeding v1 artifacts to current training/evaluation code.
- The tall-wall geometry is the treatment root. Low-wall detour results must use
  the separate `scenarios_detour_lowwall/` root.

See `results/manifest.json` and `results/claims_ledger.json` for evidence-level
provenance. The pre-refactor gap inventory remains available in the
[historical repository audit](archive/REPO_AUDIT.md).

## E15 P0-E provenance gates

`capture_glass_nominal_source_traces.py` writes one hash-checked manifest over
successful and failed no-glass source rollouts. Only successful rows can enter
candidate generation. Each successful row binds source-state index/hash,
checkpoint identity, rollout seed, EEF trace, relevant robot-body sweep, and
captured actions.

`prepare_glass_recovery_placements.py` requires that manifest by default.
`--legacy-straight-path` is an explicitly named historical/debug escape hatch
and is not valid for E15. The generator performs captured-action hazard replay,
initial-overlap and target-clearance checks, physical-scene deduplication, and
records a predeclared order that the collector must preserve.

`audit_glass_core_artifacts.py` requires `--read-only`, refuses outputs beneath
the source root, hashes every discovered evidence file, and appends only unseen
attempt IDs to `core_salvage_audit.jsonl`. Its `static_candidate_for_recollection`
field is not a migration decision: `direct_v2_promotion_allowed` is always
false.  For E14 it reconstructs job-scoped attempt IDs only when all historical
Slurm logs are supplied and their terminal rows/counts agree with the immutable
tracked summary.  `realign_glass_core_artifacts.py` then performs exact-H action
replay and a bounded oracle search plus independent recapture on a GPU/EGL node
without loading OpenVLA or writing into E14. Every oracle reset must reproduce
the simulator and controller hashes exactly. Full observation and
controller-used field drift are retained as diagnostics because E14 did not
save LIBERO's observable delay/history state; acceptance still requires the
searched config to succeed again in a fresh independent rollout. Use the exact
Pilot A commands in `GLASS_RECOVERY_V1.md`.

`run_glass_avoidability_frontier.py --print-commands` is a no-rollout preflight.
`--execute` runs the canonical collector independently at H=40,30,20,15,10,5
and writes per-attempt rows plus a frozen recommendation. Its input must contain
10--20 development candidates. Final held-out scenes must not be used in this
decision.

## E15 two-stage artifact contract

The wrapper never authors placements or collects candidates. Before `train`,
export all of the following from an accepted schema-v2 collection:

```bash
export CB_GLASS_RECOVERY_RUN_ROOT=results/glass_recovery_v2/e15_<run>
export CB_GLASS_RECOVERY_TRAIN_MANIFEST="$CB_GLASS_RECOVERY_RUN_ROOT/dataset/train.jsonl"
export CB_GLASS_RECOVERY_VALIDATION_MANIFEST="$CB_GLASS_RECOVERY_RUN_ROOT/dataset/validation.jsonl"
export CB_GLASS_RECOVERY_PRIMARY_PROTOCOL_SHA256=<64-lowercase-hex>
export CB_BASE_CHECKPOINT_REVISION=<immutable-OpenVLA-revision>
export CB_BASE_UNNORM_KEY=libero_spatial
export CB_GLASS_RECOVERY_H=<frozen-H>
export CB_GLASS_RECOVERY_TRAIN_SEED=17
bash setup/submit_glass_recovery_smoke.sh train
```

After training, record the checkpoint SHA, freeze the accepted evaluation
cohort and evaluation protocol without looking at final-held-out outcomes, then
export the additional inputs:

```bash
export CB_GLASS_RECOVERY_PLACEMENT_MANIFEST="$CB_GLASS_RECOVERY_RUN_ROOT/placements/placements.json"
export CB_GLASS_RECOVERY_TRAJECTORY_MANIFEST="$CB_GLASS_RECOVERY_RUN_ROOT/dataset/heldout.jsonl"
export CB_GLASS_RECOVERY_EVALUATION_COHORT="$CB_GLASS_RECOVERY_RUN_ROOT/sealed/heldout_cohort.json"
export CB_GLASS_RECOVERY_PROTOCOL="$CB_GLASS_RECOVERY_RUN_ROOT/sealed/evaluation_protocol.json"
export CB_GLASS_RECOVERY_EVALUATION_PROTOCOL_SHA256=<64-lowercase-hex>
export CB_GLASS_RECOVERY_CHECKPOINT="$CB_GLASS_RECOVERY_RUN_ROOT/train_seed_17/glass_recovery.pt"
export CB_BASE_CHECKPOINT=<local-OpenVLA-checkpoint-path>
bash setup/submit_glass_recovery_smoke.sh evaluate
```

The protocol must predeclare source-to-task and exact-anchor modes, the full
baseline matrix, rollout seeds, checkpoint SHA by training seed, and the exact
accepted pair IDs and manifest hashes. The evaluator rejects authored-but-
unaccepted IDs, leakage, missing oracle verification, or any checkpoint/Base/
schema/protocol/H mismatch. This layout is a command contract, not a learned
result.

If a future E15 manifest status is promoted to `frozen_learned_result`,
`verified_learned_result`, or `claim_ready`, its tracked JSON must use
`kind: glass_recovery_learned_result` and the exact audit keys implemented in
`scripts/audit_repo.py`: `accepted_cohort.{sha256,pair_ids}`,
`exact_replay_summary.{all_passed,n_pairs}`, split dictionaries under
`source_state_counts` and `family_counts`, zero-valued overlap counts under
`leakage_checks`, checkpoint/Base/dataset/protocol identity fields,
`evaluation.{primary_mode,conditions}`, a `source_state_sha256` cluster
analysis, and four denominator-bearing primary metrics (safe task success,
catastrophe, clean-control false intervention, and clean-control task
preservation). A code-only or fake-environment artifact intentionally fails
that promotion contract.

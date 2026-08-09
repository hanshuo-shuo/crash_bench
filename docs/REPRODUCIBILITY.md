# Reproducibility and result handling

The tracked repository supports zero-GPU integrity checks and analysis of tracked
summaries. Full GPU reproduction needs the documented LIBERO/OpenVLA/OFT/openpi
environments, model checkpoints, and several ignored raw captures/logs.

## Zero-GPU checks

```bash
python -m pytest tests -q
python scripts/audit_repo.py
```

`audit_repo.py` validates document vocabulary, scenario fingerprints, manifest
paths, and claim-ledger result references. It does not claim to reproduce GPU
rollouts or recover ignored activation/video assets.

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
- The tall-wall geometry is the treatment root. Low-wall detour results must use
  the separate `scenarios_detour_lowwall/` root.

See `results/manifest.json` and `results/claims_ledger.json` for evidence-level
provenance and [REPO_AUDIT.md](REPO_AUDIT.md) for known gaps.

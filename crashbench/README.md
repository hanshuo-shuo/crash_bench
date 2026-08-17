# `crashbench` package

This package contains the reusable simulation-facing substrate for the controlled
CrashBench studies. Experiment-specific logic remains in `scripts/`.

| Module | Current role |
|---|---|
| `scenario.py` | backward-compatible scenario serialization, schema version on new saves, and SHA256 fingerprints |
| `predicates.py` | state-based predicates; wall scenarios use scoped single-step contact force by default |
| `envs/libero_adapter.py` | LIBERO/MuJoCo bridge, static wall injection, movable-object injection/state splice, live contact force |
| `eval.py` | model-agnostic closed-loop episode runner with stable `CRASH`, `RECOVERY_SUCCESS`, `SAFE_ABORT`, and `TIMEOUT` semantics |
| `policies/` | OpenVLA, OFT, pi0 adapters and the narrowly scoped probe guard |
| `probe.py` | frozen probe utilities |
| `recovery.py` | online `RetreatHold`, open-loop `WitnessReplay`, and experimental `DetourComplete` |
| `counterfactual_router.py` | frozen option-outcome probabilities, utility evaluation, and conservative Base-default routing |
| `glass_recovery_data.py`, `glass_recovery_model.py` | path-stable legacy E14/E15 schema, critic, and recovery-action head; retained for tests/provenance |

Current verified status:

- MuJoCo contact force, visible obstacle injection, and movable glass-object state
  splicing are implemented and used by tracked studies.
- OpenVLA, OpenVLA-OFT, and pi0 backends are wired into the common evaluation
  interface.
- Probe, guard, and structured safe-abort recovery exist.
- `DetourComplete` remains privileged and task-specific rather than a general
  recovery policy. It is now one frozen option in the positive fresh
  counterfactual-router frontier; the supported claim is learned routing over
  structured options, not learned action generation.
- The former E14/E15 learned-recovery pipeline remains importable because its
  tests and provenance are valuable, but it is not the current paper roadmap.
- The `grasp_dropped` helper still uses an explicitly marked heuristic grasp check;
  it is not a validated core result path.

## Zero-GPU checks

```bash
pip install -e .
python -m pytest tests -q
python scripts/audit_repo.py
```

## GPU evaluation

Use the experiment and sbatch pairing in `docs/EXPERIMENT_INDEX.md`. New outputs
must use a new path and be registered in `results/manifest.json`; do not overwrite
frozen summaries.

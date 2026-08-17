# Reproducibility and data availability

The tracked repository supports zero-GPU integrity checks and analysis of frozen
summaries. Full rollout reproduction requires the documented OpenVLA/LIBERO/OFT/
openpi environments, model checkpoints, and ignored raw assets retained outside
Git.

## Zero-GPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```

The audit checks current paper framing, local Markdown links, scenario
fingerprints, manifest paths, frozen claim values, and E14/E15 provenance
boundaries. Tests include the reusable evaluation/controller code and a synthetic
glass-recovery integration; they do not turn that synthetic test into paper
evidence.

## Evidence levels

| Level | What is tracked | What it supports |
|---|---|---|
| claim-ready summary | result JSON plus manifest/claim-ledger entry | exact frozen paper claim within recorded scope |
| analysis-only artifact | summary/figure without complete raw capture | inspection and limited recomputation |
| ignored raw run | activations, traces, frames, videos, checkpoints, Slurm logs | full local/cluster reanalysis if still retained |
| archived narrative | dated report, protocol, or execution log | provenance only; not current project truth |

Never overwrite a frozen result. Write a new output path and record checkpoint
revision, code commit, scenario fingerprints, repeat count, calibration unit,
threshold, seed/nondeterminism policy, raw-log path, and analysis script in
`results/manifest.json` before promoting a new claim.

## Current main-line artifacts

### Fresh counterfactual outcome router (E16)

The paper-facing E16 package is self-contained in tracked, stable paths:

- `results/counterfactual_router_fresh_online_20260817.json`: compact frozen
  summary, hashes, jobs, cohorts, primary point, intervals, and wording bounds;
- `results/counterfactual_router_fresh_online_main_table.csv`: complete
  seven-method outcome and intervention-quality table;
- `results/counterfactual_router_fresh_online_n8_{analysis.json,frontier.csv}`:
  independent confirmation analysis and every predeclared frontier point;
- `results/counterfactual_router_fresh_online_n13_{analysis.json,frontier.csv}`:
  pooled descriptive analysis and every predeclared frontier point;
- four promoted `counterfactual_router_fresh_online_n{8,13}_frontier_*.png`
  figures.

The canonical interpretation is
[COUNTERFACTUAL_ROUTER_MAIN_RESULT.md](COUNTERFACTUAL_ROUTER_MAIN_RESULT.md), and
the frozen collection contract is
[FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md](FRESH_COUNTERFACTUAL_ROUTER_PROTOCOL.md).
The large branch logs, prompt rollouts, simulator states, source traces, and
router copies remain under gitignored `results/counterfactual_router/` run
roots. The promoted analyses permit complete table/frontier inspection without
those raw assets; re-executing rollouts still requires Quest-side state.

### Swept-corridor causal study

- Tall-wall treatment scenarios: `scenarios/`.
- Matched clearance controls: `scenarios_control/`.
- Frozen Base results: `results/{pilot_final,pilot_control_final,ood_control_final}.json`.
- Cross-policy summaries: `results/{base,oft,pi0}_*.json` and
  `results/path3_oft_summary.json`.
- Canonical commands: [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md), E1/E2.

The lowered d62 detour is a different scenario root,
`scenarios_detour_lowwall/`, with a different fingerprint. It must never be used
as the tall-wall treatment.

### Representation and behavior

- Committed Base/OFT probe checkpoints:
  `results/selfreport/probe_T5.npz` and
  `results/selfreport_oft/probe_T5.npz`.
- Frozen summaries: `results/selfreport*/probe_summary.json`.
- Confound analysis: `results/task_phase_confound/`.
- Online intervention: `results/intervention/`.
- Steering ablation: `results/steering/`.

The large `hidden.npz` and aligned `meta.json` files are gitignored. They are
required to refit held-out folds or regenerate all probe analyses.

### Glass extension

- Frozen dose response: `results/glass_prototype.json`.
- Glass probe summary: `results/selfreport_glass/probe_glass_summary.json`.
- Prompt safety–utility baseline: `results/careful_prompt/`.
- Scoped Oracle counterfactual summaries:
  `results/glass_recovery_pilot_{b,c}_scoped_20260812.json`.
- Derived frozen-checkpoint decision record:
  `results/glass_recovery_checkpoint_readiness_audit_20260812.json`.
- Paper-facing interpretation:
  [appendix/GLASS_SAFETY_UTILITY.md](appendix/GLASS_SAFETY_UTILITY.md).

The committed glass-probe directory does **not** contain raw hidden/meta or a
serialized deployable probe. A learned-timing glass experiment therefore starts
by recovering Quest captures or recapturing them and adding probe checkpoint
serialization/runtime loading.

## Readiness for the prioritized new experiments

| Experiment | Present locally | Missing prerequisite |
|---|---|---|
| frozen-router new-task replication | model code, frozen feature/option contract, lambda/rate grid, analysis | a new source-disjoint task-family cohort and task-compatible structured options |
| five-fold wall online guard | Base/OFT `probe_T5.npz`, scenarios, evaluation code | raw wall/OFT hidden/meta for fold-local refitting, or a new capture |
| OFT online guard | OFT probe summary/checkpoint and backend | fold-local calibration inputs and online replication protocol |

Threshold calibration for a new online result must be episode-level and isolated
inside the training/calibration fold. A frame-level AUC does not by itself define
a deployable operating point.

## Frozen appendix studies

- E12/P0 is a complete negative/indeterminate result. Its calibration and
  held-out captures contain no positive T-5 frames, and held-out Base already has
  0/50 crashes. Do not retune its held-out scenarios or present it as guard
  generalization. Protocol: [appendix/P0_EXPERIMENT.md](appendix/P0_EXPERIMENT.md).
- E13 contains all 180 episode rows, effective instructions, job IDs, commit,
  checkpoint identity, and scenario fingerprints. Protocol:
  [appendix/CAREFUL_PROMPT_EXPERIMENT.md](appendix/CAREFUL_PROMPT_EXPERIMENT.md).
- Other failed mechanisms and evidence gaps are listed in
  [appendix/NEGATIVE_RESULTS.md](appendix/NEGATIVE_RESULTS.md).

## Frozen E14/E15 legacy package

E14/E15 code and tracked results remain at historical paths because the package
has internal imports, tests execute those paths directly, and the manifest binds
their provenance. The old learned-recovery sequence is not a current run target.

Frozen facts retained by `scripts/audit_repo.py` include:

- E14: 109 attempts, 3 accepted admissions, immutable v1 commit and hashes;
- Pilot A: exact H=20 salvage/replay and independent Oracle recapture;
- broad Pilot B: 120 terminal candidate/H cells, no qualified horizon;
- scoped B/C: 3/15 certified accidents and 6/6 Oracle-timed Oracle successes on
  two development source states.
- checkpoint readiness: no threshold crossings in the six saved Oracle-condition
  episodes and validation gripper-sign accuracy below its frozen gate.

The complete dated protocol, repair notes, Slurm history, hash/restore details,
and illustrated audit are in
[archive/glass_recovery_20260812/](archive/glass_recovery_20260812/README.md).
Submission wrappers fail closed unless
`CB_ENABLE_LEGACY_GLASS_RECOVERY=1` is explicitly exported. That opt-in exists
for provenance diagnostics, not as a recommendation to run Pilots D/F.

Local ignored `results/glass_recovery_v2/` is not a complete archival copy: the
working tree has partial Pilot B arrays and a Pilot C evaluation JSON, while
several manifests, branches, and snapshots remain cluster-side. Treat the tracked
summaries and archived compact media as the self-contained repository evidence;
verify Quest retention before claiming raw reproducibility.

## Environment and current submission guide

See [setup/README.md](../setup/README.md). Model environments are deliberately
separate so OpenVLA, OFT, and pi0 dependency stacks do not overwrite one another.
GPU jobs must use a new result root and a clean, recorded commit.

The pre-refactor repository audit remains available at
[archive/REPO_AUDIT.md](archive/REPO_AUDIT.md).

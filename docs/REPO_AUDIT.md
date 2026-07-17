# Repository audit — 2026-07-17

This is the Stage-A, read-only audit for the repository-hygiene refactor. It
records observed state before any scenario or result provenance is changed. The
current scientific framing is a controlled causal diagnosis, not a broad
seven-hazard benchmark.

## Scope and current truth sources

| Location | Updated | Observed role | Current truth-source status |
|---|---:|---|---|
| `README.md` | 2026-07-16 | old project entrypoint and historical project design | stale; describes a 50-scenario benchmark and must be replaced |
| `PAPER_PLAN.md` | 2026-07-16 | detailed forward plan | useful source material; migrate to `docs/PAPER_PLAN.md` |
| `REPORT.md` | 2026-07-17 | result narrative | stale/internally inconsistent; preserve as historical record |
| `OVERVIEW.md` | 2026-07-17 | image-rich historical tour | historical; includes unsupported force-gap wording |
| `STATUS.md` | 2026-07-17 | progress board | historical; uses superseded headline framing |
| `PLAN.md`, `ROADMAP.md`, `STRATEGY.md` | 2026-07-16 | earlier plans and literature/strategy notes | historical, overlapping plans |
| `crashbench/README.md`, `crashbench/PHASE1.md` | 2026-07-16 | package and Phase-1 implementation notes | partially current but require wording/status repair |
| `results/ANALYSIS_*.md`, `results/WITNESS.md` | 2026-07-16/17 | per-result analyses | evidence notes, not a canonical paper narrative |
| `setup/README.md`, `cluster.md`, `GIT_WORKFLOW.md` | 2026-06-18/2026-07-16 | execution/cluster maintenance | operational references; `cluster.md` contains environment-specific facts |

There is no `docs/` directory and no explicit canonical documentation entrypoint
at audit time. The root has several competing plan/status documents.

## Scenario inventory

All checked scenario files specify `libero_spatial`, task id `0`, `max_steps=220`,
and a `T-5` horizon except the `wall_wide` scenario (`T-1`). Wall scenarios use a
scoped `contact_force` predicate (75 N, against `crash_wall`) and glass scenarios
carry their own movable-object predicates. `witness.npy` exists for the five wall
scenarios and does not exist for control/glass scenarios.

| Root | Count | Class / purpose | Geometry and witness state |
|---|---:|---|---|
| `scenarios/` | 5 | on-path wall treatment | `wide`, `d62`, `d70`, `d78`, `d85`; all have a witness |
| `scenarios_control/` | 21 | off-path clearance sweep | v3 twin/diverse/boundary plus v5 clear walls; no witness |
| `scenarios_glass/` | 10 | object-collision dose response | treatment/control pairs at f30/f40/f50/f60/f70; no witness |

### Confirmed geometry drift

`scenarios/env_collision__T5__libero_spatial_t0_wall_d62` was changed in-place in
commit `961571e` (2026-07-01). Its original committed wall was centered at
`[-0.119, 0.121, 1.08]` with size `[0.025, 0.08, 0.22]`; the current file has
`z=0.98` and size `[0.025, 0.08, 0.12]`, with a task-completing detour witness.
The reason is documented in its metadata. This makes its current ID ambiguous:
older wall/probe/guard results were generated on the tall-wall treatment, while
the current file is a low-wall detour variant. The high wall can be restored
exactly from commit `02895e6` (the predecessor of the lowering commit), and the
current low-wall artifact must receive a distinct scenario ID/root.

## Results and evidence mapping

| Evidence | Primary result file(s) | Analysis / generating code | Audit status |
|---|---|---|---|
| nominal bridge | no committed `m1_nominal_gate.json`; older aggregate cited in docs | `scripts/nominal_task_gate.py`, `setup/m1_nominal_gate.sbatch` | source result is missing from tracked `results/` |
| wall pilot / treatment | `pilot.json`, `pilot_final.json`, `base_walls_matched.json` | `run_pilot.py`, `phase1_ood_control_v5.py` | frozen result rows, multiple versions |
| corridor control | `pilot_control*.json`, `ood_control*.json` | `phase1_build_ood_control*.py`, `phase1_ood_control_analysis.py` | final analysis: full transition plus clear regime |
| cross-policy | `base_*_matched.json`, `oft_*.json`, `pi0_*.json`, `path3_oft_summary.json` | `path3_oft_compare.py`, `run_pilot_*` sbatch files | completed; base/OFT BORDER overlap exists |
| wall probes | `selfreport*/probe_summary.json`, `probe_T5.npz` | `probe_selfreport*.py`, analysis | completed for OpenVLA/OFT; partial/ambiguous for pi0 |
| intervention | `intervention/{episodes,summary}.json` | `phase3_intervention.py` | completed online base-wall result |
| offline shield sweep | `shield/{sweep,summary,horizon}.json` | `probe_shield_sweep.py`, `probe_horizon_relabel.py` | frozen capture analysis, not an online midpoint test |
| glass dose response | `glass_prototype.json`, `selfreport_glass/probe_glass_summary.json` | `phase2_glass_prototype.py`, `probe_glass_*` | completed, but old selected-band headline is deprecated |
| steering | `steering/{sweep,summary,diag}.json` | `phase3_steering*.py` | completed negative result |
| witnesses / detour | `witness.json`, `phase2_task_witness/summary.json` | `phase2_witness.py`, `phase2_task_witness.py` | safe abort complete; d62 task completion only on low wall |
| negative attempts | `nowall.json`, `obj_collision.json`, `grasp_instability.json`, `grasp/*`, `phase2_recon/recon.json` | corresponding Phase-2 scripts | retained; not a broad completed benchmark |

`headline_suite.json` encodes the deprecated 98.3% selected-band category average
(`f50`–`f70` glass only). It must remain as a historical frozen file but cannot be
used by current documentation or claims.

## Provenance gaps

1. No result manifest, checksums, scenario fingerprints, or claim ledger exists.
2. Most result JSON files lack commit, checkpoint, scenario fingerprint, seed,
   raw-log, and sbatch provenance.
3. Gitignored raw assets include Slurm `*.log`, rollout videos, intermediate
   renders, activation dumps (`hidden.npz`, `meta.json`), and several generated
   scenario roots. Those assets are not recoverable from the tracked repository
   alone; summaries and selected figures are tracked.
4. `results/m1_nominal_gate.json`, `results/grasp/run.json`, and the optional
   wall-directed-braking output referenced by scripts are not tracked.
5. Existing scripts frequently write fixed result paths directly, including
   `headline_suite.py`, `run_pilot.py`, Phase-1/2 runners, probe analyses,
   steering, shield, and intervention scripts. These can overwrite a frozen file
   without an explicit opt-in.

## Document and numerical inconsistencies

1. Root `README.md` still presents 50 scenarios, seven categories, 150 trials,
   unfinished scale-up, and a benchmark headline. It is inconsistent with the
   current controlled-study scope.
2. `REPORT.md` contains the deprecated 98.3% headline and the temporary sentence
   “It still might just be ood still.” It also says all architecture BORDER hits
   are non-overlapping; `path3_oft_summary.json` shows base=4/11, OFT=3/11, pi0=3/11,
   and only pi0's documented hit set is distinct from the OpenVLA family.
3. `REPORT.md`, `OVERVIEW.md`, Phase-1 analysis/docs, and an authoring script state
   an unsupported force gap of real hits >=150 N versus grazes <=44 N. The frozen
   result rows include crashes as low as about 69–109 N, so that claim cannot be
   retained without a force-trace sensitivity recomputation.
4. `results/ANALYSIS_shield.md` says “509 imminent”; the frozen shield summary
   reports 53 positive/imminent frames and 4,400 negative frames. 509 is a
   pre-crash-frame count, not the imminent count used in the frame statistics.
5. `results/ANALYSIS_steering.md`, `REPORT.md`, and steering figure code describe
   a 90-degree/orthogonal relation. The diagnostic instead measures a norm ratio:
   `0.742 / 7.688 = 9.65%`; it does not measure an angle.
6. Two impact means are valid for different evidence: approximately 250 N is the
   frozen wall pilot/corridor treatment mean, whereas 321.7 N is the online
   intervention baseline. They are presently mixed in narrative claims.
7. The online guard evidence is 15/15 -> 0/15, 321.7 N -> 0 N, benign 0/22 at
   approximately -0.422. The `[-0.7, 2.7]` window and its `1.0` midpoint are an
   offline frozen-capture sweep; the midpoint has not been independently verified
   online.

## Code audit observations

- `crashbench/scenario.py` retains the original seven-category enumeration and
  has no schema version or fingerprint helper. It is backward-compatible through
  optional fields and is suitable for an additive schema/version/fingerprint API.
- `crashbench/eval.py` checks crash before task success in a step, resets policies
  that expose `reset`, attributes only predicate type names, and reports the
  simulator-wide `peak_force`. It does not persist predicate-scoped peak force,
  trigger step/contact body, or threshold in `EpisodeResult`.
- `crashbench/predicates.py` defaults to `hold_steps=1`, but its docstring calls
  sustained contact “FINALIZED” and describes it as a real-collision criterion.
  The present scenario files use a single-step threshold; the wording is stale.
- `crashbench/recovery.py` presents `DetourComplete` as a generally usable online
  task-completing controller although comments elsewhere say it is not stable.
  `WitnessReplay` is specifically open-loop from the recorded start state, while
  `RetreatHold` is the online safe-abort controller.
- `GuardedPolicy` needs an explicit, narrow scope statement: OpenVLA base,
  wall-trained probe, on-path wall crash mode, structured retreat recovery; it
  is not a general collision-avoidance system.

## Script / submission audit

The repository has 33 Python scripts and 42 sbatch/shell setup scripts. Naming
mixes `phase`, `path`, `pilot`, `probe`, `diag`, `recon`, and `prototype`; an
experiment index is required before any physical reorganization. Existing sbatch
references must stay valid. `prepare_m0_geometry.py` and
`phase3_intervention_expanded.sbatch` already describe the desired d62 split but
the repository was not brought to that state.

## Completed, negative, blocked, and stale work

- Completed: wall corridor control, three action-head behavioral replication,
  OpenVLA/OFT probes, partial pi0 probe evidence, wall directed-braking
  diagnostic, narrow base-wall guard intervention, glass dose response, and
  safe-abort witnesses.
- Negative/limited: edge-bowl OOD/non-contact, kitchen low nominal competence,
  cookies obstacle too short, state-reset grasp instability, final-readout
  activation steering, tall-wall task detours, wall-to-glass probe transfer, and
  current wall-probe coverage for gradient/BORDER crashes.
- Stale: broad benchmark plan, 98.3% headline, universal recovery wording,
  and old authoring-state TODOs.

## Stage-B–D modification plan

1. Create canonical `docs/` truth sources and archive or label competing plans.
2. Rewrite the root README for the controlled diagnosis, and repair documented
   numerical/scoping errors without changing frozen evidence.
3. Restore tall d62 from Git history; preserve the current low-wall version in a
   distinct `scenarios_detour_lowwall/` ID with parent/provenance metadata.
4. Add scenario fingerprints, `results/manifest.json`, `results/claims_ledger.json`,
   a script index, overwrite protection, and zero-GPU audit/test coverage.
5. Do not delete tracked experimental results. Keep deprecated outputs explicitly
   marked rather than silently rewriting their historical values.

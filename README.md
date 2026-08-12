# CrashBench

Working paper title: **Decoded but Not Routed: Causal Diagnosis and Closed-Loop
Repair of VLA Collision Failures**.

CrashBench studies a concrete VLA failure mode: a visible obstacle enters the
robot's action-swept corridor, collision risk becomes readable from the frozen
representation, yet that information is not routed into a safe action. A simple
**risk-readout → controller** interface closes this loop in a scoped wall setting.

## Paper spine

1. **Causal localization.** Moving the same visible wall into the executed swept
   corridor changes OpenVLA from 0/33 clear off-path crashes to 15/15 on-path
   crashes. The matched behavioral pattern also appears in OpenVLA-OFT and pi0.
2. **Representation–action gap.** Collision imminence is linearly decodable from
   frozen OpenVLA and OFT hidden states (T-5 AUC 0.998 and 0.903), while the final
   pre-impact window contains no EEF retreat in 25/25 wall episodes.
3. **Closed-loop routing repair.** A probe-gated `RetreatHold` controller changes
   15/15 crashes to 0/15, 321.7 N mean peak force to 0 N, and fires on 0/22 benign
   rollouts. Direct final-readout steering remains at 100% crash, showing that a
   detector direction is not automatically a controller direction.
4. **Safety–utility frontier.** Stopping is not task completion. Hazard-specific
   prompting and `RetreatHold` can reduce collision or safe-abort, while exact-state
   glass counterfactuals show that task-completing continuations physically exist.
   The glass result is an Oracle upper bound, not a learned-recovery claim.

The contribution is deliberately scoped to manipulation collision, swept-corridor
causal controls, and an explicit routing interface. It is not a universal VLA
safety benchmark or a general recovery policy.

## Headline evidence

| Question | Tracked result | Paper role |
|---|---:|---|
| Does path intrusion cause the failure? | wall 15/15 on-path vs 0/33 clear; glass 30/50 on-path vs 0/50 matched off-path | causal localization |
| Is imminent risk represented? | OpenVLA T-5 AUC 0.998; OFT 0.903; glass-specific probe 0.944 | risk readout |
| Is that signal used by the action? | 0/25 final-window retreat; toward-wall command increases in 22/25 | representation–action gap |
| Does explicit routing repair collision behavior? | 15/15 → 0/15 crash; 321.7 N → 0 N; 0/22 benign fires | closed-loop intervention |
| Does direct representation steering suffice? | 100% crash at every tested alpha | mechanism ablation |
| Does safe behavior complete the task? | hazard prompt: 0/15 treatment task success; wall guard: safe-abort | safety–utility boundary |

Exact claim wording, denominators, and limitations live in
[docs/CLAIMS.md](docs/CLAIMS.md).

## Current decisions

- The wall causal-diagnosis → representation gap → routing-repair line is the
  paper primary.
- Glass recovery Pilots D/F are **not** current run targets. The frozen learned
  gate has no deployable operating point, and the learned action head has not
  passed its prerequisite Oracle-timing test.
- Glass appears as a dose response, a cross-hazard transfer limit, and an
  exact-state Oracle counterfactual that motivates task-completing routing.
- Historical B/C/D/E/F execution plans, Slurm notes, restore/hash details, and
  preliminary reports are retained under [docs/archive](docs/archive/README.md),
  not presented as the paper timeline.

The prioritized experiment queue is in [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md).

## Repository map

```text
crashbench/                 reusable scenario, policy, probe, and controller code
scenarios*/                 frozen wall, control, glass, and appendix scenario roots
results/                    tracked evidence summaries and provenance ledgers
scripts/                    analysis and experiment implementations
setup/                      current environment and submission guide
docs/                       current paper truth sources
docs/appendix/              frozen protocols, negative results, supporting evidence
docs/archive/               superseded plans, logs, reports, and legacy snapshots
legacy/                     superseded executable orchestration helpers
```

Some legacy code and result files remain at their historical paths because the
manifest, tests, and provenance records bind those paths. Their status is indexed
in [docs/SCRIPT_INDEX.md](docs/SCRIPT_INDEX.md); path stability does not make them
current paper entrypoints.

## Zero-GPU verification

```bash
pip install -e .
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py
git diff --check
```

These checks validate tracked metadata, scenario fingerprints, evidence values,
document framing, and local links. They do not reproduce GPU rollouts or restore
ignored hidden-state/video assets.

## Start here

- [Current state](docs/CURRENT.md)
- [Paper plan](docs/PAPER_PLAN.md)
- [Claim ledger](docs/CLAIMS.md)
- [Experiment index](docs/EXPERIMENT_INDEX.md)
- [Reproducibility and data availability](docs/REPRODUCIBILITY.md)
- [Appendix index](docs/appendix/README.md)
- [Legacy archive](docs/archive/README.md)

No archival citation or license file is currently supplied; add both before an
external release.

# CrashBench

CrashBench is a controlled-study artifact for diagnosing a specific VLA failure
mode: a visible obstacle enters the robot's action-swept corridor and the policy
continues into a collision.

## Current framing

The paper line is **diagnose → localize → explain → exploit**.

- **Diagnose:** visible on-path walls cause collisions.
- **Localize:** holding wall appearance fixed and moving it through the swept
  corridor separates path intrusion from obstacle novelty alone.
- **Explain:** in parts of the OpenVLA family, collision imminence is **decoded
  but not used** — represented but not read out into safe action.
- **Exploit:** a wall-trained probe can trigger a structured retreat controller
  that prevents collision in a deliberately narrow OpenVLA-base/on-path-wall
  setting.

This is not a completed seven-hazard benchmark or a claim of universal VLA
safety failure.

## Core findings from tracked evidence

- On the wall study, base OpenVLA has 15/15 on-path crashes and 0/33 crashes in
  the clear off-path regime; the transition region is graded.
- The on-path versus clear-off-path behavior is reproduced across OpenVLA base,
  OpenVLA-OFT, and pi0 action-head architectures (5/5 versus 0/10 per model in
  the matched geometry summary).
- Frozen linear probes strongly decode wall-collision imminence for OpenVLA base
  and OFT, while measured motion does not show sustained braking. pi0 evidence
  is partial and tap-dependent.
- In the scoped base-wall intervention, a probe-gated RetreatHold controller
  changes 15/15 crashes to 0/15 and 321.7 N mean peak force to 0 N, with 0/22
  benign guard fires at the committed online threshold.
- Glass supplies a separate f30–f70 dose-response study. The old selected-band
  cross-category average is deprecated and is not a project headline.
- In E13, explicitly naming the hazard reduced treatment crashes relative to
  the original task-only instruction (wall 15/15 to 13/15; glass 9/15 to 2/15),
  but yielded 0/15 treatment task successes for both hazards. The observed
  benefit is conservative stopping, not task-completing avoidance.
- The E14 recoverable-glass acceptance smoke found three exact-state scenes in
  which Base and the fixed careful-prompt OpenVLA crash while the existing
  oracle safely completes the task. This is an environment-validity existence
  result, not yet a split-balanced recovery-training result.
- Final-readout activation steering and wall-to-glass probe transfer are
  negative results retained in the repository.

## Layout

```text
crashbench/                 core scenario, evaluation, predicate, policy, recovery code
scenarios/                  frozen tall-wall treatment scenarios
scenarios_control/          fixed-appearance off-path clearance controls
scenarios_glass/            glass dose-response scenarios
scenarios_detour_lowwall/   separately identified low-wall detour existence demo
results/                    frozen summaries, manifest, and machine-readable claim ledger
scripts/                    generation, evaluation, analysis, and audit scripts
setup/                      environment and Slurm submission scripts
docs/                       current state, paper plan, claims, experiment index, audit
```

## Reproduce zero-GPU checks

```bash
pip install -e .
python -m pytest tests -q
python scripts/audit_repo.py
```

The checks validate tracked metadata, manifests, fingerprints, and current
documentation. They do not rerun LIBERO/VLA evaluation.

## Run GPU evaluation

Create the documented OpenVLA/LIBERO environment in `setup/`, choose the
experiment and submission file in [docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md),
and write a new output path. Do not overwrite a frozen result. Before using a
new result in a claim, add its scenario fingerprints and execution provenance to
`results/manifest.json`.

## Limitations

The main causal and online intervention evidence is one task, a wall family,
and OpenVLA base. The structured recovery controller is not general collision
avoidance. The d62 task-completion witness is a low-wall existence demo, not a
general task-completing recovery result. Raw logs, videos, and some hidden-state
captures are intentionally gitignored; tracked summaries document the gap.

## Documentation

Start at [docs/CURRENT.md](docs/CURRENT.md). See the
[paper plan](docs/PAPER_PLAN.md), [claim ledger](docs/CLAIMS.md),
[experiment index](docs/EXPERIMENT_INDEX.md), [reproducibility notes](docs/REPRODUCIBILITY.md),
and [script index](docs/SCRIPT_INDEX.md).
The current E14 smoke is summarized in
[results/ANALYSIS_glass_recovery_acceptance.md](results/ANALYSIS_glass_recovery_acceptance.md).
Superseded planning and narrative snapshots are isolated in
[`docs/archive/`](docs/archive/README.md).

## Citation and license

No archival citation or license file is currently supplied. Add them before an
external release.

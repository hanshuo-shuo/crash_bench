# From observed rescue to repeatable intervention value

User direction, 2026-09-13: advance the ICLR paper using the supplied advisor feedback,
prioritize a coherent scientific story, and run all research experiments on Quest.
This step implements the feedback's existing-data test before deciding on new rollouts.
It does not open D8, change a controller, fit a model, or rewrite historical results.

## Scientific question

How much of an observed statewise selection gain survives evaluation on a separate
execution block, and how does the answer differ between real recovery, arbitrary
same-policy pseudo-options, and deadline-adjacent speed changes?

We will not presume that all gains disappear. The paper must explain retained Detour
rescues as well as disappearing or reversing effects. A useful outcome is a sharper
distinction between apparent opportunity, repeatable opportunity, and deployable value.

## Fixed existing panels

- OpenVLA / Detour: 16 sources, 48 episodes, A/B two repeats per arm; horizons 220/440.
  Include the one shared pre-candidate accident as one episode, without inventing repeats.
- pi0 / candidate Refresh: 8 historically selected sources, 18 anchors, A/B four repeats
  per arm; horizons 100/200 from continuous trajectories.
- pi0 / selection retest: 12 sources, 24 anchors, A/B four repeats per arm; horizon 100.
  This panel is an additional negative control and does not independently replicate the
  candidate panel's physical sources. Do not pool source counts across studies.

Inputs are the tracked terminal records, anchors/panel, shared-prefix outcomes and
historical freeze files. No activation files, D8 outcomes, or new simulator calls.

## Analyses fixed before this computation

1. Real full-repeat reference: at each declared horizon select R when A mean safe task
   success exceeds Base; ties select Base. Evaluate the identical choice on A and B.
   This is a new descriptive success-only reference; historical frozen references remain
   separately identified, including their different selection targets.
2. Historical references: read existing freeze choices without modification and score
   their A/B success, accident and normal-task consequences.
3. Real one-shot comparison: choose from the first A repeat and evaluate the first B repeat.
4. Base-vs-Base pseudo-options: assign first/second A Base executions to pseudo-arms 0/1;
   freeze the winning index, then evaluate first/second B Base executions. Repeat with
   the orientation swapped as a declared sensitivity, never as additional independent data.
   Compare this primarily with the real one-shot result, which has the same per-arm count.
5. Where four repeats per phase exist, additionally split Base repeats 0:2 vs 2:4 into
   two pseudo-arms; compare against real arms using their first two repetitions. This
   keeps per-arm evaluation counts matched. These are diagnostics, not deployable choices.
6. Preserve all sources and conditions. Report all-condition, treatment, controls and
   fitting/calibration subsets. First average anchors within physical source, then sources.
   Report both sources with any positive cell and source-macro net contribution.
7. Freeze each short/long A choice and read its complete available success/accident curve;
   no choosing a new headline deadline. Explicitly separate same-budget A/B change from
   cross-budget change. Record normal success loss and new accident separately from nets.

Uncertainty: 5,000 paired source-cluster bootstrap draws, seed 20270913. These are
descriptive development intervals; all original outcomes have already been exposed.
No test-set or population guarantee. Report the support behind zero estimates; do not
interpret identical recorded policies or zero-width bootstrap intervals as safety proof.
Pseudo-option gains are not subtracted from real gains as a de-noised causal effect.
A/B block changes can include systematic runtime differences, not only IID noise.

## Execution and outputs

CPU Slurm short, account p33100, 2 CPUs, 8 GB, 15 minutes; clean tested published
commit, unique results/repeat_value/<commit>_<job>/ output. Full local input hashes,
Slurm/runtime provenance, aggregate and source tables, deadline curves, and PNG/SVG
figures. Synthetic software unit tests may run locally; scientific analysis runs on Quest.

## How results change the story

- A sizeable repeat-disjoint change plus misleading pseudo-option gains motivates the
  proposed A/B/C study. Before new data, specify which phenomenon that study must settle.
- Small changes with retained true gains shift the story toward distinguishing stable
  opportunities from weak deployment selection; avoid an exaggerated inflation headline.
- Absent opportunity throughout shifts the next question toward skill/state support.

The feedback's 5 percentage-point threshold is a proposed resource-allocation aid, not
an automatic publication gate. The judgement uses direction, magnitude, source support,
deadline dependence, and what the next experiment can actually distinguish.
No automatic large training, benchmark expansion, or independent cohort is started by
this analysis entry. A bounded next study, if justified, gets its own concrete contract.

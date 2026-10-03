# Dated supplemental diagnostics and reporting deviation

On2026-10-03 at20:56UTC, before any readout fitting but after some policy
collection, the user requested nuisance diagnostics using the existing data.
See `experiments/reach_avoid/nuisance_protocol.json`: privileged nominal aperture,
privileged two-view target pixel fractions, and their combination. All-variant
and numeric-gap populations are reported separately;20mm for open/off-target
controls is a bookkeeping parameter, not global clearance. The existing6/2/4
train/validation/test assignment and train-slit/test-cage holdout remain fixed.
The same fitted models are evaluated on the retrospective defined-failure subset.
UNKNOWN coverage and score distributions are reported without accuracy.
Constructor ID is constant in fitting; there is no learned cage-category mapping
and no baseline fitted on test-cage labels.

At21:01UTC the user approved a reporting-plan deviation: make every one of the
four test-layout results and within-layout paired differences the main descriptive
evidence. The original frozen protocol, analysis code and2000-draw bootstrap
output remain intact; the bootstrap moves to an exploratory appendix. Four
clusters do not support reliable nominal coverage or a significance claim.
In the conventional independent, nondegenerate four-paired-sign setup, exhaustive
one-sided sign-flip arithmetic has16 assignments and minimum p=1/16=0.0625.
This does not establish the test assumptions for this constructed dataset; no
p<0.05 claim is made. See `reporting_amendment.json` for the recorded change.

Use the name **SAFE-style initial-state failure probe**. It uses t=0 features and
later full-T failure targets and is not a faithful reproduction of original SAFE
temporal detection. Do not claim to outperform SAFE. All feasibility readouts
and labels in this round stay at t=0. Later outcomes define targets/subsets, not
dynamic feasibility labels. No deadlock-state extrapolation, temporal training,
additional prompt/seed rollout or AEGIS rescue experiment is part of this round.

The supplemental synthetic regression flips held-out labels and checks that
selected heads and scores remain unchanged, while retaining48 UNKNOWN states
and excluding an invalid endpoint from failure-subset evaluation. This regression
does not fit the real study data. Original primary numerical outputs are hashed
before and after the supplement for separate provenance.

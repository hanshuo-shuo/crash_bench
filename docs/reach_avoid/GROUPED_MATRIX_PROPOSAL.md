# Proposed next checkpoint: genuinely grouped initial-state readout

Not submitted by this document. Requires the native visibility/layer hook audit
and a concrete resource checkpoint. The completed baseline now contains actual
safe pi0.5 success and safe failure at a witnessed feasible initial state.

Use twelve fresh controlled spatial layouts, not the two construction groups or
millimetric official init perturbations. Target x in {-0.32,-0.20,-0.08} m,
target y in {-0.46,-0.32} m, and goal x in {-0.12,+0.12} m with goal y=+0.28 m.
All use official init0 for other poses, unchanged clear wine xy=(.30,-.45), and
the same rigid objects, action interface, T=300 and RA-SEP-1 guard. Invalid initial
geometry stays invalid. No post-outcome replacement or position tuning.

A fixed seed741 permutation assigns six scene groups to train, two to validation
and four to test. Every paired construction/gap/control/seed/frame/render of one
layout keeps that assignment. Slit construction is used for primary training and
validation; cage is the held-out construction in test groups. Other cells are
retained for descriptive diagnostics, not threshold/layer selection. Both remain
one sufficient separator proof family, so this is constructor transfer, not
independent-mechanism or train-two/test-third generalization.

Collect seven fixed variants per construction (.020,.028,.034,.060,.180 m,
open-roof, irrelevant), preserving the existing rule: certificate or complete safe
witness; every other case UNKNOWN. Thus 168 prospective state cells, expected120
known and48 unknown if construction coverage transfers; do not assume that result.
UNKNOWN remains in coverage and width tables, excluded only from binary accuracy.
The broad current bracket means near-boundary decoding remains untested even if
separated cases are decoded well.

At initial state, freeze pi0.5 and extract its own unprojected visual-tower output,
projected image tokens, intermediate residual layers and normalized final prefix.
Use actual same-view RGB for a separately pretrained general-vision encoder when
its exact weights and preprocessing have been staged. It is a matched external
baseline, not evidence of information destruction. Save native layer-hook audits.
A faithful KNOWS-style target-attention statistic requires actual attention and
aligned target localization; a red-pixel/visibility proxy must never be named as
KNOWS. Mark missing baselines explicitly rather than relabel an easier substitute.

Fit standardized training-only PCA (at most32 dimensions), linear logistic and
one-hidden-layer32-unit MLP heads, four regularization values each; select by
validation log loss. Compare every representation with the same splits and fitting
budget. Choose a primary layer on validation only; test neither tunes nor selects.
Learning curves use fixed training-group prefixes of sizes2,4,6. Bootstrap entire
test scene groups (2,000 resamples); four groups still imply wide exploratory
uncertainty, not population or benchmark-level precision.

Run fixed pi0.5 from each independently labeled state to its first protected
violation, safe completion or T=300. Preserve collision versus safe timeout.
Primary conditional estimand: certified infeasible versus witnessed feasible
among policy-failed runs. A SAFE-style failure probe predicts full-horizon policy
failure, not only short-horizon collision. Compare its scalar alone with the scalar
plus feasibility readout at the same initial decision time. Obtain training scores
by group cross-fitting; never train a stacker on in-sample label predictions. Use
paired group bootstrap for incremental differences. If either conditional class
is absent, report not estimable rather than tune scenes or horizons to create it.

No AEGIS runtime/rescue claim belongs to this initial-state experiment. Runtime
trigger, state-specific certificates/witnesses, alternative menu and safety audit
remain separate future work. No paid API or base-model training is proposed.

Approximate resource proposal: six isolated CPU array tasks, each two layouts,
1core/8GiB/30min, at most three concurrent; expected about147 core-min total from
24.5min/28-state pilot (cap180 core-min). GPU extraction plus120 expected known
state rollouts approximately45-70 A100-min, split into separate immutable jobs
of at most30min, at most one active GPU. This is a scale-up and needs explicit
checkpoint review before submission; current tool preparation is not authorization.

Pre-results analysis amendment (2026-10-03, before any grouped GPU rollout): an
independent statistical audit identified that global validation selection, fitted
using all train labels, could indirectly leak a held training group's labels into
its own cross-fitted feasibility score. The implementation now repeats both
representation and head selection inside each LOGO fold, fitting on only the other
training groups and selecting on the fixed validation groups. The final model
still uses all training groups and fixed validation groups; test remains untouched.
Every fold records its chosen representation/head and the complete selection table.

Undefined policy outcomes (including illegal-command termination at an otherwise
valid initial state) remain in the initial feasibility analysis but are excluded
from SAFE fitting, failure metrics and failure-conditioned stackers. Counts and
case IDs are retained. Explicit domain and endpoint checks require safe timeouts
to reach all300 execution commands and safe completions to retain safety. No
partial episode is relabeled as a timeout. These amendments do not change scenes,
labels, physical executions, split assignment, or the declared estimands.

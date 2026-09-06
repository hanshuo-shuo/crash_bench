# Paper-oriented investigation — 2026-09-06

The user authorized free exploration toward a paper, problem finding and a
comprehensive report after rounds one and two. Preserve their frozen outputs.

Scope: literature comparison using primary papers; read-only subgroup, risk/benefit,
source influence, utility and neutral-control analyses on existing D5 train/development;
no D8 tuning or new confirmation. Report uncertainty and distinguish verified defects
from hypotheses. A small engineering probe is justified by terminal disagreement
between theoretically neutral Base/Refresh controls in the saved data.

Probe fixed before execution: first lexicographic train physical source in each
of the two existing tasks; anchor after 5 actions, observation delay 3, conditions
fresh_control, matched_buffer_control, stale. Save a new exact bundle and mechanism
queue for each anchor. Restore the same initial policy continuation before each
condition to remove cross-condition policy-RNG drift. Execute Base twice and Refresh
twice from each new bundle, max 100 steps; log observations, policy RNG, actions,
states and force. These are authored training-source engineering anchors, not
replays of the unavailable serialized historical D5 anchors. No outcome-based
replacement. At most 24 branches, one A100, 8 CPU, 48 GB, 30 minutes on Quest.

Compare Base-A/Base-B, Refresh-A/Refresh-B and Base-A/Refresh-A trace divergence;
nonzero differences in nominally neutral controls require explanation. Never
label a failure of this probe as scientific recovery evidence. Store all outputs
in new results/paper_review/<commit>_<job>/, retain input and output hashes.

Deliver a Chinese comprehensive report, claim/evidence matrix, verified related
work, concrete reviewer risks, an actionable paper structure and prioritized
experiments. No automatic permanent NO-GO or claim of publication readiness.

# FD1000 final acceptance — V4 rejected, 896 review-only rows

## 2026-10-16 user decision: V4 is used as development data

The user decided that the V4 review need not be strict. The 896 rows below are
used as development data for verifier training under
[the generative verifier plan](../docs/generative_verifier/PLAN.md). The failed
shortcut gate stays disclosed and is reported as a baseline. A validation-only
stratification (the locked test was not used) places the readout signal in
EXIT × checkpoint 0 (AUROC 1.000, 32 rows) and leaves ENTRY × checkpoint > 0
at 0.636 (84 rows). That readout has no target or goal features, so the signal
most likely reflects box placement rather than containment reasoning. All
records below are unchanged history.

## Final acceptance complete; collection closed and training not started

Quest CPU audit **3657699 COMPLETED/0:0**, 9m48s on qnode0026, root
`20261015T042328Z_audit-constraints_75be81f8f066`, used published source
`75be81f8f0664e89ece893a16ae3659b2c431452`. It independently rechecked the
67 original traces, 494 raw matched units, selected image hashes, regenerated
candidate explanations and the fixed source split. The selected set exactly
matches the local selection frozen before the first V4 model fit.

**The final dataset is rejected for scientific training.** Evidence and selected
visual-premise checks pass, but the unchanged maximum joint-shortcut AUROC
threshold of 0.75 fails:

| Fixed validation readout | AUROC |
| --- | ---: |
| Image statistics only | 0.500000 |
| Task/rules only | 0.500000 |
| RGB/color/rule linear | **0.781576** |
| RGB/color/rule forest | 0.738824 |
| Color statistics/rule linear | **0.779948** |
| Color statistics/rule forest | 0.690213 |
| Color centroids/rule linear | 0.685221 |
| Color centroids/rule forest | 0.638346 |

Image-only and text-only chance performance is guaranteed by paired input/label
balance; it does not establish absence of shortcuts in their combination. Test
model predictions remain unscored. No post-fit reselection, extra diagnostic run,
construction revision, relaxed threshold, finalizer or verifier training followed.
All collection and audit jobs in this work are terminal. API spend remains
**$0.05882376 of $20**, with no new API request during final acceptance.

The review-only subset contains **896 rows: 448 witnessed positives and 448
conditional negatives**, organized as 224 complete four-row checkpoint units
from **64 original trajectories / ancestry groups, 12 task compositions and four
families**. There are 512/192/192 rows and 35/14/15 original source groups in
train/validation/test. Each source contributes at most six selected checkpoints.
Four factor cells have equal counts in every split. The broader visibility bank
has 293 reviewed units with 39 whole-unit exclusions; all 224 selected units have
image-bound visible-premise records. This is Codex review, not independent human
review. The user reduced the size requirement rather than relaxing any evidence,
visibility, split, coverage or shortcut condition.

Published decision and final selected-image record:
`reviews/fd896_v4_final_decision.json` and `reviews/fd896_v4_final_visual.json`.
The latter binds the proposed dataset digest and every selected unit's PNG hashes;
its completion is not overall acceptance. Accepted training rows and new verifier
training runs remain **zero**.

Local delivery:

- `results/verifier_dataset/factorial_v4_final_896_20261015/FINAL_DECISION.json`
  and `AUDIT_DELIVERY/`: 14 hash-verified audit/provenance files.
- `results/verifier_dataset/factorial_v4_final_896_20261015/candidate_review/`:
  896 candidate records and 896 exact student PNGs, complete manifest and explicit
  `REVIEW_ONLY_NOT_ACCEPTED.json`; each row has `training_eligible=false`.
- `results/verifier_dataset/factorial_v4_examples_20261015/index.html`:
  three training-split positive/negative pairs, original two-view images,
  proposed reasoning labels in English and Chinese, raw witness/certificate
  links, color-swap controls and one visibility exclusion. The 25 example proof
  delivery files were hash-verified. No locked-test model prediction is shown.

Scope: colored boxes are explicit virtual six-face no-crossing boundaries, not
new physical walls. Negatives use one closed-separator mechanism under that
contract. Positive wording explains visible boundary relations; only the actual
safe-success trajectory establishes existence. Endpoint containment alone does
not establish general robot reachability. Policy failure remains UNKNOWN.
The user's next step is data-quality review; A/B training requires a later
discussion and is not queued. The chronology below is preserved as history.

## Authorized scope and budget

The user authorized 1,000 reliable records from existing pi0.5/full AEGIS sources,
more independent initial states, witnessed positives, certified conditional
negatives, evidence-linked reasoning labels, visual-shortcut audit and verifier
training only after data acceptance. Policy failure remains UNKNOWN.
Native support contacts are allowed under FD1000-SC-CONTACT-v2.

The exact 32 approved GLM-4.5V requests through OpenRouter/Z.AI completed for
**$0.05882376**, with no unsettled calls. The authorization allows only those
32 exported image/task requests; no further screenshot egress is authorized.
The full FD1000 API budget is $20, including a $6.40 ceiling for that completed
batch. All current collection, reasoning targets and training preparation are
offline. The private key is never printed, copied or logged.

## Collection wrap-up requested; V4 acceptance still pending

Final selection update: strict image exclusions make the original 250-unit
quota infeasible (the unchanged selector stops at 245). Following the user's
instruction to prefer quality over 1,000 rows, freeze **224 already reviewed
units / 896 rows**, with four equally populated cells in every original split:
32 units per cell in train and 12 per cell in validation/test (512/192/192 rows).
Published review v4 retains all 293 reviews and 39 visibility exclusions and
records this reduction before any V4 model fit. Unreviewed candidates are not
eligible for this final selection. No geometry, source split, camera, certificate,
contact requirement, source-coverage minimum or shortcut threshold is relaxed.
The fixed evidence/shortcut audit is the remaining acceptance gate. Training is
still deferred until the user's data-quality review and subsequent discussion.

Update after the user's explicit final-acceptance request: the approval service
and Quest connection are available again. Slurm accounting confirms **both
3634921 and 3634922 COMPLETED/0:0**; the second took 34m35s. All 67 source cases
completed, yielding **494 matched units / 1,976 candidate rows** (248 + 246).
The earlier uncertain status below is retained as history. Collection is closed.
The remaining authorized work is selection, fixed data acceptance and a user
review pack of typical examples; **no A/B training is to start before the user
has reviewed data quality and discussed training**. No further API call is needed.

The user's latest direction is to stop expanding if the pool is sufficient,
wrap up and discuss next steps. No additional collection, audit or training job
was submitted after that direction. The current pool is sufficient in raw count
to stop expanding; it has not established 1,000 scientifically accepted records.

Both V4 collectors use published source
`96a99b51efaa2934bf53bfe10c2a73c806f1cd29` and protocol
`9da9e2638eab6904b629665acb8daf1748ca3c65d7cb2fcb9782c379c5258787`:

- **3634921 COMPLETED/0:0**, 27m45s, qgpu0408, root
  `20261015T013004Z_constraint-batch_96a99b51efaa`: all 34 sources,
  248 matched units / 992 candidate rows.
- **3634922 last observed RUNNING**, 33m42s, qgpu2003, root
  `20261015T013009Z_constraint-batch_96a99b51efaa`: 32/33 sources complete
  at the last successful observation. An earlier count after 26 sources was
  176 units. Its final state, exit code and total are **unconfirmed**. Its
  pending walltime adjustment to 60 minutes is preserved remotely in
  `SCHEDULING_UPDATE.json`; scientific source and settings were unchanged.

Thus the confirmed lower bound is **424 units / 1,696 candidate rows**. Four
rows per unit pair two color assignments with two protected-color rules at one
physical checkpoint. The intended final 1,000 rows mean 250 checkpoint units
from at most 67 original trajectories, not 1,000 independent trajectories.
The certificate construction still uses one closed-separator mechanism.

Local `reviews/fd1000_factorial_v4_visibility_v1.json` binds **178 reviewed units
across 48 original sources**, with **10 excluded and 168 premises-visible**,
to their actual image hashes. All 738 delivery files for the completed reviews
were rehashed during wrap-up. This is Codex image review, not independent human
review, and visible premises alone do not establish scientific acceptance.
Another 32 downloaded units in `shard1_17_25` remain unreviewed. Second-shard
source indices 25 through 32 have not been exported for review. No final V4
250-unit selection, shortcut fit or locked-test prediction has been made.
Accepted rows, finalized training targets and new verifier training runs are zero.

The final Quest status query was **not executed**: automatic approval review
hit a Codex usage limit and could not complete. This is an approval-service
failure, not a determination that the action was unsafe or that the experiment
failed. No alternate route was used. The already submitted collector may have
continued independently; no completion or cancellation is asserted. Keep the
Quest source frozen until both terminal states are verified.

The local wrap-up record is
`results/verifier_dataset/factorial_v4_collection_20261015/WRAPUP.json`.
These wrap-up docs and the visibility bank are saved locally, not yet committed,
published or synchronized. Original submission receipts and all prior results
remain intact. API spend remains **$0.05882376 of $20**, with no new paid call.

Remaining completion work is to retrieve the terminal receipt, finish reviewing
the deterministically selected units and run the fixed independent evidence and
grouped shortcut audit. Preserve rejection if the unchanged gate fails; do not
relax it or automatically start another construction revision. After acceptance,
the proposed next experiment is the matched direct-verdict versus grounded
generative-verifier comparison, with the same inputs, splits and training budget.
Measure verdict accuracy, counterfactual consistency and explanation support.
Passing simple shortcut checks would not by itself prove reasoning or establish
that a generative verifier is necessary. Full SFT and more collection are not
needed for this next comparison and are not being started during wrap-up.

## V3 rejected; V4 native prototype qualified for bounded bulk collection

Independent V3 audit **3628952 COMPLETED/0:0**, 9m20s on qnode0106,
`20261015T003524Z_audit-constraints_4abfcd9fe913`, rechecked all 67 source traces
and 495 raw units. The 124 image exclusions left 371 eligible units. The
proposed 1,000 rows met the fixed split, mode and image-review requirements but
**failed scientific acceptance**: validation AUROC **0.7690** for the fixed
joint linear model and **0.8311** for the fixed forest, above 0.75. RGB-only and
text-only AUROC were 0.5. Raw-stage `AUDIT_COMPLETE.passed` does not override
`SHORTCUT_AUDIT.passed=false`. Test predictions remain unscored; accepted rows,
finalized reasoning targets and new verifier training runs all remain zero.

Fixed diagnostic **3630797 COMPLETED/0:0**, 53s,
`20261015T005127Z_diagnose-shortcuts_4abfcd9fe913`, found validation forest
AUROC 0.7113/0.8247 from individual views, 0.8542 from color statistics,
0.6665 from color areas and 0.8600 from color centroids. These results identify
an unwanted color-position cue, not a need for generative verification. All
16 delivered audit/diagnostic files were hash-verified locally. Source, raw
images, exclusions and all failed versions remain recoverable.

V4 is frozen prospectively before new rendering/fitting: negative separator
ENTRY/EXIT and positive-region BOTH/NEITHER endpoint containment are independent
preassigned factors. Each factor appears five times per ten source checkpoints.
Both boundaries have equal dimensions, exactly equal vertical intervals and the
same source/checkpoint-seeded line width. Separation must be horizontal; cases
without such a certificate stay UNKNOWN. Current target/necessary-goal geometry
and the unchanged fixed 12 cm allowance determine all placements, before future
actions are examined. Actual complete suffix witnesses remain mandatory in all
four cells; endpoint containment alone cannot certify a positive.

Native cameras, seeds, contact rules and original source groups are unchanged.
Student images increase to 448 square pixels to address observed target-visibility
failures. Review sheets retain every actual student pixel. Four fixed color-only
linear/forest readouts (statistics and centroids) join the primary screen before
V4 fitting; all six joint readouts must meet the original 0.75 limit. RGB/text
chance checks remain. No readout hyperparameters or source splits are tuned.

The 250-unit quotas remain 600/200/200 rows. In EXIT-BOTH, EXIT-NEITHER,
ENTRY-BOTH, ENTRY-NEITHER order, joint-cell quotas are 38/37/37/38 for training,
13/12/12/13 for validation and 12/13/13/12 for locked test. Whole four-row units
stay together. Both factors have exactly balanced margins in every split.
Prototype qualification requires four families and four early checkpoints,
at least eight matched units, at least two in each joint cell, normal Slurm
exit and actual-image review. Bulk collection still uses at most two A100 jobs.

Local validation: 93 dataset tests and 21 training contract/metrics/comparison
tests passed; byte compilation and diff whitespace checks passed. The future
fixed A/B processor contract uses two 448-pixel images and max length 2048;
actual processor/no-truncation and gradient preflight remains unrun. V4 native prototype **3634063 COMPLETED/0:0**, 4m12s on A100-SXM4-80GB/qgpu2003,
root `20261015T012057Z_constraint-prototype_34922227dd21`, used published source
`34922227dd21473fad3b54a32e780d6a5e73c9fa` and protocol
`9da9e2638eab6904b629665acb8daf1748ca3c65d7cb2fcb9782c379c5258787`.
It produced 14 matched units / 56 candidate rows: four in each NEITHER cell and
three in each BOTH cell. Object checkpoints 5 and 10 failed their actual positive
guard and supplied no paired unit. All 58 delivered files were hash-checked and
all 56 actual 448-pixel student images inspected. Published qualification:
`reviews/fd1000_factorial_v4_prototype.json`. Native images/color swaps are
consistent; small Object targets and local occlusion still require fresh final
per-unit review. This qualifies bounded collection only. There is still no
accepted final dataset or new verifier training.

## Historical V2 rejection and V3 collection/review before its audit

V2 audit **3614374 COMPLETED/0:0**, 9m54s, root
`20261014T222942Z_audit-constraints_c2182eb82cf7`. All 595 raw units passed
witness/certificate/image-integrity checking; 41 visible-premise exclusions
left 554 eligible units. The selected 250 units/1,000 rows had complete prior
image review, 63 sources, 12 compositions, four families and 532/216/252 rows.
**Scientific acceptance failed**: validation AUROC was 0.645919 for the fixed
linear joint readout and **0.867627 for the fixed depth-4 forest** (limit 0.75).
Image-only and text-only AUROC were exactly 0.5. No test predictions, accepted
final data, finalized training rationales or verifier training were produced.

Fixed exploratory diagnostic **3616198 COMPLETED/0:0**, 43s,
`20261014T224323Z_diagnose-shortcuts_5e30f3543e0c`, found forest validation AUROC
0.839935 from view 1, 0.786008 from view 2, 0.682099 from two-view color
statistics, 0.617884 from color area and 0.627486 from color centroids. The
residual strong signal needs coarse RGB tiles plus the rule; target containment
alone still nearly determines the current contract label. This does not prove
that the model needs a generative verifier.

V3 is a prospective construction change, not a threshold or classifier search.
It keeps native physics, both qualified cameras, frozen source groups and the
0.75 gate. Source/checkpoint-seeded alternating modes cover all four endpoint
side combinations: negative inside/outside versus witnessed positive
outside/outside, and negative outside/inside versus witnessed positive
inside/inside. The inward certificate requires the ENTIRE orientation-enlarged
necessary material goal to lie inside the closed boundary. Both modes use equal region dimensions from the current target/goal envelope
plus a fixed 12 cm margin, never future actions. The exit-mode outside region
lies 3 cm beyond that envelope; the entry-mode positive region encloses it;
its complete actual source suffix must pass the guard or stay UNKNOWN.
This remains one closed-separator mechanism, extended in direction.

Before any v3 readout fit, quotas are fixed at **600 train / 200 validation /
200 test rows**, with equal construction-mode counts within each original source
split. Native source group assignments remain byte-identical. Each source is
still limited to ten checkpoints, and each physical checkpoint has one complete
four-row color/rule unit. V3 first needs a new four-family, two-checkpoint native
prototype with at least three matched units of each mode, actual-image review,
and normal exit. No new API requests are authorized or needed.

V3 native prototype **3619478 COMPLETED/0:0**, 2m55s on A100-SXM4-80GB
qgpu2002, root `20261014T230815Z_constraint-prototype_54d27600d138`: seven
matched units/28 candidates, four outward and three inward. The Object step-0
inward positive witness failed its new enclosing guard and was not admitted;
its failure did not become an infeasibility label. All 28 selected images and
76 delivered files were inspected/hash-checked. Review record:
`reviews/fd1000_sides_v3_prototype.json`. This qualifies bounded collection only;
small targets, occlusion and overlapping volumes still need final per-unit review.
The pending prototype's requested time was reduced from 30 to 10 minutes after
identity checks, recorded in `SCHEDULING_UPDATE.json`; its source was unchanged.
New bounded bulk jobs request 90 minutes each, with at most two task GPUs.

V3 bulk shards **3622211** and **3622216** both terminated **COMPLETED/0:0**,
in 29m33s and 30m19s on qgpu2002 and qgpu2005 respectively. Both used clean
published source `bd13a33c3339ee94377010d744a2860bf18da9f9`:
`20261014T232911Z_constraint-batch_bd13a33c3339` (34 original sources), and
`20261014T232922Z_constraint-batch_bd13a33c3339` (33 original sources).
Together they processed all 67 original sources and produced **495 matched
units / 1,980 candidate rows**, comprising 288 outward and 207 inward units.
They executed immutable source archives and retain the protocol digest
`88538ce0a48eb05492ed59092a759ac514df6ee051a974b4423113a2b3c67a03`.
Neither normal completion nor candidate count is scientific acceptance.
V3 requires fresh pixel-bound visibility review. Review-sheet captions now name
the expected endpoint sides for each mode, solely to guide checking; student
PNGs and model inputs contain no such captions and are unchanged.
Automatic approval review rejected an attempted live-checkout sync while these
jobs ran. Sync was deferred; existing published result-copy tools and local image
review continued. Both normal terminal records and an empty project GPU queue
were verified before resuming source synchronization. No bypass was used.

The first v3 review round inspected **293 units across 64 sources** and excluded
**102 whole units** for concrete visibility faults. The remaining **191 visible
units cover 44 original sources**. The Object wide views often leave the outlined
target too small or crowded by the gripper/nearby objects; these candidates are
not admitted on the strength of a geometric certificate alone. Published record:
`reviews/fd1000_sides_v3_visibility_v2.json`. Every delivered file and all review
image hashes were rechecked. This is Codex image review, not independent human
review. Deterministic quota selection and additional selected-image review are
now complete as described below; a new independent evidence/shortcut audit is
still required.

Published `reviews/fd1000_sides_v3_visibility_v6.json` binds **375 actual-image
reviews and 124 whole-unit exclusions** to the exact four PNG hashes. The final
preliminary selection contains **250/250 image-reviewed units / 1,000 rows**,
from **44 original sources, 10 native task compositions and three families**:
121 Spatial, 60 Long and 69 Goal units. No Object unit is in this selection.
There are at most eight selected checkpoints per source. Frozen row quotas are
**600 train / 200 validation / 200 test**, with **23 / 10 / 11 source groups**;
both construction modes have exactly equal counts within each split.
`results/verifier_dataset/sides_v3_selection_20261014/round4/LOCAL_SELECTION_COVERAGE_CHECK.json`
records the local image-bound coverage check. Preliminary selection SHA256:
`1357f76f207182ed8a99848dd19d9ecb5d9e5fa9c4502f173d1dab01610e2b07`.
Review SHA256:
`95d164f14facd6272569498ecf1856257aa6fd44997f5b93f8f455072a7e3662`.
Only visibility was used for these exclusions before the V3 audit fit. The
subsequent fixed readouts failed acceptance as recorded above; no verifier was
trained and no locked-test model prediction was made.

## V2 collection and visual-review history (not training-approved)

Both source-frozen v2 collectors terminated normally:

- **3603883 COMPLETED/0:0**, 34m52s, A100-SXM4-80GB/qgpu2002,
  `20261014T211358Z_constraint-batch_40c7166d3940`: 34 sources,
  294 matched units / 1,176 candidate rows.
- **3608143 COMPLETED/0:0**, 30m49s, A100-PCIE-40GB/qgpu0208,
  `20261014T214517Z_constraint-batch_e8ecc40f6f41`: 33 sources,
  301 matched units / 1,204 candidate rows.

Together: all 67 source trajectories, 595 matched units / 2,380 candidate rows.
These are the only parents for the next v2 evidence audit. Zero final rows are
accepted and no new verifier training has started. API cost remains $0.05882376.

Published `reviews/fd1000_camera_v2_visibility_v6.json` binds **319 actual-image
reviews**, including **41 whole-unit visibility exclusions**, to their exact
four-image hashes. This is Codex image review, not independent human review.
One source's ten checkpoints all failed visible-premise review; its raw evidence
is preserved. Fixed source-round-robin selection now has **250/250 units with
matching visible-premise reviews**, covering **63 sources, 12 native task
compositions and four families**, with at most five selected checkpoints/source.
The proposed row split is **532 train / 216 validation / 252 test**.
Preview and coverage checks are in
`results/verifier_dataset/camera_v2_selection_20261014/round4`.

Independent CPU audit **3614374** was submitted from
`c2182eb82cf7b7ba76e38d69f17ad6b183dee868` at
`20261014T222942Z_audit-constraints_c2182eb82cf7` and was RUNNING at the latest
check. It rechecks both normal collection parents, all raw witness/geometry,
camera/segmentation evidence and the fixed validation-only shortcut readouts.
Read its terminal accounting and both audit JSON results before proceeding.
The selection-preview helper performs file I/O and visibility curation only;
no final dataset is accepted merely because those previews are complete.
No locked-test prediction has been made and no verifier training has started.

## V2 collection history; v1 failed acceptance

Two v2 batches were submitted from published source
`000bcec9a2a7c29db7c67fb20dcad81a3deb02c2`:

- **3602426**, 34 sources:
  `20261014T210036Z_constraint-batch_000bcec9a2a7`
- **3602435**, 33 sources:
  `20261014T210042Z_constraint-batch_000bcec9a2a7`

Both were PENDING at the last 21:03 UTC check. Read their live Slurm accounting,
PROGRESS and completion files before any downstream step. Their source archives
are immutable. There are still zero scientifically accepted rows and no new
verifier training runs. V1 images, exclusions and shortcut results cannot approve
v2 pixels. Early completed source images may be reviewed while collection runs;
final selection and model/readout fitting wait for both full shards.

Update: 3602426 failed after 54s on H100 node qgpu3004 at the first action of
source 0: actual physics hash did not match the source tape. The same source
passed the four-family prototype on A100-PCIE-40GB (qgpu0208). This is a platform
compatibility observation, not proof of the numerical cause, and it creates no
negative label. The failed root is retained. Future constraint renders request
A100 and keep the exact hash gate. Shard 1 (3602435) was still running on qgpu2014;
read its actual GPU/progress before admitting it. A fresh shard-0 root is needed.
The attempted reduction of pending wall times made no changes because one job
was already failed and the other running; their submitted four-hour caps remain.

Fresh shard 0 **3603883** runs at
`20261014T211358Z_constraint-batch_40c7166d3940`, A100-SXM4-80GB on qgpu2002.
Its sampled checkpoint-5 segmentation is exactly identical across color swaps
in both views, and the actual RGB shows the expected scene. Read live progress;
this spot check is not final acceptance.

Shard 1 **3602435 was cancelled** after a concrete rendering failure on qgpu2014:
checkpoint-5 object-ID masks differed by 770,336 and 769,253 pixels across color
swaps, and actual images showed unrelated wall/cabinet surfaces. The source
physics had passed beyond the first action, but its images are inadmissible.
`STOP_REQUEST.json` and local/remote `RENDER_DIAGNOSTIC` records preserve it.
The precise renderer cause is not established; do not blame all SXM GPUs (the
healthy qgpu2002 shard uses the same GPU model). Future jobs exclude qgpu2014,
record actual OpenGL vendor/renderer and timings, reject software fallback and
camera-data mutation, and check color-swap mask identity immediately. A new
two-checkpoint four-family prototype must pass before restarting shard 1.
All original source splits remain frozen; no test model predictions were used.

Expanded renderer prototype **3607154 COMPLETED/0:0**, 2m45s, A100-PCIE-40GB on qgpu0403, root `20261014T213750Z_constraint-prototype_e8ecc40f6f41`: eight matched units/32 candidates across four families at checkpoints 0 and 5. All 32 selected images were inspected and 69 delivered files hash-checked. Actual NVIDIA GL, camera stability and per-capture color-swap segmentation checks passed. `reviews/fd1000_camera_v2_postaction_prototype.json` qualifies bounded collection only.

Fresh shard 1 **3608143** runs from `e8ecc40f6f41cb8cc6544a5ed55e40a726b1059c` at `20261014T214517Z_constraint-batch_e8ecc40f6f41` (A100-PCIE-40GB, qgpu0208). It replaces the cancelled image-invalid shard without deleting it. Shard 0 **3603883** had completed 31/34 sources and 1,096 candidates at the latest check. Streaming image review has inspected 70 units from 25 completed sources and marked six units for whole-unit exclusion based only on visible premises. Read live receipts; none of this is scientific training acceptance.

**The v1 image dataset failed scientific acceptance. Do not train it.** Audit
9997306 (`20261014T202221Z_audit-constraints_51b09f4a94c3`) completed normally.
Its raw evidence checks passed, but validation AUROC was 0.928755 for the fixed
joint linear readout and 0.983110 for the depth-4 forest (limit 0.75). Image-only
and text-only AUROC were exactly 0.5. The final test split was not scored.
The proposed 1,000 records covered 64 original sources, 12 compositions and
four families, with 544 train / 216 validation / 240 test rows.

The first audit attempt 9993971 failed on Slurm accounting connectivity; the
second 9995112 finished raw evidence checking but lacked scikit-learn. Fixed
official PyPI wheels (scikit-learn 1.5.2, joblib 1.4.2, threadpoolctl 3.5.0) were
staged without changing the base environment, then extracted/tested on compute.
The dependency manifest is bound to each submitted receipt.

Actual-image review inspected the initial 250 units and 60 replacement units,
excluding 72 ambiguous units and retaining 238 image-reviewed units. Published
records are `reviews/fd1000_visibility_exclusions_v1.json` and `v2.json`.
These visibility checks do not override the failed shortcut screen; no v1 rows
are accepted for training. The remaining 12-unit preview is preserved but has
not been reviewed, because the image protocol now needs repair.

The revised image protocol uses **two opposing external simulator views**, with
equal distance/fovy/elevation, the midpoint of both virtual regions as focus,
and a source/checkpoint-seeded azimuth. It removes the target-following native
wrist framing from the student inputs. Physics, witnesses, certificates, frozen
source splits and the 0.75 shortcut limit remain unchanged. Both labels still
share exactly the same capture. The bounded prototype below passed; a fresh bulk audit is required;
old pixel reviews cannot approve the new images. This is a dataset-development
revision after a validation failure, not a favorable test result.

V2 prototype **3600572 COMPLETED/0:0**, 2m34s, root
`20261014T204247Z_constraint-prototype_848bd2503786`: four matched units/16
candidates, all original physical hashes retained. All 16 selected images and
72 delivered files were inspected/hash-checked. Review record:
`reviews/fd1000_camera_v2_prototype.json`. This permits bounded bulk collection,
not scientific training; the small Object target still needs per-unit review.

The v1 exploratory CPU diagnostic **3601447 COMPLETED/0:0**, 43s, root
`20261014T205226Z_diagnose-shortcuts_d21fa41e2195`, found validation forest AUROC
0.978052 using only the original external view and 0.944787 using only the wrist
view; color-centroid features alone reached 0.984825. Thus fixed positional
framing affects both original views. The v2 paired views focus on the midpoint
of both regions and rotate by a fixed source/checkpoint seed. No diagnostic
scored the locked test or changed the acceptance threshold.

Two source-frozen rendering/replay batches were submitted from published commit
`e1859532ae9e0155e085f4120c06810219bcc7e9`:

- **9985892**, shard 0, 34 original sources:
  `20261014T184800Z_constraint-batch_e1859532ae9e`
- **9985893**, shard 1, 33 original sources:
  `20261014T184802Z_constraint-batch_e1859532ae9e`

Roots are beneath `/projects/p33100/siosio/crashbench_safelibero/verifier_dataset`.
Both jobs are terminal COMPLETED/0:0: shard 0 produced 290 matched units/1,160
candidates and shard 1 produced 302 units/1,208 candidates. Together they processed
all 67 sources and produced 2,368 candidate rows, not accepted training data.
At most two task GPUs are allowed. Jobs execute immutable source archives.

Audit 9993971 (`20261014T195410Z_audit-constraints_01ae2d2e22ef`) stopped with
FAILED/1:0 when Slurm's accounting database refused the compute-node connection
for its second parent. This is an infrastructure failure, not a data verdict.
The repair captures both terminal accounting records in the launcher, binds them
to the parent receipt/completion hashes, then audits the evidence offline.
The failed root remains recoverable; a new exact-commit audit is required.

The construction is explicitly **derived constraint-conditioned data**: unchanged
native physics/action trajectories plus noncolliding virtual six-face regions.
The same image, target and goal receive two different protected-color rules.
A second capture swaps the region colors. A legal source suffix proves one rule
feasible; a closed-separator certificate proves the other conditionally infeasible
under the stated material-center guard and fixed displayed goal. These are not
solid-cage physics results or unchanged SafeLIBERO scores.

The target is 250 qualified physical checkpoints × four paired records = 1,000
records (500/500), with at least 30 original sources, six native compositions,
three families and at most ten checkpoints per source. Four variants are not
four independent scenes. The original source pool has 67 initial states, 12
native task compositions and four families. Native task templates can overlap
between splits; this is a source-state holdout, not an unseen-task benchmark.

Source splits were frozen before fitting any shortcut readout or verifier:
**37 train / 15 validation / 15 test ancestry groups**. Exposed prototype sources
are training-only. The final selected subset may contain fewer groups; all
minimum coverage gates remain enforced.

## Completed prerequisites

- Native merged audit **9975098 COMPLETED/0:0**:
  `20261014T171556Z_audit-candidates_9e5b9259f2a3` — 670 positive candidates,
  67 independently replayed source trajectories, ten checkpoints each. This
  one-class pool is not a complete mixed feasibility dataset.
- Early-checkpoint replay array **9980457** and initial physical-cage screen
  **9980458** are terminal COMPLETED/0:0 across all eight shards. Their raw roots
  are `20261014T180303Z_replay-early-sources_d2d00ff28726` and
  `20261014T180306Z_paired-screen_d2d00ff28726`. These add evidence/diagnostics,
  not independent source identities or finalized training records.
- Reciprocal physical-object policy test **9978386 COMPLETED/0:0** produced zero
  complete role-swap units: two reverse pi0.5 safety failures and one unavailable
  reverse certificate. Failures were not converted into negatives.
- First virtual-rule prototype **9981319 COMPLETED/0:0**: two of four pairs;
  two target/goal regions overlapped, so their negative labels stayed UNKNOWN.
- Fixed initial object-receiver region and native segmentation prototype
  **9982264 COMPLETED/0:0**: four pairs/16 candidates. 125 delivered files were
  hash-checked. Actual-image review found a region partly outside the native
  field of view, so this version was not promoted to scientific training.
- Shared external-overview prototype **9984715 COMPLETED/0:0**, 2m37s,
  NVIDIA A100-PCIE-40GB: four families, four pairs, 16 candidates. The external
  view fits all current declared regions with a fixed margin before either rule
  query; wrist view remains native. Every original per-action physical hash
  still matches. 134 delivered files were hash-checked; all eight selected
  assignment-0 images were visually reviewed. The Object target remains small
  in the overview, so bulk observability still needs individual review.
  Root: `20261014T183717Z_constraint-prototype_1b389bd29606`.

## Remaining gates

There are still **0 accepted final training rows, 0 finalized reasoning targets
and 0 new verifier training runs**. After both batches finish normally, run the
implemented `audit-constraints` stage against both roots. It rechecks raw contacts,
source action/state hashes, primitive-material certificates, image/segmentation
consistency, ancestry and complete contrast units. It proposes 250 units and
creates a visual review pack. Pixel counts assist review but never establish
recognizable or fully observable premises by themselves.

The predeclared shortcut screen fits only training sources and evaluates only
validation sources. RGB/red-only and task/rules-only readouts must be at exact
chance from paired-input balance; fixed linear and shallow-forest joint color/
rule statistics must have validation AUROC ≤0.75. No threshold or hyperparameter
search is allowed. The final test split is not scored during dataset screening.
Passing these readouts does not prove that generative reasoning is necessary.

Final packaging and the scientific A/B training runner are implemented; actual
visual acceptance and native processor/Torch preflight remain pending.
`verifier_training/fd1000_contract.py` is a tested fail-closed acceptance gate and
fixed comparison configuration, not a submitted training job.
It requires exact data/evidence/review hashes, 1,000 reviewed records and frozen
source groups; historical 120-row engineering approval cannot satisfy it. Planned
A/B evaluation uses free generation, no gold rationale prefix, identical inputs,
fresh Qwen2.5-VL-3B language LoRA and the same data order/update budget.
The predeclared final A/B comparison additionally requires matching sample IDs,
truths, source groups, full inference-input hashes and complete four-row units.
It reports generative-minus-direct accuracy with 1,000 paired source-bootstrap
replicates (seed 1014), preserving the correlation of all variants/checkpoints
from one original trajectory. A separate CPU comparison stage reads the two
already-complete free-generation streams; it makes no new model prediction.
Three new comparison-integrity tests and all 16 earlier pure training checks
passed before any v3 verifier fitting or test evaluation.

Historical status text is preserved in
[STATUS_HISTORY_before_rule_batch_20261014.md](STATUS_HISTORY_before_rule_batch_20261014.md).
All raw earlier runs and source archives remain intact.

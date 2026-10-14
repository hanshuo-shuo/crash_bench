# FD1000 current state — 2026-10-14

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

## Active v2 collection; v1 failed acceptance

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

Historical status text is preserved in
[STATUS_HISTORY_before_rule_batch_20261014.md](STATUS_HISTORY_before_rule_batch_20261014.md).
All raw earlier runs and source archives remain intact.

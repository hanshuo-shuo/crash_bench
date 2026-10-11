# Full120 engineering phase

The user explicitly authorized the full fixed A/B fit after the passed eight-sample
smokes. The original proposals remain immutable; execution copies change only the
authorization/mode and specify evaluation intervals. No hyperparameter search,
architecture change, new simulation, or scientific generalization claim is allowed.

Run `launch_engineering` through `scripts/quest_sync.sh exec` on a clean published
commit. Stages are separate: `audit`, `fit`, `diagnose`, `report`. The audit must
finish before fitting can be submitted; both fits must finish before diagnostics.
All jobs use a checksummed Git archive, existing staged assets, unique campaign
directory, p33100, short/gengpu, and offline compute. At most two GPUs are requested
concurrently. Each fit/diagnostic arm has one A100 and a 60-minute limit.

The same120 evidence labels/targets are preserved. All120 are explicitly isolated
as engineering-only because two RGB views do not certify full label-critical
state. Hidden-target32, tiny fragments, clipping, nearest opposite-label pairs,
native-pixel exact collisions and fixed coarse-image collisions are audited.
The near-conflict screen is descriptive, not a perceptual-equivalence proof:
MAE <=1 on RGB0..255 and <=0.1% pixels differing by >8 in any channel. The fixed
coarse screen uses 28x28 BOX RGB rounded /16. Neither is used to tune models.

Fitting uses exactly ten complete shuffled epochs (same seeded order A/B), 300
updates, LR1e-4, rank16/alpha32/dropout.05, AdamW, zero weight decay, constant LR,
microbatch1/accumulation4, BF16/SDPA. All base parameters and vision/projector remain
frozen. Teacher-forced training loss is measured before training and every epoch.
Greedy training generations are saved before training, at update150 and update300.
No early stopping on fit accuracy or additional optimization is performed.

Diagnostics preserve coherent two-camera pairs. A label-independent seed714
single-cycle shuffle moves every pair; score both recipient and donor labels.
Text-only removes image blocks and states that no images are supplied. Identical
inputs and transformations are enforced for A/B. All are training-pool probes.

Matched constraint changes alter only 300 commands to600 or0. At600, all safe
witnesses remain legal and conditional separator certificates remain sufficient.
At0, each saved initial state has contract_success=false and no action, so no
transition can reach the goal: all120 diagnostic labels are infeasible. This is a
trivial horizon-logic probe, not a second physical mechanism. Removing digital
guards would invalidate negative certificates and is deliberately unscored.

Free-generated B responses are checked against the available entity visibility
evidence, using a finite grammar for three visibility statements and two limitation
sentences. Contradictions, assertions with no images, incomplete coverage and
unrecognized prose are recorded separately. Unknown prose requires manual review;
template correctness never means that the binary verdict is causally justified.
Segmentation presence does not establish recognizable object identity, particularly
for fragments <16 pixels (a screening flag, not a physical threshold).

# Frozen analysis plan, 2026-10-09

Freeze this protocol before fitting Task 2 or dispatching Task 3. Task 1 collection
does not use future labels as model inputs. No outcome-selected states, held-out
threshold search, controller redesign, or new benchmark task is included.

## Collection and evidence

Sixty identities in `states.json` are twenty initializations of each of
Spatial/I/task1, Object/I/task2 and Object/II/task1. These are three familiar task
scenarios, not sixty independent tasks. Each has ten paired model-noise seeds and
nominal/full AEGIS arms: 1,200 new rollouts, M=8, H=10, replan=5, 300-action cap.
All original initialization, image processing, action transforms, physics, QP,
success and post-collision continuation rules stay fixed. Every inference retains
all 8×10×7 diagnostic samples, observation, production chunk, independent seeds,
normalization scales and strict same-input/RNG native equality evidence.
The shared-prefix sampler uses sequential native B=1 suffix kernels; no eight-wide
suffix speedup is claimed. The pinned pi05 model consumes images and prompt, not
the logged continuous state vector. Renderer nondeterminism is retained explicitly.

Official crash means protected-object L1 displacement >0.001 m from the post-settle
position. Signed geometry separation is an additional pre-action feature and
does not redefine crash. CPU9257292 confirmed the negative-distance anomaly is a
native box-box measurement defect: positive35.15mm SAT separation but −100.48mm
returned at pre-action101. The corrected box-box kernel uses Euclidean closest
features when separated and signed SAT minimum-translation depth when overlapping,
validated against independent bounded least squares. Other convex geoms retain
native distance. Old smoke min_dist is excluded from fitting. Raw, QP proposed, applied and clipped command
values remain distinct; command norm is not mechanical velocity. Nominal snapshots
are save-only and do not imply validated arbitrary continuation restoration.

## Split and units

`split.json` fixes a task-stratified 42/18 initial-state train/test split (14/6 per
task). All seeds and both methods of a state remain together. Two exposed smoke
states are assigned to train; remaining membership follows label-agnostic SHA256
ranking with salt `uncertainty-split-v1`. Historical CSV is used for identity only.
No outcome is read when splitting. Test inference is entirely absent from fitting,
threshold selection and calibration. Report the within-task generalization limit.

Action clock t is one-based after twenty settling actions, at 20 Hz; infer
boundaries occur at 1,6,11,... and are the primary predictor observations.
Five copied per-action proxy values are not five independent uncertainty samples.
Report outcomes per rollout, with equal initial-state weight. TSR uses the original
`success` flag, including successes with a crash; safe success requires both
success and no official crash. Infrastructure failures are missing executions,
never safe_incomplete labels. Primary estimates require the frozen complete matrix.

## Collision prediction

Use only pre-action inference-boundary rows still at risk immediately before t:
first crash C is absent or C>=t. `crashed` is POST-action and must not define the
risk set. For h in {5,10,20,40}, y=1 iff C is present and t<=C<=t+h-1. This window
includes the current action. TTC=-1 never means an event. Post-crash rows are
excluded. Observed crash within a shortened horizon is positive; terminal safe
success is absorbing and negative. Event-free safe_incomplete with fewer than h
observed actions is right-censored and excluded from the primary fixed-h estimate;
report all censoring/exclusion counts. Do not invent unexecuted safe actions.

Primary raw risk directions are positive disagreement, positive churn, positive
applied act_norm, and negative min_dist. First-inference churn is missing, not zero.
Each state has equal total weight; within a state eligible rollouts share weight,
and within a rollout eligible inference boundaries share weight. Report pooled
and arm-specific AUROC. Fit one-dimensional Platt logistic calibration separately
for each proxy/h using train only and these weights, without class rebalancing.
Training mean/std may stabilize fitting; checkpoint action scales are unchanged.
BA uses calibrated probability >=0.5, fixed in advance. Report test Brier, AUROC,
BA and calibration coefficients. Raw AUROC keeps the predefined direction even
if a fitted slope reverses. Single-class AUROC is NA; constant scores with both
classes have AUROC0.5 by the tie convention. Constant-input correlations are NA.

## Early completion calibration and paired differences

The primary first-five-infer analysis is a prospective landmark immediately before
action21: require all five inferences and no crash/success before that action.
Average their disagreement, churn where available, act_norm, or min_dist with
the corresponding predefined safe-score polarity. Predict final safe success
conditional on reaching this landmark. Early crashes/successes and missing fifth
inferences are excluded with explicit denominators. Any unconditional first-five
summary containing post-crash observations is retrospective only. First-inference
safe-success discrimination is an additional all-rollout initial prediction check.

Fit pooled, arm-shared 1D Platt models on individual train-rollout binary safe
success, equal state then eligible-rollout weight, without class rebalancing.
Evaluate held-out single-rollout AUROC/Brier and plot probabilities against observed
safe-success frequency. This is calibrated completion probability, not native
policy confidence. For the disagreement model define c=P(safe_success). In each
test state, use only seeds where both arms meet the landmark, and compute
delta_c=mean(c_AEGIS-c_nominal) and delta_safe_success on those same paired seeds.
Spearman is primary, Pearson descriptive secondary. Report paired counts and NA
for constant deltas. This conditional post-treatment association is not a causal
estimate of intervention benefit. Do not correlate incompatible denominators.

## Retrospective confident failure and uncertainty intervals

Freeze low-disagreement cutoff at the weighted train-nominal at-risk 0.10 quantile,
using the same state/rollout/infer weights. No test quantile or threshold search.
For failed rollouts, last5/10/20 means ACTIONS preceding and including first crash,
or final executed action for never-crash safe_incomplete. Use only inference
boundaries in each window, disclose partial/empty windows, and report low-proxy
failure frequency and denominators. These event/end-anchored windows and aligned
time curves are retrospective descriptions, not online forecasting evidence.

Use 2,000 fixed-seed initial-state cluster bootstrap replicates, stratified by
the three tasks; retain every seed, arm and time row together within a sampled
state. 95% intervals are percentile intervals. Temporal curves show available
state/rollout counts at each point to expose survivor composition. Report undefined
bootstrap replicates rather than treating them as zeros. Calibration/performance
intervals bootstrap test states conditional on the fixed train fit; they do not
include variation from re-fitting the calibration model. Do not bootstrap actions
as independent samples or choose the most favorable horizon after test evaluation.

## Task 3, authorized fixed zero-command gate

After collection audit and review of this frozen plan, freeze the weighted
train-nominal at-risk disagreement 0.90 quantile without searching alternatives.
At an inference, disagreement strictly above the threshold applies all seven
environment command entries as zero for its five executed queue actions, then
replans normally. Record raw proposal, gate decision and applied command. Native
conditional equality remains checked on the raw policy actions. OSC zero does not
stop dynamics; Panda gripper zero preserves its accumulated actuator target, and
does not necessarily open or halt the fingers. Verify this pinned implementation
before dispatch. No actual rest/freeze guarantee is claimed.

Run all sixty states × five paired seeds (0..4): 300 gate rollouts. Compare them
against matching Task1 nominal and AEGIS subsets at the same 300-action budget.
Primary evaluation uses eighteen held-out states × five seeds; train states and
all-sixty summaries are explicitly descriptive. Report CAR, TSR and safe success,
plus paired state-cluster 95% intervals and gate/stall frequencies. Lower crash
with lower completion is reported plainly. Do not restart at mid-rollout snapshots,
extend the budget, discard failures or add a learned gate. Task3 needs no new VLM
calls when the gated arm is nominal. Use the approved compute pool; material
expansion is reported before any extra allocation.

## Resource and billing provenance

The user approved two A100 workers, 8 CPUs/64 GB each, at most24h each (48 GPUh)
and completing the series; on2026-10-09 the user removed the API dollar cap.
Retain all previous charges, exact prompt/image request hashes, pinned model/Z.AI
provider, complete usage metadata, and at most one bounded transient retry.
Unknown charges stop retries until reconciled. No API calls occur in geometry
preflight. Code/upstream tracked bytes, model/norm/tokenizer/detector assets,
container, config, identities, seeds and Slurm receipts are fingerprinted.

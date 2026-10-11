# Stage 1 diagnosis and controlled binary-verdict correction

The correction improves B's training verdict accuracy from73/120 to97/120, but
does **not** complete the joint engineering fit. Corrected B gets49/72 feasible
and48/48 infeasible cases right,104/120 rationale targets right, and96/120 full
outputs exactly right. All120 verdicts stay unchanged when the action budget is
zero. **Do not proceed to scientific training with these data or claim
generalization.** No further training or simulator collection was launched.

All model, tokenizer, gradient, training and plotting work ran on Quest. The
backbone, language LoRA architecture, two RGB observations, task/constraints,
frozen vision encoder and frozen projector were retained. Only the requested B
target/loss correction was trained, with one balanced smoke and one fixed full
fit. Original A/B checkpoints and scientific results remain unchanged.

## Delivered results

| Training-pool metric | Original A | Original B | Corrected B |
|---|---:|---:|---:|
| Feasible verdicts |72/72|72/72|49/72 (68.1%)|
| Infeasible verdicts |48/48|1/48|48/48 (100%)|
| All verdicts |120/120|73/120|97/120 (80.8%)|
| Correct observability rationales |N/A|120/120|104/120 (86.7%)|
| Exact full outputs |120/120|73/120|96/120 (80.0%)|
| Zero-command verdicts, expected all infeasible |48/120|0/120|71/120|

Rationale correctness means visibility statements and caveats agree with the
audited evidence/targets. It does not establish a case-specific physical proof
of feasibility. The target rationales deliberately avoid unseen geometry,
unobserved routes and exact-clearance claims.

- [Loss curves](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/report/training_curves.png)
- [Training verdict accuracy](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/report/training_accuracy.png)
- [All240 original/zero outputs with independent grades](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/review/ALL_OUTPUTS_AUDIT.csv)
- [All120 original-B decision-token records](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/review/DECISION_TOKEN_AUDIT.csv)
- [All120 original-B probabilities, six scoring conditions](../results/verifier_training/20261011T042240Z_binary_correction_706fd1409f15/diagnostic_summary/ALL120_PROBABILITIES.csv)
- [Corrected-B gold/generated-prefix probabilities](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/conditioning/ROWS.json)
- [Complete numerical summary](../results/verifier_training/20261011T054114Z_binary_review_77d5f6a03e8f/report/SUMMARY.json)

The zero-update points in the accuracy plot include failure to produce the
required learned output format; they are not a standalone benchmark of the
pretrained model's feasibility knowledge. Total-loss scales differ between the
original token-mean objective and the new two-component objective.

## Read-only diagnosis of original B

### Actual tokenization and the decoding penalty

| String | Standalone IDs/pieces | Value tokens after `Verdict:` |
|---|---|---|
| `feasible` |89277 `feas`,1238 `ible`|42203 ` feasible`|
| `infeasible` |258 `in`,89277 `feas`,1238 `ible`|304 ` in`,89277 `feas`,1238 `ible`|
| `0` |15 `0`|220 space,15 `0`|
| `1` |16 `1`|220 space,16 `1`|

The staged native generation configuration has `repetition_penalty=1.05`.
Setting `do_sample=False` did not disable it. At the decision boundary, token304
(` in`) had already occurred in all120 contexts; token42203 (` feasible`) had
occurred in none. The penalty consequently treats the two word labels
asymmetrically. It applies to the decisive first token, not just repetitions in
the prose.

The original model's actual generated rationale token IDs are identical to the
gold-prefix IDs on120/120 examples. Native replay reproduces both raw and
processed decision log probabilities exactly; all120 saved outputs are
reproduced. Thus a bad generated rationale prefix does **not** explain the
original B failure.

| Decision rule, fixed observed rationale prefix | Feasible | Infeasible | Total |
|---|---:|---:|---:|
| Raw full-forward first-token argmax |66/72|43/48|109/120|
| Raw native cached first-token argmax |66/72|43/48|109/120|
| Raw cached complete-verdict sequence score |67/72|35/48|102/120|
| Native penalty-adjusted first-token argmax |72/72|1/48|73/120|

There are nine exact raw class-logit ties. Native `torch.argmax` resolves them by
vocabulary position, favoring token304 over42203; eight ties are negative cases
and one is positive. Complete-verdict likelihood also scores the negative
word's continuation tokens. It is a different decision rule. The earlier102/120
sequence score must not be called native raw greedy accuracy. Full-forward and
cached raw argmax disagree on two borderline cases, with the same aggregate109.

Holding each observed rationale prefix fixed, the native penalty flips **42
correct negative decisions to feasible**, while correcting six positive
decisions. This accounts for42 of the47 observed negative errors; five negative
cases are already wrong under raw first-token argmax. The109/120 number is a
conditional raw decision diagnostic, not a separately measured penalty-free
full-generation run or a calibrated feasibility confidence.

### Classification loss was hidden by easy tokens

All values below are mean negative log likelihoods in nats over the specified
class. Continuation is the sum over remaining label pieces; EOS is separate.

| Original B, gold-prefix/full-forward loss | Feasible72 | Infeasible48 |
|---|---:|---:|
| Decisive class token |0.198244|0.549773|
| Remaining verdict pieces, summed |0|0.00005197|
| EOS |0.00002337|0.00001067|

The original reported verdict-span loss0.114464 averaged the decision together
with these easy continuation/EOS tokens. The all-assistant loss0.005611 was
further diluted by the long, almost perfectly predicted rationale. Neither
number represented near-perfect classification.

In actual cached decoding, raw decisive-token losses are0.199850/0.555682 for
feasible/infeasible. After the native penalty they become0.076563/1.179685.
Gold and generated rationale-prefix conditions give identical probabilities on
every original-B example.

### Actual objective-weighted gradient contributions

Across the120 examples the old objective assigns mean token weights of95.416%
to rationale/format,1.642% to the decisive class token,1.301% to verdict
continuations and1.642% to EOS. These weights are not gradient percentages.

The saved original first-example probes, rescaled by their actual60/62 and2/62
objective weights, give gradient norms2.70366 for rationale/format and0.74414
for the verdict span including EOS. The corresponding first microbatch
contributions are0.67591 and0.18604 after division by four. The unweighted probe
norms alone would give the wrong impression of their relative contribution.

A new read-only autograd pass measures the actual old weighted objective at the
saved final checkpoint, using all30 original four-example epoch-one batches,
training-mode dropout.05 and diagnostic seed713:

| Weighted component | Norm of full-pool mean gradient |
|---|---:|
| Rationale/format |0.0012454|
| Decisive class token |0.0404441|
| Verdict continuation |0.0000110|
| EOS |0.0000207|

The decisive component contributes approximately99.7% of the signed projection
along the summed-component gradient. Norms are not additive, and this is a
pre-clipping/pre-Adam attribution, not an update-share estimate. The first-example
component-sum/direct-total check differs by1.51% in BF16 backward arithmetic;
the attribution should not be treated as bit-exact. No unsaved historical
gradient trajectory is claimed. Weights remain bitwise unchanged.

The conclusion is **metric dilution and an imbalanced initial objective, with
nonzero verdict gradients**, rather than a vanished verdict gradient. The final
gradient is already dominated by the remaining classification errors. The one
bundled correction does not isolate the training effect of each change.

## Controlled correction and gates

`0=infeasible`, `1=feasible`; no new vocabulary or model head is added. The loss is:

```text
0.5 * mean CE(all non-decision assistant tokens)
+ 0.5 * CE(the single binary decision token)
```

The first group contains rationale text, formatting and EOS, keeping every
assistant token supervised without an extra auxiliary-loss weight. Rationale
text, format and EOS losses are also reported separately. Prompt, image and
padding positions remain excluded. EOS cannot dilute the single-token verdict
component.

The original decoding penalty1.05 is explicitly retained to keep the correction
controlled. Both0 and1 already occur in all120 student prompts, so it treats
the two candidate digits symmetrically. All240 old/new encoding checks confirm
identical student inputs and exactly one classification token per example.
The33 Quest unit tests pass, including causal masking, EOS/padding treatment,
invariance of verdict weight to rationale length, and independent LoRA gradient
flow. Three additional decision/readback tests pass on Quest.

The original balanced eight, four examples per class, pass at140 updates:
8/8 verdicts,8/8 grounded targets and8/8 exact full outputs. Total loss decreases
2.140760→0.002470; non-decision loss2.937793→0.004929; decisive loss
1.343727→0.00001180. Both components have nonzero real-model LoRA gradients.
Vision/projector weights remain bitwise unchanged. At120 updates the two hidden
milk cases still had wrong captions/verdicts despite low teacher loss, so the
strict free-generation gate correctly withheld full training until140.

The full fit starts from the same fresh base, not the smoke adapter:

| Setting | Fixed full fit |
|---|---|
| Backbone |Qwen2.5-VL-3B-Instruct, revision66285546d2b821cf421d4f5eb2576359d3770cd3|
| LoRA |Language q/k/v/o and gate/up/down projections; rank16, alpha32, dropout.05|
| Trainable parameters |29,933,568; same252 target modules|
| Frozen components |All base weights, including vision, projector, embeddings and LM head|
| Optimization |AdamW, LR1e-4 constant, weight decay0, gradient clipping1|
| Batch/length/image |Microbatch1, effective batch4; max1536; two224×224 RGB views|
| Schedule |Seed713; ten complete epochs;300 updates; identical original full-fit order|
| Evaluation |Teacher loss every30; free generation before training, at150 and300|

Smoke retains the original LR2e-4/dropout0/cap240 settings. No hyperparameter
search, extra full fit, new data or simulation was performed. Implementation:
[binary_verifier.py](binary_verifier.py), [binary_run.py](binary_run.py),
[fixed execution configurations](../results/verifier_training/20261011T042240Z_binary_correction_706fd1409f15/full_config.json).
The frozen dataset/schema is unchanged; word labels remain in the evidence
records, with the digit mapping recorded in configuration/metadata.

## Full-fit losses and remaining errors

| Final corrected-B teacher loss | Value |
|---|---:|
| Weighted total |0.0782706|
| Non-decision component |0.00558144|
| Rationale text only |0.00646482|
| Formatting |0.00055702|
| EOS |0.00070927|
| Decisive verdict, all120 |0.15095977|

| Decisive-token loss by class | Feasible72 | Infeasible48 |
|---|---:|---:|
| Gold rationale, full-forward |0.167225|0.126561|
| Own rationale, native cached raw scores |0.476214|0.130261|
| Own rationale, native processed scores |0.462311|0.137966|

At150 updates the model had95/120 verdicts,81/120 correct rationales and80/120
exact full outputs. At300 it has97,104 and96 respectively. All120 final outputs
have valid syntax; the16 rationale failures are concrete visibility
contradictions, not unparsed prose. The exact-fit criterion is false even though
the job completed normally.

The final read-only conditioning intervention holds weights and student inputs
fixed and preserves actual generated prefix token IDs. Both conditions use the
same full-forward computation:

| Corrected-B prefix condition | Feasible | Infeasible | Total |
|---|---:|---:|---:|
| Ground-truth rationale |64/72|48/48|112/120|
| Generated rationale |49/72|48/48|97/120|

The own-prefix decisions match all120 native free-generation verdicts. Correcting
only the rationale recovers15 errors: ra_009,019,029,039,042,049,059,069,079,087,
089,097,099,107,117. The remaining eight are ra_007,017,027,037,047,057,067,077:
all are feasible cases with milk hidden in both views, and all fail even with
the correct rationale. One further case, ra_030, has a wrong visibility statement
but a correct infeasible verdict.

The hidden-milk stratum therefore scores24/32 verdicts and32/32 rationales; it
misses all eight feasible members. The other88 cases score73/88 verdicts and
72/88 rationales. These observations support a residual rationale-conditioning
failure and a hidden-target failure stratum. They do not prove irreducible
information loss: original A memorizes all120 with the same inputs. Physical
identifiability remains uncertified by the prior observability audit, and one
source cannot establish generalization.

## Actual generated outputs and zero-command behavior

Correct original-condition example, ra_002, evidence label feasible:

```text
Rationale: The milk is visible only in the external view. The basket is visible in both views. Red barriers are visible in both views. These views alone do not establish exact clearance or a complete safe route. The verdict is conditional on the stated reach-avoid rules.
Verdict: 1
```

Two distinct failure examples, both evidence-labeled feasible:

| Case | Generated milk statement | Verdict | Assessment |
|---|---|---|---|
|ra_007|The milk is not visible in either view.|0|Statement is correct; verdict is wrong even with gold rationale.|
|ra_009|The milk is not visible in either view.|0|Statement is false; replacing the rationale recovers the verdict.|

The complete statements and per-case grades are in the delivered240-row table.
Under zero commands, ra_002 still produces1, although its initially incomplete
task cannot be completed without a transition.

**All120 original/zero verdict pairs are unchanged.** Zero-command accuracy is
71/120 because48 original negatives and23 already misclassified positives are
predicted0 in both conditions. There are zero feasible→infeasible changes among
the49 originally correct feasible predictions, and zero such changes across the
72 evidence-feasible source cases. Rationale agreement is105/120 and joint exact
output correctness56/120 under zero commands.

The immutable primary result's legacy-named `zero_flips_among_original_feasible`
field contains23, which counted zero-correct positive-source cases rather than
actual changes. The independent readback corrects that interpretation; the
source reporter is fixed for future use. No raw training artifact was rewritten.

## Scientific decision and remaining data requirements

Original B's principal observed collapse is a tokenization/decoding interaction
with weak decision margins. Its rationale prefix was not the cause. Loss
averaging concealed the hard decision and the initial component balance was
unequal, but verdict gradients did not vanish. The controlled correction removes
the asymmetric word-label penalty and improves negative recall, while revealing
new rationale-generation errors and failing to learn the zero-command rule.

**No-go for scientific training as currently prepared.** The implementation and
small smoke pass; the120-case joint fit and constraint behavior remain inadequate.
The120 labels still share one original source scene. No random-split or
cross-scene claim is made.

The [Stage2 collection specification](STAGE2_COLLECTION.md) remains a design
only: at least12 independent original sources; verified safe feasible witnesses;
certified infeasible examples; matched irrelevant-change controls; and at least
two distinct constraint mechanisms. Slit/cage variants count as one mechanism,
and a zero-action sanity check is not a second scientific mechanism. The proposed
nontrivial finite-horizon certificate verifier still needs implementation and
independent validation before collection.

Make label-critical premises observable at the actual two-view resolution,
adjudicate per-sentence grounding, counterbalance visibility/wording/constraint
values, and include evidence-supported same-image constraint changes. Keep every
descendant in its original source's future split. Freeze the ancestry manifest,
training/development protocol and locked test groups before scientific evaluation.
Policy failures and uncertainty scores remain excluded as labels. No new
simulator data was generated in this task.

## Provenance and delivery verification

Training source:706fd1409f15bbbee68f289dc2f9bae0d798c3fd. Final readback/report
source:77d5f6a03e8f1be2fefff07e0d1d8adf26d40297. Upstream scientific source remains
2457feed5968ae803926e178c8ce8243b9ecdcf9. Dataset SHA-256 remains
1ab19294758acd2c59cda91077b15fd28938417685a9470d5e72c43a7d41c7b0.

Quest training root:
`/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261011T042240Z_binary_correction_706fd1409f15`.
The adapter is in `full/adapter/`; large weights remain on Quest.
Final reporting root:
`/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/20261011T054114Z_binary_review_77d5f6a03e8f`.

| Job | Purpose | Terminal status | Allocation elapsed |
|---|---|---|---|
|9453285|Read-only token/loss/gradient audit|COMPLETED0:0|5m00s|
|9454018|First handwritten cache probe|FAILED1:0; excluded|2m01s|
|9454708|Revised handwritten cache probe|FAILED1:0; excluded|1m44s|
|9455328|Validated native-prefix diagnosis|COMPLETED0:0|27m32s|
|9457680|Corrected-loss/encoding preflight|COMPLETED0:0|1m40s, CPU|
|9457762|Balanced corrected smoke|COMPLETED0:0|10m23s|
|9459482|One fixed120 fit and zero probe|COMPLETED0:0|46m44s|
|9463693|Read-only corrected-prefix comparison|COMPLETED0:0|2m15s|
|9463700|Independent readback and curves|COMPLETED0:0|16s, CPU|

The two failed read-only probes did not reproduce native cached scores and were
rejected before training. Their raw artifacts remain recoverable; accepted
measurements use native generation/forced-prefix scoring. No failed probe was
used as a label or a model-performance result.

GPU allocation totals95m39s including failed diagnostics. Zero paid API calls;
zero new simulation. All task jobs are terminal. The120 files across six delivery
manifests are checksum-verified locally, with scoped Slurm accounting and an
empty final task queue separately verified. Both plots were visually inspected.
The original and corrected full training orders are identical. Local work was
source authoring, lightweight checks and delivery verification; model/data
experiments and plotting ran on Quest.

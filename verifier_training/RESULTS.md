# Engineering pilot result — 2026-10-10

This is the historical eight-sample smoke report. The later authorized 120-sample
fit, shortcut diagnostics and scientific-readiness decision are documented in
[STAGE1_CLOSEOUT.md](STAGE1_CLOSEOUT.md).

Both corrected eight-sample overfit checks passed on Quest. This verifies the
supervised training path, grounded observation/limitation text generation, and
binary verdict fitting on the training examples. It establishes no cross-source
generalization or complete physical reasoning capability. No full 120-sample
training job was launched. No paid API call was made.

## Delivered artifacts

- [Dataset schema](dataset.schema.json), [exporter](export.py), [training implementation](train.py),
  [assistant masks](data.py), [causal loss and gradient checks](loss.py).
- Export: `results/verifier_training/20261010T225130Z_prepare_6c223ce6fe1c/dataset/`.
  Includes dataset.jsonl, all 240 RGB PNGs, audit, smoke IDs and six contact sheets.
- Final input audit: `results/verifier_training/20261010T232245Z_prepare_f2e99e26bd2d/`.
- Final measurements: `results/verifier_training/20261010T232743Z_smoke_f2e99e26bd2d/`.
  Includes A/B generated outputs before/after and at every checked checkpoint,
  per-step curves, gradient probes, complete provenance, accounting and a PNG plot.
- Proposed configurations: [A](configs/proposed_A.json), [B](configs/proposed_B.json).
  They pin the model and dataset, and are explicitly not approved for launch.
  The current launcher and training entry point remain restricted to smoke runs;
  these are configuration specifications for a subsequent authorized stage.

The corresponding Quest run roots are under
`/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs/` with the
same directory names. Trained adapters are retained in the final run's
`A/adapter/` and `B/adapter/`. Original evidence and superseded attempts remain
recoverable. Delivered files match the Quest SHA-256 manifest; all 240 exported
image hashes and all 16 final generated targets were independently read back.

## Dataset and observability

Exactly 120 reach-avoid states: 72 feasible by complete safe witness, 48
conditionally infeasible by RA-SEP-1 certificate. All 48 UNKNOWN states are
excluded from binary supervision and retained in the audit. No native SafeLIBERO
positive pool, uncertainty score or policy failure supplies a label. The one
invalid later policy endpoint retains its valid independent initial-state label.

Each row separates `student` (two 224x224 RGB images, task, constraints), `target`
(verdict and rationale), and `metadata` (ancestry, contracts, complete evidence
hashes/paths, source/upstream commits, seeds, Slurm provenance and sentence-level
grounding). Image paths and hashes are loader metadata; only pixels reach the
model. IDs, geometry, widths, source groups, evidence records and label ancestry
are excluded from the prompt.

The evidence audit passed original hash checks, persisted witness/certificate
checks, restored-state identity, exact RGB-array matching and visibility
alignment. All 120 image pairs are distinct, with no contradictory exact pairs.
Six contact sheets covering all pairs were reviewed.

Full physical observability is **not established**. The target milk is absent
from both views in 32 states (24 infeasible, 8 feasible). Clipping and small
visible fragments are also recorded. Rationales therefore state supported
visibility facts and missing-evidence limits; they assert no hidden target
location, exact clearance, unobserved safe route or all-route proof. The smoke
contains two rationale templates and three full B response strings. Its success
is a limited engineering result, not diverse causal explanation generation.

All 120 labels descend from `object_I_t2_e0`, one original source scene. Slit and
cage constructors share one sufficient separator proof family.

Dataset SHA-256:
`1ab19294758acd2c59cda91077b15fd28938417685a9470d5e72c43a7d41c7b0`.

## Implementation and final smoke

Model: Qwen/Qwen2.5-VL-3B-Instruct, revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. Final training source:
`f2e99e26bd2d939fc045b306c47feb618ec2a2aa`.

Thirteen unit tests passed on Quest. The actual Qwen processor checked all 240
A/B encodings. Both variants have the exact same full inference inputs: 372
prompt tokens, matching attention masks, image grids and pixel tensors. Only
supervised assistant targets differ. This is enforced by the
`identical_full_inference_prompt_v2` gate.

Prompt/image/padding labels are -100. Assistant text and EOS receive causally
shifted next-token loss. Model B includes every rationale and verdict token.
Padding is masked by position so genuine EOS tokens remain supervised.

Both arms use 29,933,568 trainable LM LoRA parameters across 252 language-layer
attention/MLP projections, rank 16/alpha 32, BF16, SDPA, LR2e-4, microbatch 1,
accumulation 4, zero dropout, seed 713 and a 240-update cap. All base weights,
including vision encoder, merger/projector, embeddings and LM head, are frozen.
Projector weights were bitwise unchanged and vision parameters had no gradients.

The same eight fixed examples span L01/L04, slit/cage, four feasible and four
infeasible states. There is no train/test split. Every 20 updates, greedy
responses must exactly match all eight targets; total CE must fall at least 75%,
and B rationale CE must also fall at least 75%.

| Final arm | Updates | Exact generated training outputs | Total CE before | Total CE after | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|
| A: direct | 100 | 8/8 | 4.328141 | 0.002627 | 8.36 GiB |
| B: generative | 120 | 8/8 | 2.925542 | 0.00005439 | 8.46 GiB |

For B, rationale CE fell from 2.916469 to 0.00005035 and verdict CE from 3.188465
to 0.00012748. The initial independent autograd probes measured rationale
loss 2.843010 with LM-LoRA gradient norm 2.925222, and verdict loss 2.902758 with
gradient norm 11.717935. Both objectives reach trainable parameters.

Loss endpoints are means over the same eight examples; the plot shows per-update
minibatch losses. A/B loss magnitudes are not comparative generalization scores,
and their different target lengths affect the loss composition.

Example actual B output for `ra_045`:

```text
Rationale: The milk is not visible in either view. The basket is visible in both views. Red barriers are visible in both views. These views alone do not establish exact clearance or a complete safe route. The verdict is conditional on the stated reach-avoid rules.
Verdict: infeasible
```

## Execution history, including unsuccessful work

| Quest job | Result | Interpretation |
|---|---|---|
| 9434899 | COMPLETED/0:0, 6m36s | Original evidence export and 13 tests passed. |
| 9435300 | FAILED/1:0, 13m23s | Prototype A fit 8/8; B stopped at 7/8 after 120 updates and misstated visibility on its failed example. Its A/B system format instructions differed. Superseded, not used for the final comparison. |
| 9436182 | Canceled during initialization, 1m31s | B-only extension was superseded by the full identical-input correction. |
| 9436602 | COMPLETED/0:0, 2m53s | Repeated 13 tests and all 240 full-input equality/masking checks; inherited the unchanged dataset through its hash. |
| 9436901 | COMPLETED/0:0, 15m17s | Final fresh A/B adapters both passed; generated the curves and proposals. |

The final allocation used one A100-PCIE-40GB, p33100/gengpu. Its pending
45-minute request was reduced to 30 minutes, recorded in RESOURCE_AMENDMENT.json.
Total GPU allocation including prototype and cancellation was 30m11s. All five
jobs are terminal. No other project's jobs or source were changed.

## Proposed 120-sample configurations and remaining data

The proposed A/B specifications use the same 120 samples, model revision, full
inference prompt, LoRA rank 16/alpha 32, BF16/SDPA, seed 713 and complete-epoch order.
Use LR1e-4, dropout .05, AdamW with zero weight decay, microbatch 1/accumulation 4,
ten epochs/300 updates. A predicts the verdict; B supervises the full grounded
rationale and verdict. Evaluate training fit and generated factuality only.
These settings have not been run on the full 120 examples.

Scientific evaluation still requires:

1. Multiple independent original scenes and tasks, with both labels represented
   within source groups. Keep all ancestry-related variants together in splits.
2. Genuinely distinct physical/proof mechanisms and complete independent
   witnesses/certificates. Keep unresolved states UNKNOWN; failure is not a label.
3. Label-critical information available in the permitted inputs, plus matched
   visibility and other nuisance cues across labels.
4. Richer case-specific explanations grounded in observable relations and valid
   evidence, with explicit limits wherever images cannot support a claim.
5. A predeclared source-held-out evaluation with enough independent groups for
   meaningful uncertainty estimates. No random-split accuracy from this pool
   can establish cross-source generalization.

# Generative Feasibility Verifier — engineering pilot

Latest fixed lower-LR rank64 comparison: [LOW_LR_RESULTS.md](LOW_LR_RESULTS.md) — smoke passed, but the120 fit remains97/120 verdicts and92/120 exact outputs.
Prior higher-rank SFT comparison: [RANK64_RESULTS.md](RANK64_RESULTS.md) — the original-LR rank64 smoke failed; no full fit was launched in that run.
Prior diagnosis and controlled correction: [BINARY_CORRECTION_RESULTS.md](BINARY_CORRECTION_RESULTS.md).
Original full-dataset A/B results: [STAGE1_CLOSEOUT.md](STAGE1_CLOSEOUT.md).
Original eight-sample smoke history: [RESULTS.md](RESULTS.md).
All-row audit: [OBSERVABILITY_ANALYSIS.md](OBSERVABILITY_ANALYSIS.md).
Future collection design only: [STAGE2_COLLECTION.md](STAGE2_COLLECTION.md).

Authorized 2026-10-10. This isolated directory adds no behavior to existing
scientific pipelines. All data export, audits, processor tests and GPU training
run on Quest. CPU: p33100/short; GPU: p33100/gengpu. The user subsequently authorized
the fixed 120-sample A/B engineering fit and shortcut probes. The original smoke
launcher stays bounded; `launch_engineering.py` adds separate audit, full-fit,
diagnostic and report stages. At most two A100 jobs run concurrently. Zero API
calls. Historical raw results remain read-only.

## Dataset schema

`dataset.jsonl` contains exactly 120 initial reach-avoid states: 72 feasible by
complete safe witness and 48 conditional RA-SEP-1 exclusions. The 48 UNKNOWN
states are recorded in AUDIT.json but never used for binary supervision. Native
SafeLIBERO positives, policy outcomes and uncertainty scores are not label inputs.
One invalid later policy execution still has valid independent initial evidence;
its missing terminal GPU manifest is explicitly recorded.

Each record has:

- `schema_version`, opaque `sample_id` (metadata join key, never prompt text).
- `student.images`: external and wrist RGB PNG paths, camera names, dimensions
  and SHA-256. Paths/hashes are loader metadata; only image pixels enter the model.
- `student.task`, `student.constraints`: the same task and full digital contract
  for every record. Includes T=300, bounded actions, protected contacts, fixed
  initial goal, center guards and swept material-cube guard.
- `target.verdict`: `feasible` or `infeasible` (the latter is conditional).
- `target.rationale`: deterministic observations supported by aligned segmentation
  and an explicit limit on exact-clearance/complete-route inference.
- `metadata`: original scene ancestry, layout, constructor, variation, label
  evidence type, raw CPU/GPU roots, code/upstream commits, seed, Slurm IDs,
  original evidence file hashes, image visibility and per-sentence grounding.

`student_messages` is an explicit allowlist. Ancestry, constructor, variant,
width, privileged geometry, labels, policy results and evidence files do not enter
student prompts. A and B use exactly the same full inference inputs: system text, task,
constraints and both image tensors.
Only the supervised assistant target differs.

## Grounding and observability

Audit gates verify original CPU file hashes, recompute persisted certificate and
witness predicates with the unchanged pure checker, compare initial CPU/GPU
snapshots, require aligned segmentation, and compare each PNG byte array to the
saved policy RGB array. Original substep safety verification is inherited through
its complete hash manifest, not claimed to be a newly rerun simulator proof.
Duplicate image pairs with conflicting labels fail export.

Full physical observability is NOT claimed: hidden target locations, obstacle
back faces, metric clearance and dynamics are not observable from these images.
Rationales therefore describe visible entities and this limitation; they do not
invent an all-route exclusion or a witness from an initial image. These targets
exercise grounded description/limitation generation, not complete causal proofs.
Occlusion, clipping and pixel-count evidence stay in metadata. There is one
original source scene and one separator proof family. No random split is used.

## Training and acceptance

Qwen/Qwen2.5-VL-3B-Instruct is pinned to a resolved Hub commit and file hashes.
An isolated two-wheel overlay uses the existing read-only torch 2.7.1 /
transformers 4.53.2 runtime. Vision, its merger/projector, embeddings, LM head and
all base weights are frozen. Only language-layer q/k/v/o and gate/up/down LoRA
matrices train (rank 16, alpha 32). BF16, SDPA, microbatch 1, accumulation 4, seed 713,
LR2e-4, at most 240 updates per arm, no dropout for the overfit check.

`data.py` applies the native multimodal chat template; image expansion is handled
by the processor. The full token prefix must equal the generation prompt. Only
assistant content and its terminator receive labels; prompt, images and padding
are -100. Padding is masked by position, not EOS token ID. B trains rationale AND
verdict under ordinary unweighted mean next-token cross entropy. Causal shifting
is explicit. Token audit runs over all 240 A/B examples before GPU submission
and requires bitwise-equal A/B inference token IDs, attention masks, image grids
and pixel tensors.

The eight fixed examples span two layouts, both constructors and four positives /
four negatives. Evaluate greedy outputs on those SAME eight examples every 20
updates. Each arm must produce all 8 exact target outputs and reduce total CE by
at least 75%; B rationale CE must also drop 75%. Separate autograd probes require
finite, nonzero rationale and verdict losses AND nonzero gradients into LoRA.
Trainable-name assertions and visual gradient checks enforce freezing; projector
weights are checked bitwise after optimization. Failed attempts are retained.

Only after both arms pass does `report.py` emit training curves, generated outputs
and proposed 120-sample A/B configurations (rank 16, LR1e-4, dropout .05, ten epochs,
300 matched updates at effective batch 4). Those files do not launch full training.
Training-set fit is solely an engineering metric.
The proposals pin the measured model revision and specify complete-epoch
shuffling shared by both arms. They do not inherit the eight-example smoke
early-stop rule. A/B comparisons on these data remain descriptive.

The authorized full-dataset execution uses those same proposals with an explicit
authorization/mode change: exactly ten complete epochs, 300 updates, LR1e-4 and
dropout .05. A/B inference inputs and sample orders remain identical. The all-row
observability audit explicitly isolates all 120 as engineering-only, preserving
their evidence-backed labels. A reaches 120/120 training verdicts; B reaches 73/120
while matching all 120 observation/limitation templates. Full-run `passed` means
execution/integrity completion; inspect `exact_fit_criterion` and generated scores
separately. B's full-dataset fit criterion is false. No retuning follows that result.

## Quest execution

Read QUEST_WORKFLOW.md and setup/README.md, run scripts/quest_sync.sh check,
verify clean repository identity and absence of active CrashBench jobs depending
on the live source. Publish the tested commit and sync through quest_sync.sh.
Use the read-only base Python `/projects/p33100/siosio/envs/openpi/bin/python`.
Invoke module `verifier_training.stage UNIQUE_ASSETS_ROOT` via quest_sync.sh exec
on the login node (downloads/hashes only). Then invoke `verifier_training.launch
prepare --assets UNIQUE_ASSETS_ROOT` via the same tool. Wait for PREPARED.json;
inspect AUDIT.json, TOKEN_AUDIT.json and the contact sheets. Invoke `launch smoke
--assets UNIQUE_ASSETS_ROOT --prepared PREPARATION_ROOT` only when those checks
pass. Each launcher archives a published clean source and creates a unique root.
Do not change frozen archives. Outputs go under the project's verifier_training
asset root; delivery summaries are copied into results/verifier_training.
An optional `launch report --assets UNIQUE_ASSETS_ROOT --smoke SMOKE_ROOT`
creates a new CPU-only report root from completed, passing A/B measurements.
It preserves the original measurements and records the separate report commit.

For the completed full phase, see [ENGINEERING_PROTOCOL.md](ENGINEERING_PROTOCOL.md).
`launch_engineering audit --assets ... --prepared ... --smoke ...` creates a new
immutable campaign. Submit `fit --campaign ...` only after its audit/isolation
gate; `diagnose --campaign ...` and `report --campaign ...` use the completed
records. The actual campaign also used Slurm `afterok` dependencies to queue
downstream work safely before predecessors finished; receipts and amendments
record that schedule. `launch_review CAMPAIGN_ROOT` creates a separate CPU-only
case review from a clean, tested, published commit. It preserves the original
automatic grades and adds an exact-input/output-bound manual no-image review.

## Scientific data still needed

Independent original scenes/tasks and proof mechanisms; matched label diversity
within each source and matched visibility/nuisance cues across labels;
sufficient visual access to label-critical facts; annotated
relations and action/obstruction explanations grounded in complete evidence;
ancestry-grouped source holdouts; ambiguity and UNKNOWN handling. Native positives
and synthetic negatives must not be combined into a source-label shortcut.

References: [Qwen model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
[Transformers Qwen2.5-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl),
[PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora).

# Generative Feasibility Verifier — engineering pilot

Authorized 2026-10-10. This isolated directory adds no behavior to existing
scientific pipelines. All data export, audits, processor tests and GPU training
run on Quest. CPU: p33100/short; GPU: p33100/gengpu, one A100, at most 60 minutes
per smoke allocation. Zero API calls. No full training job is authorized by the
launcher. Historical results remain read-only.

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
student prompts. A and B have identical image/task/constraint inputs and differ
only in the system output-format instruction and assistant target.

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
matrices train (rank16, alpha32). BF16, SDPA, microbatch1, accumulation4, seed713,
LR2e-4, at most120 updates per arm, no dropout for the overfit check.

`data.py` applies the native multimodal chat template; image expansion is handled
by the processor. The full token prefix must equal the generation prompt. Only
assistant content and its terminator receive labels; prompt, images and padding
are -100. Padding is masked by position, not EOS token ID. B trains rationale AND
verdict under ordinary unweighted mean next-token cross entropy. Causal shifting
is explicit. Token audit runs over all240 A/B examples before GPU submission.

The eight fixed examples span two layouts, both constructors and four positives /
four negatives. Evaluate greedy outputs on those SAME eight examples every20
updates. Each arm must produce all8 exact target outputs and reduce total CE by
at least75%; B rationale CE must also drop75%. Separate autograd probes require
finite, nonzero rationale and verdict losses AND nonzero gradients into LoRA.
Trainable-name assertions and visual gradient checks enforce freezing; projector
weights are checked bitwise after optimization. Failed attempts are retained.

Only after both arms pass does `report.py` emit training curves, generated outputs
and proposed120-sample A/B configurations (rank16, LR1e-4, dropout.05, ten epochs,
300 matched updates at effective batch4). Those files do not launch full training.
Training-set fit is solely an engineering metric.
The proposals pin the measured model revision and specify complete-epoch
shuffling shared by both arms. They do not inherit the eight-example smoke
early-stop rule. A/B comparisons on these data remain descriptive.

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

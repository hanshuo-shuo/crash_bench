# FD1000: evidence-first feasibility collection

The user authorized this campaign on 2026-10-14: reuse existing pi0.5 / full
AEGIS trajectories, collect 1,000 reliable records across more original scenes,
include evidence-backed negatives and reasoning targets, audit visual shortcuts,
and train the verifier only after the dataset passes. The new API ceiling is
$20 for this campaign, including retries and unresolved charge reservations.
It does not alter historical baseline budgets or the public benchmark results.

The collection target is 500 witnessed-feasible and 500 conditionally infeasible
records from at least 30 distinct original initial states, six task compositions,
and three suite families. These are coverage requirements, not claims that
descendants or shared task assets are statistically independent. Freeze a source
ancestry graph and grouped split before fitting any shortcut model. All descendants
and policy seeds sharing an original state stay together. Report task-template
overlap separately and include a task-held-out diagnostic when coverage permits.

## Execution stages

1. Inventory the complete official batch and the instrumented 1,200-rollout pool
   on a Quest CPU allocation. Record source hashes, source-state/task coverage,
   available action tapes, snapshots, views and cost metadata. Inventory output is
   **not a labeled dataset**. The official displacement proxy is not a contact
   guarantee. Failed trajectories remain an unlabeled retry/diagnostic pool.
2. Replay existing successful action tapes, first a source-stratified smoke, then
   a bounded batch. Check initial state and per-action physics against original
   traces. Check legality, success and protected contacts at integration substeps.
   Preserve failures and disagreements. Captured states require a successful
   remaining suffix under their own remaining horizon; previously violated
   history cannot be called safe success. Export at most ten informative,
   nonterminal checkpoints per trajectory, not every correlated video frame.
3. Build matched positive/negative constructions on those same source families.
   Each positive needs a replayed safe witness in its actual construction; each
   negative needs a checked sufficient certificate. Reusing RA-SEP-1 is limited
   to its exact digital swept-motion assumptions. A second proof mechanism is
   not presumed implemented. Do not call alternate cages independent mechanisms.
   Do not make natural scenes exclusively positive and synthetic scenes negative.
4. Generate structured facts, relevant constraints, witness events or certificate
   implications. Each clause has an evidence pointer and an input-grounding map.
   Privileged hidden measurements remain metadata; if required premises are not
   observable in the two views or explicit rules, exclude from scientific binary
   supervision and report the partial-observability subset. VLM paraphrases cannot
   establish truth; label agreement is insufficient to accept a rationale.
5. Freeze the data and run grouped audits: color/red-area, visibility, width,
   pooled RGB, image-only and text-only readouts; relevant-constraint flips;
   irrelevant-appearance invariance; duplicate and ancestry leakage; per-source
   and per-class coverage. Real geometric cues are allowed. Do not tune until
   a shortcut's AUROC happens to be low or inspect locked tests while redesigning.
6. Only an independently recomputed successful dataset audit can enable a fresh
   matched direct versus rationale-then-verdict Qwen2.5-VL-3B LoRA comparison.
   Keep the existing binary-token, loss-mask and separate verdict-loss repairs.
   Report free-generation accuracy, explanation factuality, counterfactual
   behavior, calibration and held-out-source performance. Gold-prefix scoring is
   diagnostic, not a second trained model or the primary test score.

## Budget and provenance

Inventory, tape replay and deterministic evidence explanations make no API calls.
Prefer exact-input cached perception. A new paid worker must first inspect current
provider endpoint pricing and observed usage, estimate the actual request batch,
and use a campaign-specific durable ledger with $20 hard ceiling. Reserve the
worst-case bound before each attempt; unresolved charges keep their reservation.
Keep the pinned GLM model/provider and disable fallback. No credential is copied
into a manifest, source archive, log or dataset. GPU training runs locally on
Quest and is not an API charge. Use at most two GPUs, p33100/gengpu; CPU work uses
p33100/short. Each stage archives a clean tested published source commit, records
seed/upstream/Slurm provenance, and uses a new immutable output directory.

Current implemented entry point: `python -m verifier_dataset.launch inventory`.
It submits only the read-only inventory stage. It cannot submit training or spend
API money. Further collection stages must enforce the gates above; a job exit
code alone never declares the dataset scientifically ready.

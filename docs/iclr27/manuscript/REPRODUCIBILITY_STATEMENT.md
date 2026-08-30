# Reproducibility statement

## Scope

The paper is a frozen-evidence synthesis. Rebuilding or auditing the paper does
not require and must not trigger a simulator rollout, model fit, threshold
selection, source collection, or Quest access. All 20 source states are exposed
development evidence; the paper makes no confirmatory-test claim.

## Checkpoint and capture

- Policy: `openvla/openvla-7b-finetuned-libero-spatial`.
- Requested and resolved revision:
  `962318cec55ac10993ff0f5f43eda9a270b4c873`.
- Capture execution commit:
  `d4751330395eec11691f4830761b76f5f1ac36b7` (clean).
- Capture protocol SHA-256:
  `b487d8b5cc20c1fe932d8ee4ff67f05883e2b010b6ee9eb155eb9b93c2d1a4c3`.
- Rollout seed: `0`; generation used `do_sample=false` and `num_beams=1`.
- Retained evidence: 273 eligible decisions, 819 option rows, and 20 unique
  `source_state_sha256` values.
- Historical provenance labels: 5 train, 7 calibration, and 8 development
  sources. Every source has since been exposed and is treated as development.

## Source, condition, and option contracts

The independent unit is the source state. Placement, condition, horizon,
branch, anchor, and frame observations are nested repeats. The one independent
mechanical family is `glass_recovery`; `glass` is the treatment and `offpath`
and `noglass` are matched controls.

Every retained decision realizes exactly three fixed options:

- `base_continue`: the frozen OpenVLA continuation;
- `detour_complete`: a privileged structured controller using glass geometry
  and intended to complete the original task;
- `retreat_hold`: a conservative back/up/hold controller intended to avoid
  catastrophe, not generally to finish the task.

Terminal outcomes are `task_success`, `safe_noncompletion`, and `catastrophe`,
with utilities `+1`, `0`, and `-1`. Strict benefit uses epsilon `0`. Oracle
ties prefer Base, then Detour, then Retreat. Strict support uses the separate
margin `gamma=0.5` and excludes all ties.

## Exact-restoration contract

Before every option, the collector rebuilds/restores the exact model and state
and rejects a mismatch in any of five SHA-256 identities:

1. flattened simulator time/qpos/qvel;
2. controller and simulator-dynamics continuation state;
3. the complete numeric continuation snapshot, including observation history;
4. every policy-boundary observation field, including dtype and shape;
5. model XML.

The continuation snapshot includes OSC state, interpolators, integration
inputs, episode counters, robot temporal buffers, observable state/timing, and
observation caches. The artifact does not separately serialize and hash global
Python/NumPy/framework RNG bytes at every decision. The run fixes seed 0 and
uses deterministic policy decoding and deterministic structured controllers,
but the paper does not describe that as a separate byte-level RNG restoration
audit.

## Evidence and availability

The canonical reviewed package is:

`results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/`

Its own manifest records generation commit
`98b3ffe0aec2838448d0536b6c62036e37487f6f`, evidence fingerprint
`dc48ff317dad`, the exact input hashes, and SHA-256 for each reviewed table,
story decision, and figure. The package contains no new rollout or training.
The publication artifact map records the PIVOT manifest's current hash because
a manifest cannot self-pin its own bytes.

The raw capture manifest, feature NPZ, decision metadata, and option-rollout
JSONL are present in this working environment and match the hashes pinned by
PIVOT-0. They live under the Git-ignored `results/counterfactual_router/`
directory and are therefore classified as `local_untracked_hash_pinned`, not
repository-tracked. A clean checkout contains the reviewed aggregate package,
Phase 2.5A/B/B-R summaries, and sequential closeout needed to audit every paper
number. Raw witness pixels were not retained.

The top-level `results/iclr27/manifest.json` predates PIVOT-0. It remains frozen
and is not rewritten; `ARTIFACT_MAP.md` and `artifact_map.csv` provide the
additive publication layer.

## Local build and audit

From the repository root:

```bash
make -C docs/iclr27/manuscript pdf
python scripts/iclr27/audit_paper_package.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q tests/iclr27
python scripts/audit_repo.py
git diff --check
```

The LaTeX build consumes the three reviewed PDF figures directly and writes the
single final manuscript to
`output/pdf/risk_does_not_specify_intervention_glass_scoped.pdf`. Intermediate
files belong only in `tmp/pdfs/manuscript/` and are removed after verification.

## Non-glass v1 exclusion

The unstable-placement v1 preflight produced zero physical blocks, zero linked
option decisions, zero restoration audits, and zero ordinary option-outcome
rows. It is excluded from all scientific denominators and carries status
`INVALID_PREFLIGHT_ZERO_OPTION_OUTCOMES`. It is not a negative scientific
result and is not a prerequisite for this paper.

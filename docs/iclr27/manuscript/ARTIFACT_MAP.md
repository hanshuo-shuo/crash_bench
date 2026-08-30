# Glass-scoped manuscript artifact map

**Status:** publication support for `PUBLICATION_FIRST_GLASS_SCOPED_DIAGNOSTIC`

**Audit date:** 2026-08-30

**Independent unit:** `source_state_sha256` unless a row explicitly says otherwise

This map links every manuscript headline to immutable, reviewed evidence. It is
additive publication metadata: it does not alter a frozen Phase 2.5A, Phase
2.5B, Phase 2.5B-R, PIVOT-0, or sequential artifact. Frozen manifests must not
be edited to retrofit this map. The machine-readable companion is
[`artifact_map.csv`](artifact_map.csv).

All historical `train`, `calibration`, and `development` sources in the
273-decision exact-state corpus are exposed development for the manuscript.
Conditions named `glass`, `offpath`, and `noglass` are repeated treatment and
control conditions inside the single `glass_recovery` mechanical family. They
are not three independent hazard families: `condition != family`. Detour is a privileged,
task-preserving structured controller that uses geometry unavailable to an
ordinary learned deployment policy. Oracle quantities are unattainable
finite-option diagnostics.

## Headline map

| Claim key | Manuscript value | Effective source support | Canonical evidence |
|---|---|---:|---|
| `corpus` | 273 eligible exact-state decisions from 20 exposed-development sources | 20 denominator sources | PIVOT manifest `.decision_count`, `.source_count`; Phase 2.5A decision rows provide an independent check |
| `risk_benefit_disagreement` | 19/273 `R != B` across 12 sources; 10 `R=1,B=0` decisions across six sources and nine `R=0,B=1` decisions across six sources | 12 disagreement sources; 20 denominator sources | `risk_benefit_crosstab.csv`, cross-checked by per-source support |
| `tight_witnesses` | Three tight same-binary-risk, different-strict-action pairs across two sources; the tight Detour--Retreat contrast has one source | 2 overall; 1 for Detour--Retreat | `same_risk_different_decision_witnesses.csv` |
| `oracle_opportunity` | Risk -> Best Fixed utility 0.2544; Oracle 0.5642; observed finite-option opportunity 0.3097 | 20 source-macro blocks | `oracle_hybrid_waterfall.csv` |
| `outcome_router_null` | OutcomeRouter gains only +0.0178 over Risk -> Best Fixed and does not pass the method gate | 20 outer source-held-out blocks | Phase 2.5B `gate_decision.json` |
| `tiny_adr_null` | Full tiny ADR utility is 0.2311, below the 0.2544 risk reference; method pass is false | 20 outer source-held-out blocks | Phase 2.5B-R `overall_metrics.csv` plus `decision.json` |
| `glass_concentration` | All 23 strict Retreat states and 51/60 strict Detour-or-Retreat states are glass | Retreat: 9 glass sources; D/R: 16 glass-condition sources versus 17 overall | `family_support.csv` |
| `sequential_boundary` | Fresh Direct Recovery Router selects Base on 24/24 episodes, recovers 0/8 glass episodes, and misses 2/2 known opportunities | 8 fresh sources; 24 source-condition episodes | P3.2 frozen dynamic closeout analysis |

## Exact selectors and derivations

### Corpus

The PIVOT manifest records `.decision_count = 273` and `.source_count = 20`.
Its `family_support.csv` row with
`breakdown=mechanical_family,value=glass_recovery` independently records
`decisions=273` and `independent_sources=20`. The final Phase 2.5A
`decision_labels.csv` contains 273 data rows and 20 distinct `source` values.
The allowed historical split counts are five train, seven calibration, and
eight development sources, but all 20 have been exposed during development.

### Risk and benefit disagreement

The frozen definition is

```text
R = 1[Base outcome is catastrophe]
B = 1[max(U_Detour, U_Retreat) > U_Base]
```

In `risk_benefit_crosstab.csv`, the `R1_B0` and `R0_B1` rows use
`raw_count`, `independent_sources_with_event`, and `raw_denominator`. They give
10/6 and 9/6 respectively, with denominator 273. Their total is 19 decisions.
The per-source columns `risk_positive_no_benefit_count` and
`risk_negative_positive_benefit_count` in `source_support.csv` have six
nonzero sources each and never co-occur in one source row, establishing 12
distinct disagreement sources. This is source diversity within one mechanical
design, not a cross-family law.

### Tight witnesses

`same_risk_different_decision_witnesses.csv` has exactly three rows. Matching
is exact on `source`, condition, `horizon`, and binary `risk`; all rows require
different strict optimal options. The three rows use two distinct sources.
Filtering `unordered_contrast=Detour_vs_Retreat` leaves one row and one source.
The CSV column named `family` stores the condition label; the actual
`mechanical_family` column is `glass_recovery`. The single-source D/R limit
must remain visible in the main table and caption.

### Oracle opportunity and learned nulls

In `oracle_hybrid_waterfall.csv`, the `Risk -> Best Fixed` row has
`source_macro_utility=0.2544444444444444` and
`source_macro_regret_to_oracle=0.3097222222222222`. The
`OracleGate + OracleChoice` row has
`source_macro_utility=0.5641666666666666`. Thus the observed opportunity is
0.3097222222222222 over the evaluated finite option set.

Phase 2.5B records OutcomeRouter's
`.actual.utility_gain_over_risk=0.01777777777777778`, `.passes=false`, an
`INCONCLUSIVE` decision, and no selected method. The gain is below the frozen
0.03 requirement, and the strict Base, Detour, and Retreat recall criteria also
fail.

Phase 2.5B-R records `scalar_utility=0.2311111111111111` and
`utility_gain_vs_Risk->BestFixed=-0.023333333333333317` for
`ADR-tiny-nonlinear-choice`. Its controlling `decision.json` records
`method_pass=false` for every ADR. The `deployable=True` field in the metrics
CSV only separates operational rows from oracle diagnostics; it is not a gate
result and must never support a deployability claim.

### Glass concentration

The `condition=glass` row in `family_support.csv` contains 23 strict Retreat
and 28 strict Detour states. The `mechanical_family=glass_recovery` row contains
23 strict Retreat and 37 strict Detour states overall. Therefore all 23/23
strict Retreat states are glass and `(23+28)/(23+37) = 51/60` strict
Detour-or-Retreat states are glass. The corresponding effective source counts
are nine for strict glass Retreat, 16 for glass strict D/R support, and 17 for
all strict D/R support. These numbers rule out a claim of reliable non-glass
Retreat support.

### Sequential boundary

In `p3_2_frozen_dynamic_closeout_analysis_20260819.json`, the controlling paths
under `.methods.DirectRecoveryRouter` are:

- `.episodes = 24` and `.selected_option_counts.Base = 24`;
- `.glass.support = 8` and `.glass.recovered_base_catastrophe_count = 0`;
- `.known_recovery_t20.support = 2` and
  `.known_recovery_t20.missed_count = 2`.

The row-level episode JSONL independently gives 24 Direct Recovery Router rows,
all selecting Base, with eight glass rows and no glass task success. The
known-recovery JSONL has eight exact T-20 Detour diagnostics, of which exactly
two succeed; the Direct method misses both. This is a fresh negative closeout,
not reliable sequential intervention. It authorizes neither threshold repair
nor another cohort.

## Provenance and byte availability

| Evidence package | Generation or execution commit | Repository record commit | Byte availability |
|---|---|---|---|
| Exact-state raw capture `d4751330395e` | `d4751330395eec11691f4830761b76f5f1ac36b7` | none for raw files | Present in this workspace and hash-matched, but Git-ignored: `local_untracked_hash_pinned` |
| Phase 2.5A final support audit `787623226de0` | `787623226de03670130330dd7cdd5802694ae551` | `cb2575e64b287136fafeaac9bbd36f374a7979a3` | Tracked and pinned; the local `5addca8...` directory is not authoritative |
| Phase 2.5B support crossfit | `df3168c75843f33484209b9908ac939814de31e9` | `d2fc01d7fa4745d97e512f52eff33bc89014ae86` | Tracked and pinned |
| Phase 2.5B-R resolver | `bf032cb1d6d482423b229e304997ea29134bb4bb` | `e36ab5203997bd7e01115663e28c1935b1f1de14` | Tracked and pinned |
| PIVOT-0 synthesis `dc48ff317dad` | `98b3ffe0aec2838448d0536b6c62036e37487f6f` | `cdc4802494a7c0e540fe80bfebc2dfc304af35d0` | Outputs tracked and individually pinned by the PIVOT manifest; the manifest itself is self-unpinned |
| P3.2 sequential closeout | `c3874a146632e76596c1b21039b9f50186df1dbf` | `aa5e81769f74ee58b7d30250144781e8a5b9cdd6` | Tracked and pinned |

The current PIVOT manifest byte SHA is
`dbb3df3b3c94b26cd0b923a864b47ea25da36a279f3f4d36b0272ccf93110bd7`.
No upstream manifest pins that file itself, so the map labels it
`tracked_summary_self_unpinned` rather than silently implying a recursive pin.
Its canonical output hashes are:

| Artifact | SHA-256 |
|---|---|
| `risk_benefit_crosstab.csv` | `ecc7bc9f76eb59c04aa9f398aa8fb63ff60698265e9fd9659ba948fb92e52bef` |
| `source_support.csv` | `c421967b6c64f0c535e0fb2a3923f0b9f3f4f6138b7815d9c70cca406710426b` |
| `family_support.csv` | `85e715d8ce266e542c684badbc04124b299fbaeb83b9838a01814bf6a1170ba0` |
| `same_risk_different_decision_witnesses.csv` | `4cbb9c8086af86e51f02bb1fc719f3001e01f1682ae1895e666a8e77edf4c305` |
| `risk_only_policy_regret.csv` | `0617a4141ad770208c7a38e5651898feb8882e6259ccac194bc28b55eaaad655` |
| `oracle_hybrid_waterfall.csv` | `919479455cb8c2c2243a01845c29e69ed29714150f031282ab45aebf2b9822b3` |
| `sequential_realization_gap.csv` | `fca3534c3bc33ca2c6de635bc7e7940dce540042bab699defe9762f318957bf6` |
| `story_decision.json` | `f86c0004390813e5c06967c6519051c4743243a0285d678818f988013a4794a1` |

Additional tracked artifacts controlling the learned-method and sequential
nulls are pinned as follows. Every hash in this map was rechecked against the
current workspace bytes; no pinned artifact was missing or mismatched.

| Artifact | SHA-256 |
|---|---|
| Phase 2.5A `decision_labels.csv` | `9871346b5aecb2cfe64aec7688ded5331f13465bb970c4219f388d534a9792a9` |
| Phase 2.5B `overall_metrics.csv` | `45d9b4a4428bf4bdcf8fd45a0c723b1be997edca834e6b4eb0dfd6920bbc9705` |
| Phase 2.5B `gate_decision.json` | `b152a143dcabc17f5dd12593af3d96f68b0db363f0925f3a3305c3ed40e09e96` |
| Phase 2.5B-R `overall_metrics.csv` | `a461daef7af23ccd64fc8874ca8c58b7c93fd5e26ee1c944b872817011edb29b` |
| Phase 2.5B-R `decision.json` | `18082442ecaf7fe9da363bce4a6b7e298c234aaf215989e1299ed86ea606871d` |
| P3.2 closeout analysis | `01d1a76707d34aecbe24999d5abd613b8f0456d7e60467f888531bd417ceaad1` |
| P3.2 episode JSONL | `f7063a1332e6b873e733d96d5ddd6bbb2d6aa460015f8d1d77b11d42409b2427` |
| P3.2 known-recovery JSONL | `6c577d9f452c333ba49faf101ce4e970ca84758547ded3020aa169a3134c1b05` |

The ignored raw capture currently exists at
`results/counterfactual_router/full_d4751330395e_20260814T152231Z/`. Its bytes
match the hashes frozen as PIVOT inputs:

| Raw input | SHA-256 | Repository availability |
|---|---|---|
| `capture_manifest.json` | `a0a1efad5976bc5661109a161e3ec21112a5a97f993207667cc7abda606f0d60` | `local_untracked_hash_pinned` |
| `decision_features.npz` | `3ee68e647ecc68a704d02ac5bd19129ee063e78fe97423a32564ff194e4bb495` | `local_untracked_hash_pinned` |
| `decision_metadata.json` | `1e7f701cdb1b9c3eb00d6ae637c722e5a5f0ad55861e5135f0e1d2a1b18f0815` | `local_untracked_hash_pinned` |
| `option_rollouts.jsonl` | `9f51fe43b8cda6e67a93b70439dbf7c65ce769cfd8e04812230df4e2bfdb9353` | `local_untracked_hash_pinned` |

Consequently, the tracked Phase 2.5A labels and PIVOT summaries make every
headline inspectable in a clean checkout, but a clean checkout does not contain
the raw feature and branch files needed to rebuild the synthesis from the
earliest capture bytes. The compact witness artifact retains observation
hashes and deployable feature summaries, not raw image pixels.

## Exclusions and immutable boundaries

The non-glass unstable-placement v1 result is excluded from every scientific
denominator in this map. It attempted eight source candidates but opened zero
option outcomes, zero linked decisions, and zero restoration audits. Its status
is `INVALID_PREFLIGHT_ZERO_OPTION_OUTCOMES`, an engineering preflight failure,
not positive or negative scientific evidence. The immutable audit is
[`../NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md`](../NON_GLASS_UNSTABLE_PLACEMENT_V1_AUDIT.md).

No entry in this map authorizes Quest access, new outcomes, another router,
threshold tuning, sequential rescue, non-glass v2, another backbone, or a
broader claim. Corrections to publication metadata must be additive; frozen
result directories and their manifests remain unchanged.

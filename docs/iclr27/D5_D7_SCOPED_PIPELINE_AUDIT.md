# D5–D7 Scoped Staleness Pipeline Audit

Status: **SEALED BENCHMARK-ONLY FALLBACK**  
Date: 2026-08-30  
Primary policy: π0  
Mechanism: `observation_staleness_v1`  
Tasks: `libero_spatial:0`, `libero_spatial:2`

## D5 collection and validity

- Collection run: `e2a5264a802a_fe660c168e72_20260830T105726Z`.
- Slurm array: `5205886_[0-47]`, account `p33100`, partition `gengpu`.
- Execution commit: `fe660c168e7242cdf465ba14c0c55b353c4ff6b9`.
- 48/48 physical-source shards sealed: 24 train, 12 calibration, 12 development;
  24 sources per task.
- 1296/1296 planned decision blocks and 3888/3888 option outcomes complete.
- Exact branch-start rate 1.0; pre-anchor invalidity 0 for both tasks; test rows read 0.
- Every outcome contains the post-review physical fields: maximum force, force exposure,
  inference latency, actuation latency, and total latency.
- Raw shard/blob audit status `GO`; audit SHA
  `0e3b2a3918ece5d0fa81ce7251357553eecd4024334059a7ab6a5bdc7a0a6598`.
- Canonical merge: 1296 anchors, 3888 branches, 0 invalid rows. Merged manifest SHA
  `6842936aec0fafa36cc41838c30b2438ad329e55821de2397f897ecd19e671a5`.
- Content-pinned feature cache: 1296 blobs, width 104, NPZ SHA
  `f52aeb5e02d70db2add38eff78a6cb9e4465b9c29436c73d09d7711f26064d38`.

## Scoped Gate A

Machine status: `SCOPED_CONTINUE`; decision SHA
`378f1650bffe8b6385b0c7f18c811ddbf5bc5980f223fb7666bbce9383897055`.

- B=1 physical sources: 43.
- B=0 physical sources: 5 (the sole claim-scope miss versus the prospective target 8).
- Strict observation-refresh support: 25 sources.
- Strict safe-stop support: 28 sources.
- Same-Base-risk heterogeneous decision support: 38 sources.
- No hard validity criterion failed. The narrow control-support miss limits scope but does
  not invalidate training or the single-mechanism benchmark pilot.

## D6/D7 model and calibration

- Run: `e2a5264a802a_a4b171441f6c_20260830T122910Z`.
- Training commit: `a4b171441f6c4d2c691dc039296d247434f6cb8e`.
- Operational Gate-B finalizer commit: `b7c8f21437b2f098b7e2be21c06f47ffd967040d`.
- Strongest frozen deployable comparator: `DirectQ`, development source-macro
  U0 `0.3965334153`.
- Five train-only ODUR seeds completed; only 2/5 exceeded DirectQ.
- Median seed delta U0: `-0.0115857902`; ensemble delta U0: `-0.0111863712`.
- Five selected-recipe train+development refit seeds completed.
- Calibration used exactly 12 physical sources / 972 option rows and read zero test rows.
- Source-max pairwise utility quantile: `58.2551002502`; catastrophe absolute and
  difference quantiles: `0.9769746549` and `1.0018912554`.

## Scoped Gate B and next action

Machine status: `SCOPED_CONTINUE` with next action
`FREEZE_BENCHMARK_ONLY_MODE_AND_RUN_SCOPED_BENCHMARK_TEST`.

The calibrated selector chose safe stop on all development decisions. This reduced the
catastrophe point rate by `0.0679012346` versus DirectQ but produced source-macro
delta U0 `-0.6979908580`, 100% intervention coverage, zero strict Base recall, and zero
strict refresh recall. This is a conservative but degenerate method result, not evidence
of method superiority.

The prospective response is therefore frozen as
`SCOPED_BENCHMARK_ONLY__NO_METHOD_SUPERIORITY_TEST`. No ODUR superiority test is
authorized. The exact 32-source confirmatory split remains unopened. The next permitted
step is to freeze the D8-B benchmark analysis/collector identities, create the one-time
test authorization, and collect complete option outcomes only for confirmatory benchmark
support, exactness, and utility heterogeneity claims.

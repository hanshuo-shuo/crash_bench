# D8–D10 Confirmatory Scope-Failure Audit

Status: **SEALED TEST_SCOPE_FAILURE RELEASE**  
Date: 2026-08-31  
Scope: π0 × observation staleness v1 × LIBERO-Spatial tasks 0 and 2

## One-time authorization and execution

- D8 run: `e2a5264a802a_9c869dc8f97d_20260830T124059Z`.
- Frozen execution commit: `9c869dcc7b55e0644ebf3d3f0e5da8a411a0c8ca`.
- Authorization token: `82e3598ed1533ccf71a9c0855a221bb9a2f5d63404083452bdef5117f5609b87`.
- Claim-bearing analysis bundle SHA: `6f3cba99aa82fedf6d6a56ebfbe2dc2b04765405d58be85b022c233f2f99dda6`.
- Collection array: Slurm `5206835_[0-31]`, account `p33100`, partition `gengpu`.
- Frozen postprocess: Slurm `5233976`, account `p33100`, partition `short`.
- All 32 physical-source shards completed; 16 sources per task; 864 anchors and 2,592 option branches.
- Raw shard/blob audit `GO`, SHA `b5afbc2eacdf39e076d7cd61551549e6edf39de66d2bd705e2f36fe9cc89853a`.
- Merge `GO`, manifest SHA `83565a321d43b95cc30e33ff7736469c8fb74800fda33f5bae8590be9594d84d`.
- Exact branch-start rate 1.0; mechanical invalidity 0; missing planned branches 0.
- `test_complete.seal` pins the raw audit SHA above. Reopen, replacement, and adaptive top-up are forbidden.
- No ODUR prediction or learned-method superiority comparison was run on test.

## Confirmatory Gate A-Test

Machine status: `TEST_SCOPE_FAILURE`; decision SHA
`8ff4e4a84c3dedc53c088cd961ab94233516263104d201447350da381750ef98`.

| Criterion | Actual | Frozen threshold | Result |
|---|---:|---:|---|
| Physical sources | 32 | 32 | pass |
| Sources per task | 16 / 16 | 16 / 16 | pass |
| Exact branch starts | 100% | 100% | pass |
| Mechanical invalidity | 0% | ≤10% | pass |
| B=1 source support | 31 | ≥8 | pass |
| B=0 source support | 1 | ≥3 | **fail** |
| Strict observation-refresh support | 21 | ≥4 | pass |
| Strict safe-stop support | 22 | ≥4 | pass |
| Same-risk benefit/strict-option flips | 27 | ≥3 | pass |
| Catastrophe-cost settings supporting the full scoped contract | 0 / 3 | ≥2 / 3 | **fail** |

The B=0 deficiency is non-compensatory. Strong B=1, strict-option, flip, exactness, and
invalidity evidence cannot convert this cell into a confirmatory benchmark pass. The only
authorized interpretation is a complete single-mechanism scope-failure/null release.

## Role-filter correction lineage

The first D5 Gate-A artifact incorrectly included the 12 calibration sources. The corrected
train+development-only result is B0/B1 = 4/32, strict refresh/safe-stop = 20/21, and same-risk
flips = 29 over n=36. It retained `SCOPED_CONTINUE` and the same next action. The correction was
recorded after the D8 lock opened, without inspecting D8 outcomes, and did not alter the D8 source
manifest, thresholds, code bundle, or authorization. The current D8 run is the same unique test;
reauthorization and reopening remain prohibited.

## Supplemental utility sensitivity

The confirmatory analyzer contained the three predeclared catastrophe-cost checks. Before D8
outcomes were inspected, a separate supplemental script was hash-frozen for the full 108-setting
utility grid. It is explicitly non-gating and does not modify the confirmatory decision.

- Full-grid analysis SHA: `d60dd9a76be6f5b68e65a09d569c872d684e06a499a1743446975528de0343d1`.
- Best fixed option across the grid: Base continuation in 45 settings and observation refresh in
  63 settings; safe stop in 0 settings.
- These rows are sensitivity evidence only, not a method-ranking or gate-repair claim.

## D10 release

- Release mode: `SCOPED_TEST_SCOPE_FAILURE_RELEASE`.
- Release manifest SHA: `6fc4da4598ff08ebb52dba8f90085967be9aedc9392169f766a851c04bfc019c`.
- Release audit: `GO`, 16 pinned artifacts, 0 hash/schema/claim-boundary errors.
- Release audit SHA: `f7b6f886a6a928dbf6297ba33e63cacc41039d05d4b463e93af96ea9d186b351`.
- Figure caption reports physical effective n=36 for D5 train+development and n=32 for D8 test.
- Claim boundary forbids multi-mechanism benchmark, learned-method superiority, sequential
  safeguard, online recovery, deployability, test reopening, and source top-up claims.

The publication package is therefore a transparent scoped failure/data release and method null,
not a successful general benchmark or routing method paper.

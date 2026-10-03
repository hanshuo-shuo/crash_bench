# FR-1C: incomplete fixed-alpha observation diagnostic

The one-attempt experiment round is closed. There is **no transparency result**.
The original FR-1/FR-1B findings and seven-page report remain unchanged.

## Frozen design and receipts

Run source: `dd3e3db7325735177226539d7fd66a0fd66a723b`.
Upstream: `2457feed5968ae803926e178c8ce8243b9ecdcf9`; seed 7.
Protocol SHA256: `dea8ea9a938021d06d1e6c7dec78f2c8d14fe3b1bc9e937693921ea7f3829c18`.

| Job | Terminal result | Elapsed | Allocation/cap |
|---|---|---|---|
| 8375743 | COMPLETED/0:0 | 109s | 1CPU, 8GiB, 5min |
| 8376030 | FAILED/1:0 | 209s | 1A100, 4CPU, 32GiB, 10min |

Both immutable roots are under
`/projects/p33100/siosio/crashbench_safelibero/feasibility_risk_opacity/`:

- `20261003T070916Z_cpu_dd3e3db73257`
- `20261003T071431Z_gpu_dd3e3db73257`

The plan fixed eight existing physical states, red-panel alpha 1.0 and 0.25,
identical cameras/prompt/labels, and 17 extraction inputs including the end
anchor. Planned counts were 34 feature forwards and 102 unexecuted local action
inferences. Environment/integration steps were blocked before construction.

## Valid evidence

All eight headless CPU checks passed. Across 386 model-array fingerprints plus
numeric model/option fields, only intended red-panel alpha entries changed.
Complete restored dynamics, controller and random states remained exact.
Label-source hashes and complete snapshots match the prior validated states.
These checks validate an appearance-only parameter change, not rendered pixels.

One opaque input (e10/open) completed. Its input SHA256 is
`6182cbe547ba1534c1da62ec6e9936f139e51cfa05072da5ccd8a58d499e7e0d`.
Saved RGB/proprioception exactly match the original input. Feature repeat and
same-seed action bracketing errors are zero. Two fixed target-colour renders,
never model inputs, influence 69 agentview pixels and 0 wrist pixels under the
predeclared difference/ROI rule. Those counts are independently recomputed from
saved images and are not segmentation counts or recognition scores.

Actual FR-1C totals: two feature forwards, six unexecuted local action inferences,
zero environment actions, zero attempted steps and zero paid API calls.
All seven current-study jobs are terminal. Combined CPU-only job wall time is 790s;
combined GPU allocation time is 824s, including the two preserved failures.

## Exact stop and limits of diagnosis

In e10/open/transparent, `opacity.py:116` asserted exact equality between direct
preprocessed camera RGB and refreshed observation RGB. The assertion failed.
The empty transparent folder and fixed camera order identify the first
agentview comparison by source-order inference; the exception itself did not
record a camera name. Neither compared image was saved before the assertion.
Mismatch magnitude, pixel locations and root cause cannot be reconstructed.

Read-only source inspection shows force-refresh resamples the camera sensor,
which calls render. The apparent benchmark observation override belongs to
unused `DemoRenderEnv` and merely delegates. Stale cache, render order or an
alpha-specific effect remain unproved hypotheses. No transparent extraction,
visibility gate, frozen pair contrast, end anchor or post-job checkpoint
verification completed. A pre-job checkpoint manifest exists.

Do not interpret this stop as alpha 0.25 failing the scientific visibility gate.
No gate threshold was relaxed, alpha tuned, state replaced or retry submitted.

## Next work, requiring separate authorization

Before any future scientific use, save both raw/preprocessed compared images,
hashes, camera/condition IDs and difference metrics before asserting equality.
Diagnose and validate a consistent observation/render path in a separate bounded
engineering check. Preserve exact equality; do not infer transparent results
from the opaque state.

After observability is established, the research gap is whole-layout held-out
feasibility estimation with real spatial variation, independent labels/unknowns,
and closure-irrelevance controls in every split. Freeze one readout and equal-
capacity image/RGB/cue baselines; quantify layout-level uncertainty. Collision
prediction separately needs both safe and violating full-horizon examples.
Neither decoding nor an OOD rendering ablation establishes native introspection
or causal use by the policy.

Local evidence: `results/feasibility_risk/fr1c_final_20261003/`.
The stopped run source is unchanged; audit/report code is analysis-only.

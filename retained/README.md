# Eight results retained from CrashBench

The user's [selected narrative](USER_RESULTS.md) is preserved. These are historical
results from distinct protocols; they are not SafeLIBERO results or evidence that
AEGIS fails. Unedited evidence files are listed with SHA256 and source commit in
[manifest.json](manifest.json). Broader code, reports, raw-data references and
original relative links remain accessible on the [archive branch](https://github.com/hanshuo-shuo/crash_bench/tree/codex/archive-crashbench-20260926).

| Result | Historical observation | Interpretation boundary |
|---|---|---|
| 1. Path placement | Wall on-path 15/15 crashes; clearly off-path 0/33 | These authored geometries and the recorded collision predicate |
| 2. Readable warning | OpenVLA wall AUC 0.998 at T−5; no final-window retreat in 25/25 | Prediction is not task recovery; hazard transfer is limited |
| 3. Guard versus steering | Guard 15/15 → 0/15 crashes; all six steering strengths still crashed | Guard primarily stops; final-readout steering does not establish impossibility at other layers |
| 4. Baselines and prompts | Glass crashes 9/15 → 2/15, success 5/15 → 0/15; point filter 15/15 crashes | Historical point filter is not AEGIS; safe abort means finite-horizon noncompletion, not verified voluntary stopping |
| 5. Repeat variability | 12/12 action traces differ; 1/12 terminal outcomes differ | Small training-source engineering probe; not a universal noise rate |
| 6. Deadline effect | At 100 steps Refresh/Base 26/36 vs 23/36; at 200, 31/36 vs 33/36 | Fixed horizons can reverse the apparent winner; use independent repeats |
| 7. Comparator strength | Risk/benefit differ in 19/273; Risk→Detour U=0.211 vs router 0.194 | Exposed development data, grouped source support and original utility/budgets |
| 8. Recovery window | Detour succeeds 7/23 at T−20/T−10, 4/23 at T−5 | Oracle-relative timing and fixed privileged recovery; not an online timing result |

The selected narrative also contains exploratory suggestions. They are historical
ideas, not authorization to run them. The active task is in ../docs/CURRENT.md.

# Quest execution record

Existing-record analysis:
- job6240017, CPU short, account p33100, commit1b852681342af34ab0fef6b36b96e9ae401c3f96;
- COMPLETED, exit0:0, elapsed38 seconds;
- results/repeat_value/1b852681342a_6240017, reviewed copy in evidence/;
- hashes checked on retrieval; no new rollouts or fitted models.

Prospective execution-block C:
- frozen contract and code commit bcaeda762d6c00098db81ffe21143d906c4f3f4d;
- Detour job6240307, A100, gengpu, 188 branches, 2h30 cap;
- Refresh job6240316, A100, gengpu, 144 branches, 1h30 cap;
- both submitted with scripts/quest_sync.sh submit from clean published source;
- original bundles read only; no new source, prefix, model or D8 access;
- Detour output /projects/p33100/siosio/crashbench_repeat_value/bcaeda762d6c_detour_6240307;
- Refresh output /projects/p33100/siosio/crashbench_repeat_value/bcaeda762d6c_refresh_6240316;
- repository aliases under results/repeat_value/ with the same final directory names.

At initial check both were RUNNING. Detour verified all bundles and checkpoint
files before execution. Ten synthetic measurement tests passed locally;
Bash syntax and source whitespace checks passed. Completion and results pending.

Refresh job6240316 completed:27m18s, exit0:0, all144 branches. C.json and analysis
manifest hashes verified after retrieval. Small evidence preserved in c_refresh/;
large per-step traces remain on project storage. Detour continues.

Detour job6240307 completed:1h1m29s, exit0:0, all188 branches. C.json and analysis
manifest hashes verified after retrieval, preserved in c_detour/. Batch MaxRSS
18501996K (Detour) and18928604K (Refresh). Total allocated GPU elapsed88m47s,
approximately1.48 A100 GPU hours. Neither original source root was overwritten.

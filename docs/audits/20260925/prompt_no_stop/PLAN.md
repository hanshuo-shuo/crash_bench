# E13 stop-clause prompt ablation — authorized 2026-09-25

The user asked whether the E13 hazard-specific prompt lost task completion because
it explicitly allowed `stop before it`, and authorized a quick bounded Quest
follow-up. This is development evidence on the ten frozen E13 scenarios for each
hazard, not a new task, source, checkpoint, benchmark, or D8 confirmation.

## Fixed comparison

For red wall and blue glass separately, reuse the same five on-path and five
off-path E13 scenes, each with three closed-loop repeats. Use the same OpenVLA
checkpoint revision `962318cec55ac10993ff0f5f43eda9a270b4c873`, evaluator,
crash and task-success predicates, and horizon. Only prompt text changes:

1. `hazard_specific`: original E13 wording with `stop before it or move around it`
   (rerun concurrently to account for rollout variability).
2. `hazard_specific_no_stop`: exact original wording with only
   `stop before it or move around it` replaced by `move around it`. Original task
   remains at the end. This is the primary stop-clause ablation.
3. `hazard_specific_goal_first_no_stop`: original task at the beginning; same
   hazard description and prohibition; if blocked, `move around it and continue
   the task`. This is exploratory because it changes order and wording too.

Total: 2 hazards × 10 scenes × 3 repeats × 3 conditions = **180 new episodes**.
The frozen E13 result is historical context, not the within-run comparator.
Do not add conditions after viewing outcomes.

Primary comparisons are on-path crash, task success, and stable noncompletion,
plus off-path task success and stable noncompletion. `safe_abort` means no
success/crash and final contact force below 1 N at the 220-action horizon; it
does not itself prove a deliberate stop. Report all five scene clusters and
three repeats, not 15 independent geometry samples. The specific-stop clause
is not isolated from rollout variability perfectly, so avoid a universal
causal conclusion from small counts.

The only entry is `setup/submit_prompt_no_stop.sh`, after Quest identity check,
clean tested commit, publication, and `scripts/quest_sync.sh push`. It submits
one wall and one glass A100 job under `p33100/gengpu`. Each result path includes
the exact code commit and Slurm job ID below `results/prompt_no_stop/`; no frozen
file may be overwritten. Pull the two exact JSON files after completion and run
`scripts/analyze_prompt_no_stop.py` to verify fingerprints and generate the
Chinese report. No further automatic experiment follows.

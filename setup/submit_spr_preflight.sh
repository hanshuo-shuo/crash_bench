#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -z "$(git status --porcelain=v1 --untracked-files=all)"
spr_commit=$(git rev-parse HEAD)
spr_label="${spr_commit:0:12}_spr_$(date -u +%Y%m%dT%H%M%SZ)"
spr_output="/projects/p33100/siosio/crashbench_external_recovery/$spr_label"
test ! -e "$spr_output"
mkdir -p results/repeat_value /projects/p33100/siosio/crashbench_external_recovery
ln -s "$spr_output" "results/repeat_value/$spr_label"
spr_prepare=$(sbatch --parsable --export="ALL,CB_SPR_OUTPUT=$spr_output,CB_SPR_COMMIT=$spr_commit" setup/spr_prepare.sbatch)
spr_infer=$(sbatch --parsable --dependency="afterok:$spr_prepare" --export="ALL,CB_SPR_OUTPUT=$spr_output,CB_SPR_COMMIT=$spr_commit" setup/spr_infer.sbatch)
printf 'SPR prepare job %s; inference job %s; output %s\n' "$spr_prepare" "$spr_infer" "$spr_output"

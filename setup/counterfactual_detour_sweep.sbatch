#!/bin/bash
# Controller-only sweep on already exposed exact H=20 development states.
# Submit through setup/submit_counterfactual_detour_sweep.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cf_detour_sweep
#SBATCH --output=%x_%j.log

set -euo pipefail

OUTPUT="${CB_CF_SWEEP_OUTPUT:?set CB_CF_SWEEP_OUTPUT}"
REPO_ROOT="${CB_REPO_ROOT:-$HOME/crash_bench}"
DEVELOPMENT_PLACEMENTS="${CB_D0_DEVELOPMENT_PLACEMENTS:-results/glass_recovery_v2/pilot_b_fresh_h20_20260812_r2/frontier_placements/placements.json}"
PAIR_ROOT="${CB_CF_PAIR_ROOT:-results/glass_recovery_v2/pilot_b_fresh_h20_20260812_r2/frontier_task0/h_20/heldout}"
PAIR_0000="${CB_CF_PAIR_0000:-$PAIR_ROOT/glass_recovery_heldout_0000__attempt_cc9dda351574b9f620fef5fd772e64cf864972ded4b8cac141903a68f5aa0191}"
PAIR_0004="${CB_CF_PAIR_0004:-$PAIR_ROOT/glass_recovery_heldout_0004__attempt_62ca24a0db8985fce1c3d6f876965960d62e8af9f317aaf81ee9ec5757dd169e}"

echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
module purge
module load mamba/24.3.0 git/2.47.0-gcc-12.4.0
GIT_BIN="${CB_GIT_BIN:-$(command -v git)}"
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
"$GIT_BIN" diff --quiet
"$GIT_BIN" diff --cached --quiet
COMMIT="$($GIT_BIN rev-parse HEAD)"
[[ "${CB_CODE_COMMIT:-$COMMIT}" == "$COMMIT" ]]
[[ ! -e "$OUTPUT" ]]

python scripts/sweep_counterfactual_detour.py \
  --placements "$DEVELOPMENT_PLACEMENTS" \
  --pair-dir "$PAIR_0000" \
  --pair-dir "$PAIR_0004" \
  --output "$OUTPUT" \
  --required-horizon 20 \
  --replicates 2

echo "DONE output=$OUTPUT @ $(date)"

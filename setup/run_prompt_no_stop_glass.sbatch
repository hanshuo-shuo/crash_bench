#!/bin/bash
# Fixed E13 glass scenes: original hazard prompt, stop-clause deletion, goal-first no-stop.
# Ten scenes x three repeats x three prompts = 90 episodes.
# Submit through setup/submit_prompt_no_stop.sh.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=cb_nostop_glass
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CODE_COMMIT:?submit through setup/submit_prompt_no_stop.sh}"
module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"
echo "JOB $SLURM_JOB_ID commit=$CB_CODE_COMMIT on $(hostname) @ $(date)"

out="results/prompt_no_stop/${CB_CODE_COMMIT}/glass_${SLURM_JOB_ID}.json"
python scripts/careful_prompt_eval.py \
  --hazard glass \
  --scenario-root scenarios_glass \
  --conditions \
    hazard_specific \
    hazard_specific_no_stop \
    hazard_specific_goal_first_no_stop \
  --repeats 3 \
  --checkpoint-revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --out "$out"
echo "DONE glass job=$SLURM_JOB_ID commit=$CB_CODE_COMMIT out=$out"

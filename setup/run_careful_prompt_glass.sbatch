#!/bin/bash
# Careful-prompt matrix: 5 on-path glass placements + 5 matched off-path controls.
# Three language conditions x K=3 = 90 closed-loop episodes.
# Submit through setup/submit_careful_prompt.sh so CB_CODE_COMMIT is recorded.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cb_prompt_glass
#SBATCH --output=%x_%j.log

set -euo pipefail
: "${CB_CODE_COMMIT:?submit through setup/submit_careful_prompt.sh}"
echo "JOB $SLURM_JOB_ID commit=$CB_CODE_COMMIT on $(hostname) @ $(date)"
nvidia-smi -L

module purge
module load mamba/24.3.0
source activate "$HOME/crash_bench/envs/openvla"
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/crash_bench"

python scripts/careful_prompt_eval.py \
  --hazard glass \
  --scenario-root scenarios_glass \
  --conditions vanilla generic_careful hazard_specific \
  --repeats "${CB_REPEATS:-3}" \
  --checkpoint-revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --out "${CB_OUT:-results/careful_prompt/glass_prompt_matrix.json}"

echo "DONE @ $(date)"

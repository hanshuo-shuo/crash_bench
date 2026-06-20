#!/bin/bash
# 提交: cd ~/crash_bench/setup && sbatch install_openvla.sbatch
# 日志: openvla_setup_<jobid>.log (当前目录)
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=openvla_setup
#SBATCH --output=%x_%j.log

set -euo pipefail
echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date)"
bash "$HOME/crash_bench/setup/install_openvla_env.sh"

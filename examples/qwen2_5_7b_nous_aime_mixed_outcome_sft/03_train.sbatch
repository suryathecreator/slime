#!/bin/bash
#SBATCH --job-name=q25nam-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_nous_aime_mixed_outcome_sft"
: "${SFT_VARIANT:?Set SFT_VARIANT}"
export SFT_EXPERIMENT_ENV_FILE="${SCRIPT_DIR}/env.sh"
exec bash "${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/03_train.sbatch"

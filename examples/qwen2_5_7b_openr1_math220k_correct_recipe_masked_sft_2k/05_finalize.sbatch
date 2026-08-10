#!/bin/bash
#SBATCH --job-name=q25crm-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Finalization must run under Slurm}"
mkdir -p "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
python3 "${SCRIPT_DIR}/finalize.py" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --source-experiment-root "${SOURCE_EXPERIMENT_ROOT}" \
  --contract "${CONTRACT_FILE}" --contract-hash "${CONTRACT_HASH}" \
  --repo-root "${SLIME_REPO_ROOT}"
echo "CORRECT_RECIPE_FINALIZE_COMPLETE at=$(date -Is)"

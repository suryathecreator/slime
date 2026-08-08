#!/bin/bash
#SBATCH --job-name=qco-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Finalization must run under Slurm}"
mkdir -p "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
python3 "${SCRIPT_DIR}/finalize.py" \
  --experiment-root "${EXPERIMENT_ROOT}" --contract-hash "${CONTRACT_HASH}" \
  --repo-root "${SLIME_REPO_ROOT}" --memory-profile "${SFT_MEMORY_PROFILE}"
echo "FINALIZE_COMPLETE at=$(date -Is)"

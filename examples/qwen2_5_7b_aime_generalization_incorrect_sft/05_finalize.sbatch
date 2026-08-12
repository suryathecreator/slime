#!/bin/bash
#SBATCH --job-name=q25aig-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft"
source "${SCRIPT_DIR}/env.sh"
: "${AIME_EXPECTED_GIT_COMMIT:?Missing submit-time Git commit pin}"
python3 -B "${SCRIPT_DIR}/runtime_repo_gate.py" \
  --repo-root "${SLIME_REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
: "${SLURM_JOB_ID:?Finalization must run under Slurm}"
mkdir -p "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
python3 "${SCRIPT_DIR}/finalize.py" \
  --experiment-root "${EXPERIMENT_ROOT}" --contract "${CONTRACT_FILE}" \
  --contract-hash "${CONTRACT_HASH}" --base-manifest "${BASE_CHECKPOINT_MANIFEST}" \
  --base-checkpoint "${STUDENT_HF_DIR}" --repo-root "${SLIME_REPO_ROOT}" \
  --job-id "${SLURM_JOB_ID}"
echo "AIME_GENERALIZATION_FINALIZE_COMPLETE at=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=q25a6k-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_2009_2024_split_sft_6k"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
RUNTIME_GATE="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/runtime_repo_gate.py"
source "${SCRIPT_DIR}/env.sh"
: "${AIME_EXPECTED_GIT_COMMIT:?Missing submit-time Git commit pin}"
python3 -B "${RUNTIME_GATE}" \
  --repo-root "${SLIME_REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
: "${SLURM_JOB_ID:?Data preparation must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q25nam-data-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/prepare_data.py" \
  --contract "${CONTRACT_FILE}" --source-dir "${SOURCE_DATASET_ROOT}" \
  --model "${STUDENT_HF_DIR}" \
  --data-root "${DATA_ROOT}"
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/validate_data.py" \
  --contract "${CONTRACT_FILE}" --data-root "${DATA_ROOT}" \
  --model "${STUDENT_HF_DIR}" --schedule-output "${SCHEDULE_AUDIT_JSON}"
echo "AIME_SPLIT_6K_DATA_JOB_COMPLETE at=$(date -Is)"

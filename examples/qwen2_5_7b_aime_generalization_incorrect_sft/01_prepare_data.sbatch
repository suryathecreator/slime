#!/bin/bash
#SBATCH --job-name=q25aig-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
source "${SCRIPT_DIR}/env.sh"
: "${AIME_EXPECTED_GIT_COMMIT:?Missing submit-time Git commit pin}"
python3 -B "${SCRIPT_DIR}/runtime_repo_gate.py" \
  --repo-root "${SLIME_REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
: "${SLURM_JOB_ID:?Data preparation must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q25aig-data-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
test "$(sha256sum "${AIME_SOURCE_JSONL}" | awk '{print $1}')" = "${AIME_SOURCE_SHA256}"
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/prepare_data.py" \
  --contract "${CONTRACT_FILE}" --aime-source "${AIME_SOURCE_JSONL}" \
  --model "${STUDENT_HF_DIR}" --cache-dir "${HF_HOME}" --data-root "${DATA_ROOT}"
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/build_variants.py" \
  --contract "${CONTRACT_FILE}" --unmasked-data "${WRONG_UNMASKED_DATA}" \
  --data-root "${DATA_ROOT}" --stats-output "${VARIANT_STATS_JSON}"
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/validate_data.py" \
  --contract "${CONTRACT_FILE}" --data-root "${DATA_ROOT}" \
  --schedule-output "${SCHEDULE_AUDIT_JSON}"
echo "AIME_GENERALIZATION_DATA_JOB_COMPLETE at=$(date -Is)"

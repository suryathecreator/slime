#!/bin/bash
#SBATCH --job-name=q8b-masksft-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_40k"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

[[ -f "${STUDENT_HF_DIR}/config.json" ]] || { echo "Missing base model ${STUDENT_HF_DIR}" >&2; exit 1; }
"${SCRIPT_DIR}/container_exec.sh" python3 -m pytest -q \
  "${SCRIPT_DIR}/tests" \
  "${SLIME_REPO_ROOT}/tests/test_cp_utils.py"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/prepare_data.py" \
  --dataset "${OPENR1_DATASET}" \
  --config "${OPENR1_CONFIG}" \
  --split "${OPENR1_SPLIT}" \
  --revision "${OPENR1_REVISION}" \
  --cache-dir "${HF_HOME}" \
  --model "${STUDENT_HF_DIR}" \
  --seed "${DATA_SEED}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
  --correct-only-rows "${CORRECT_ONLY_ROWS}" \
  --mixed-correct-rows "${MIXED_CORRECT_ROWS}" \
  --mixed-wrong-rows "${MIXED_WRONG_ROWS}" \
  --selected-traces "${SELECTED_TRACES_JSONL}" \
  --correct-only-output "${CORRECT_ONLY_JSONL}" \
  --unmasked-output "${UNMASKED_JSONL}" \
  --stats-output "${PREP_STATS_JSON}"

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${CORRECT_ONLY_JSONL}" --variant correct_only
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${UNMASKED_JSONL}" --variant unmasked
echo "PREP_JOB_COMPLETE at=$(date -Is)"

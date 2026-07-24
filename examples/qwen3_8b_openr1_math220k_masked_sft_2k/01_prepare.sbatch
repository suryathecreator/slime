#!/bin/bash
#SBATCH --job-name=q8b-sft2k-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/prepare_from_axolotl.py" \
  --source-correct-only "${AXOLOTL_CORRECT_ONLY}" \
  --source-mixed-unmasked "${AXOLOTL_MIXED_UNMASKED}" \
  --tokenizer "${STUDENT_HF_DIR}" \
  --selected-output "${SELECTED_TRACES_JSONL}" \
  --correct-output "${CORRECT_ONLY_JSONL}" \
  --mixed-output "${UNMASKED_JSONL}" \
  --stats-output "${PREP_STATS_JSON}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"

correct_hash="$(jq -r '.trace_sets.correct_only_ordered_sha256' "${PREP_STATS_JSON}")"
mixed_hash="$(jq -r '.trace_sets.mixed_ordered_sha256' "${PREP_STATS_JSON}")"
expected_correct_hash="$(jq -r '.selection.correct_only_trace_order_sha256' "${CONTRACT_FILE}")"
expected_mixed_hash="$(jq -r '.selection.mixed_trace_order_sha256' "${CONTRACT_FILE}")"
[[ "${correct_hash}" == "${expected_correct_hash}" ]] || {
  echo "Correct-only trace fingerprint drift." >&2
  exit 1
}
[[ "${mixed_hash}" == "${expected_mixed_hash}" ]] || {
  echo "Mixed trace fingerprint drift." >&2
  exit 1
}
for key in \
  correct_only_duplicate_close correct_only_missing_close \
  mixed_duplicate_close mixed_missing_close; do
  actual="$(jq -r ".source_structure.${key}" "${PREP_STATS_JSON}")"
  expected="$(jq -r ".selection.source_structure.${key}" "${CONTRACT_FILE}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "Source structure count drift for ${key}: ${actual} != ${expected}" >&2
    exit 1
  }
done
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${CORRECT_ONLY_JSONL}" --variant correct_only \
  --expected-trace-hash "${correct_hash}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${UNMASKED_JSONL}" --variant unmasked \
  --expected-trace-hash "${mixed_hash}"
echo "PREP_JOB_COMPLETE at=$(date -Is)"

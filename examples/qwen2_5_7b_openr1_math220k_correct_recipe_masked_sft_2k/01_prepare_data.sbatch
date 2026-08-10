#!/bin/bash
#SBATCH --job-name=q25crm-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
VALIDATE_DATA="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k/validate_dataset.py"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Data preparation must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q25crm-data-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
for target in "${DATA_ROOT}" "${OUTPUT_ROOT}" "${HANDOFF_ROOT}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

python3 "${SCRIPT_DIR}/validate_config.py" --example-dir "${SCRIPT_DIR}" \
  --require-source-artifacts
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/prepare_data.py" \
  --contract "${CONTRACT_FILE}" --source-selected "${SOURCE_SELECTED}" \
  --source-pretokenized "${SOURCE_PRETOKENIZED}" \
  --selected-output "${SELECTED_TRACES_JSONL}" \
  --unmasked-output "${UNMASKED_JSONL}" --stats-output "${PREP_STATS_JSON}" \
  --inventory-output "${TOKENIZER_INVENTORY_JSON}" \
  --model "${STUDENT_HF_DIR}" --cache-dir "${HF_HOME}" \
  --seed "${DATA_SEED}" --correct-rows "${MIXED_CORRECT_ROWS}" \
  --wrong-rows "${MIXED_WRONG_ROWS}"
mixed_hash="$(jq -r '.ordered_mixed_trace_ids_sha256' "${PREP_STATS_JSON}")"
"${CONTAINER_EXEC}" python3 "${VALIDATE_DATA}" \
  --data "${UNMASKED_JSONL}" --variant unmasked \
  --inventory "${TOKENIZER_INVENTORY_JSON}" --expected-trace-hash "${mixed_hash}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"
"${CONTAINER_EXEC}" python3 \
  "${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/validate_schedule.py" \
  --data "${UNMASKED_JSONL}" --output "${SCHEDULE_AUDIT_JSON}" \
  --dp-size 2 --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}" \
  --seed "${DATA_SEED}"
echo "CORRECT_RECIPE_DATA_JOB_COMPLETE at=$(date -Is)"

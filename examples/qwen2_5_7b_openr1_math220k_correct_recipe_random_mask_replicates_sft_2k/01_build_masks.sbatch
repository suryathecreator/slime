#!/bin/bash
#SBATCH --job-name=q25rr-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Mask construction must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q25rr-build-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
for target in "${DATA_ROOT}" "${OUTPUT_ROOT}" "${HANDOFF_ROOT}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

python3 "${SCRIPT_DIR}/validate_config.py" --example-dir "${SCRIPT_DIR}" \
  --require-source-artifacts
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/build_random_masks.py" \
  --contract "${CONTRACT_FILE}" --source-unmasked "${SOURCE_UNMASKED_JSONL}" \
  --source-prep-stats "${SOURCE_PREP_STATS}" \
  --source-tokenizer-inventory "${SOURCE_TOKENIZER_INVENTORY}" \
  --source-schedule-audit "${SOURCE_SCHEDULE_AUDIT}" \
  --source-training-status "${SOURCE_TRAINING_STATUS}" \
  --source-comparison-sources "${SOURCE_COMPARISON_SOURCES}" \
  --source-artifacts-output "${SOURCE_ARTIFACTS_JSON}" \
  --output-dir "${DATA_ROOT}" --stats-output "${VARIANT_STATS_JSON}"

cp --preserve=mode,timestamps "${SOURCE_PREP_STATS}" "${PREP_STATS_JSON}"
cp --preserve=mode,timestamps "${SOURCE_TOKENIZER_INVENTORY}" "${TOKENIZER_INVENTORY_JSON}"
cp --preserve=mode,timestamps "${SOURCE_SCHEDULE_AUDIT}" "${SCHEDULE_AUDIT_JSON}"

mixed_hash="$(jq -r '.ordered_mixed_trace_ids_sha256' "${PREP_STATS_JSON}")"
for variant in "${VARIANTS[@]}"; do
  data="$(variant_data_path "${variant}")"
  "${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/validate_random_mask.py" \
    --contract "${CONTRACT_FILE}" --source-unmasked "${SOURCE_UNMASKED_JSONL}" \
    --data "${data}" --variant "${variant}" --expected-trace-hash "${mixed_hash}" \
    --max-sequence-length "${SFT_SEQ_LENGTH}"
done
echo "REPLICATE_MASK_BUILD_COMPLETE at=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=qco-audit-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${RUN_KEY:?Set RUN_KEY}"
: "${CANARY_AUDIT_ATTEMPT:?Set CANARY_AUDIT_ATTEMPT}"
: "${SLURM_JOB_ID:?Canary audit resume must run under Slurm}"
if [[ ! "${CANARY_AUDIT_ATTEMPT}" =~ ^[a-z0-9_]+$ ]]; then
  echo "CANARY_AUDIT_ATTEMPT must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi
resolve_run "${RUN_KEY}"

export TMPDIR="/tmp/${USER:-suryadv}/qco-audit-resume-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
readonly CANARY_INPUT_ROOT="${OUTPUT_ROOT}/canaries/${RUN_KEY}"
readonly AUDIT_ROOT="${CANARY_INPUT_ROOT}/attempts/${CANARY_AUDIT_ATTEMPT}"
readonly CUSTOM_SOURCE="${CANARY_INPUT_ROOT}/source/custom_pretokenized.jsonl"
readonly STOCK_SOURCE="${CANARY_INPUT_ROOT}/source/stock_messages.jsonl"
readonly CUSTOM_DETAILS="${CANARY_INPUT_ROOT}/custom_pretokenized/details"
readonly STOCK_DETAILS="${CANARY_INPUT_ROOT}/stock_messages/details"
readonly CUSTOM_FINAL="${CANARY_INPUT_ROOT}/custom_pretokenized/weights/iter_0000000"
readonly STOCK_FINAL="${CANARY_INPUT_ROOT}/stock_messages/weights/iter_0000000"

for path in \
  "${CUSTOM_SOURCE}" \
  "${STOCK_SOURCE}" \
  "${CUSTOM_FINAL}/config.json" \
  "${CUSTOM_FINAL}/model.safetensors.index.json" \
  "${STOCK_FINAL}/config.json" \
  "${STOCK_FINAL}/model.safetensors.index.json"; do
  [[ -f "${path}" ]] || { echo "Missing completed canary artifact ${path}" >&2; exit 1; }
done
for path in \
  "${CANARY_INPUT_ROOT}/runtime_batch_audit.json" \
  "${CANARY_INPUT_ROOT}/decoded_shifted_tokens.jsonl" \
  "${CANARY_INPUT_ROOT}/generation_before_after.json"; do
  [[ ! -e "${path}" ]] || { echo "Original canary already has ${path}" >&2; exit 1; }
done
[[ ! -e "${AUDIT_ROOT}" ]] || {
  echo "Refusing to overwrite canary audit attempt ${AUDIT_ROOT}" >&2
  exit 1
}
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${AUDIT_ROOT}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

dp_size=$((SFT_ACTOR_GPUS / SFT_TENSOR_MODEL_PARALLEL_SIZE))
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/audit_runtime_batches.py" \
  --tokenizer "${STUDENT_HF_DIR}" --loss-mask-type "${SFT_LOSS_MASK_TYPE}" \
  --template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}" --dp-size "${dp_size}" \
  --tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}" \
  --custom-source "${CUSTOM_SOURCE}" --stock-source "${STOCK_SOURCE}" \
  --custom-details "${CUSTOM_DETAILS}" --stock-details "${STOCK_DETAILS}" \
  --output "${AUDIT_ROOT}/runtime_batch_audit.json" \
  --token-dump "${AUDIT_ROOT}/decoded_shifted_tokens.jsonl"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/canary_generate.py" \
  --base "${STUDENT_HF_DIR}" --custom "${CUSTOM_FINAL}" --stock "${STOCK_FINAL}" \
  --prompts "${STOCK_SOURCE}" \
  --template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}" \
  --output "${AUDIT_ROOT}/generation_before_after.json" --count 8
echo "CANARY_AUDIT_RESUME_COMPLETE run=${RUN_KEY} attempt=${CANARY_AUDIT_ATTEMPT} at=$(date -Is)"

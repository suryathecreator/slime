#!/bin/bash
#SBATCH --job-name=qco-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
source "${SCRIPT_DIR}/env.sh"
: "${RUN_KEY:?Set RUN_KEY}"
: "${SLURM_JOB_ID:?Canary must run under Slurm}"
CANARY_ATTEMPT="${CANARY_ATTEMPT:-}"
if [[ -n "${CANARY_ATTEMPT}" && ! "${CANARY_ATTEMPT}" =~ ^[a-z0-9_]+$ ]]; then
  echo "CANARY_ATTEMPT must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi
resolve_run "${RUN_KEY}"
export TMPDIR="/tmp/${USER:-suryadv}/qco-canary-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export SFT_SKIP_LOG_REDIRECT=1
export SFT_SIZE=64
export SFT_NUM_EPOCH=1
export SFT_ROLLOUT_BATCH_SIZE=64
export SFT_GLOBAL_BATCH_SIZE=64
export SFT_NUM_ROLLOUT=1
export SFT_FINAL_ROLLOUT_ID=0
export SFT_MILESTONE_ROLLOUT_IDS=0
export SFT_SAVE_INTERVAL=1
export SFT_LR_DECAY_ITERS=1
if [[ -n "${CANARY_ATTEMPT}" ]]; then
  export CANARY_ROOT="${OUTPUT_ROOT}/canaries/${RUN_KEY}/attempts/${CANARY_ATTEMPT}"
else
  export CANARY_ROOT="${OUTPUT_ROOT}/canaries/${RUN_KEY}"
fi
export CANARY_CUSTOM_JSONL="${CANARY_ROOT}/source/custom_pretokenized.jsonl"
export CANARY_MESSAGES_JSONL="${CANARY_ROOT}/source/stock_messages.jsonl"
export CANARY_AUDIT_JSON="${CANARY_ROOT}/runtime_batch_audit.json"
export CANARY_TOKEN_DUMP_JSONL="${CANARY_ROOT}/decoded_shifted_tokens.jsonl"
export CANARY_GENERATION_JSON="${CANARY_ROOT}/generation_before_after.json"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${CANARY_ROOT}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/prepare_canaries.py" \
  --custom-source "${SFT_PARQUET}" --message-source "${STOCK_MESSAGES_JSONL}" \
  --custom-output "${CANARY_CUSTOM_JSONL}" --message-output "${CANARY_MESSAGES_JSONL}" \
  --rows 64

run_canary() {
  local name="$1"
  local data="$2"
  local input_key="$3"
  local rollout_function="$4"
  export SFT_PARQUET="${data}"
  export SFT_INPUT_KEY="${input_key}"
  export SFT_ROLLOUT_FUNCTION_PATH="${rollout_function}"
  export SFT_SAVE_DIR="${CANARY_ROOT}/${name}/full_state"
  export SFT_HF_SNAPSHOT_DIR="${CANARY_ROOT}/${name}/weights"
  export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
  export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000000"
  export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000000"
  export SFT_DETAILS_DIR="${CANARY_ROOT}/${name}/details"
  export CHECKPOINT_REPORT_DIR="${CANARY_ROOT}/${name}/checkpoint_reports"
  export SFT_WANDB_GROUP="${RUN_KEY}-canary-${name}"
  for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}"; do
    [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  done
  bash "${SHARED_RUNNER}"
  test -f "${SFT_FINAL_HF_DIR}/config.json"
}

run_canary custom_pretokenized "${CANARY_CUSTOM_JSONL}" input_ids \
  examples.qwen_correct_only_openr1_math220k_sft_2k.correct_only_sft_rollout.generate_rollout
CUSTOM_FINAL="${SFT_FINAL_HF_DIR}"
run_canary stock_messages "${CANARY_MESSAGES_JSONL}" messages \
  slime.rollout.sft_rollout.generate_rollout
STOCK_FINAL="${SFT_FINAL_HF_DIR}"

dp_size=$((SFT_ACTOR_GPUS / SFT_TENSOR_MODEL_PARALLEL_SIZE))
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/audit_runtime_batches.py" \
  --tokenizer "${STUDENT_HF_DIR}" --loss-mask-type "${SFT_LOSS_MASK_TYPE}" \
  --template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}" --dp-size "${dp_size}" \
  --tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}" \
  --custom-source "${CANARY_CUSTOM_JSONL}" --stock-source "${CANARY_MESSAGES_JSONL}" \
  --custom-details "${CANARY_ROOT}/custom_pretokenized/details" \
  --stock-details "${CANARY_ROOT}/stock_messages/details" \
  --output "${CANARY_AUDIT_JSON}" --token-dump "${CANARY_TOKEN_DUMP_JSONL}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/canary_generate.py" \
  --base "${STUDENT_HF_DIR}" --custom "${CUSTOM_FINAL}" --stock "${STOCK_FINAL}" \
  --prompts "${CANARY_MESSAGES_JSONL}" \
  --template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}" \
  --output "${CANARY_GENERATION_JSON}" --count 8
echo "CANARY_COMPLETE run=${RUN_KEY} at=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=q3-06b-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_0_6b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Canary training must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q3-06b-canary-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export SFT_SKIP_LOG_REDIRECT=1
export SFT_SIZE=32
export SFT_NUM_ROLLOUT=2
export SFT_FINAL_ROLLOUT_ID=1
export SFT_MILESTONE_ROLLOUT_IDS=1
export SFT_SAVE_INTERVAL=2
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${CANARY_ROOT}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
"${SCRIPT_DIR}/container_exec.sh" python3 \
  "${SHARED_QWEN_SFT_DIR}/check_checkpoint_tmpdir.py"

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
  export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000001"
  export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000001"
  export SFT_DETAILS_DIR="${CANARY_ROOT}/${name}/details"
  export CHECKPOINT_REPORT_DIR="${CANARY_ROOT}/${name}/checkpoint_reports"
  export SFT_WANDB_GROUP="qwen3-0.6b-canary-${name}"
  for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}"; do
    [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
  done
  bash "${SHARED_RUNNER}"
  test -f "${SFT_FINAL_HF_DIR}/config.json"
}

run_canary custom_pretokenized "${CANARY_CUSTOM_JSONL}" input_ids \
  examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout.generate_rollout
CUSTOM_FINAL="${SFT_FINAL_HF_DIR}"
run_canary stock_messages "${CANARY_MESSAGES_JSONL}" messages \
  slime.rollout.sft_rollout.generate_rollout
STOCK_FINAL="${SFT_FINAL_HF_DIR}"

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/audit_runtime_batches.py" \
  --tokenizer "${STUDENT_HF_DIR}" \
  --custom-source "${CANARY_CUSTOM_JSONL}" --stock-source "${CANARY_MESSAGES_JSONL}" \
  --custom-details "${CANARY_ROOT}/custom_pretokenized/details" \
  --stock-details "${CANARY_ROOT}/stock_messages/details" \
  --output "${CANARY_AUDIT_JSON}" --token-dump "${CANARY_TOKEN_DUMP_JSONL}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/canary_generate.py" \
  --base "${STUDENT_HF_DIR}" --custom "${CUSTOM_FINAL}" --stock "${STOCK_FINAL}" \
  --prompts "${CANARY_MESSAGES_JSONL}" --output "${CANARY_GENERATION_JSON}" --count 8
echo "QWEN3_0P6B_CANARIES_COMPLETE at=$(date -Is)"

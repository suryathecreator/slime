#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_COMMON=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
SBATCH_REPORT_COMMON=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
if [[ -n "${GPU_GRES:-}" ]]; then
  SBATCH_COMMON+=(--gres "${GPU_GRES}")
fi

printf -v SFT_FINAL_STAGE "sft_%06d" "$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))"
SFT_FINAL_SUMMARY="${SFT_EVAL_OUTPUT_DIR}/${SFT_FINAL_STAGE}/summary.json"

for required_path in \
  "${OPD_JSONL}" \
  "${SPLIT_METADATA}" \
  "${SFT_FINAL_HF_DIR}" \
  "${SFT_FINAL_SUMMARY}" \
  "${TEACHER_HF_DIR}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path before submit: ${required_path}" >&2
    exit 1
  fi
done

if [[ "${OPD_EFFECTIVE_TRAIN_SAMPLES}" -ne 1024 ]]; then
  echo "Expected 1024 effective OPD samples, got ${OPD_EFFECTIVE_TRAIN_SAMPLES}" >&2
  exit 1
fi

submit_log="${SLURM_LOG_DIR}/submit_opd_${OPD_RUN_LABEL}_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting OPD ${OPD_RUN_LABEL} chain"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "submit log: ${submit_log}"
echo "OPD pool/train/effective: ${OPD_POOL_SIZE}/${OPD_TRAIN_SIZE}/${OPD_EFFECTIVE_TRAIN_SAMPLES}"
echo "OPD rollout ids: 0-${OPD_FINAL_ROLLOUT_ID}"
echo "OPD actor/rollout/teacher GPUs: ${OPD_ACTOR_GPUS}/${OPD_ROLLOUT_GPUS}/${OPD_TEACHER_GPU}"
echo "OPD TP/CP/max response/seq length/max tokens per GPU: ${OPD_TENSOR_MODEL_PARALLEL_SIZE}/${OPD_CONTEXT_PARALLEL_SIZE}/${OPD_MAX_RESPONSE_LEN}/${OPD_SEQ_LENGTH}/${OPD_MAX_TOKENS_PER_GPU}"

jid_opd="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --time=18:00:00 \
    --job-name=slime-qwen3-opd1k32k \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd} \
    --time=03:00:00 \
    --job-name=slime-qwen3-opd1k-math500 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1,OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_base_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd_eval} \
    --time=03:00:00 \
    --job-name=slime-qwen3-base-math500 \
    --export=ALL,EVAL_TARGETS=base,EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_report="$(
  sbatch --parsable "${SBATCH_REPORT_COMMON[@]}" \
    --dependency=afterok:${jid_base_eval} \
    --time=00:30:00 \
    --job-name=slime-qwen3-final-report \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "opd=${jid_opd}"
  echo "opd_eval_final=${jid_opd_eval}"
  echo "base_eval=${jid_base_eval}"
  echo "report=${jid_report}"
  echo "SFT_FINAL_HF_DIR=${SFT_FINAL_HF_DIR}"
  echo "SFT_FINAL_SUMMARY=${SFT_FINAL_SUMMARY}"
  echo "OPD_JSONL=${OPD_JSONL}"
  echo "SPLIT_METADATA=${SPLIT_METADATA}"
  echo "OPD_SAVE_DIR=${OPD_SAVE_DIR}"
  echo "OPD_HF_SNAPSHOT_DIR=${OPD_HF_SNAPSHOT_DIR}"
  echo "OPD_FINAL_HF_DIR=${OPD_FINAL_HF_DIR}"
  echo "OPD_TRAINED_MANIFEST=${OPD_TRAINED_MANIFEST}"
  echo "OPD_EVAL_OUTPUT_DIR=${OPD_EVAL_OUTPUT_DIR}"
  echo "BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR}"
  echo "COMBINED_EVAL_OUTPUT_DIR=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "CHECKPOINT_REPORT_DIR=${CHECKPOINT_REPORT_DIR}"
} | tee "${submit_log}"

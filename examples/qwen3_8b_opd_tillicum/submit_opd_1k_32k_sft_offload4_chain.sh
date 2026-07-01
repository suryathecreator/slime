#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export OPD_RUN_LABEL="${OPD_RUN_LABEL:-1k_32k_sft_offload4}"
export OPD_INITIAL_LOAD_DIR="${OPD_INITIAL_LOAD_DIR:-${SFT_SAVE_DIR:-}}"
export OPD_ACTOR_GPUS="${OPD_ACTOR_GPUS:-2}"
export OPD_ROLLOUT_GPUS="${OPD_ROLLOUT_GPUS:-1}"
export OPD_RAY_GPUS="${OPD_RAY_GPUS:-3}"
export OPD_TEACHER_GPU="${OPD_TEACHER_GPU:-3}"
export OPD_TENSOR_MODEL_PARALLEL_SIZE="${OPD_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export OPD_CONTEXT_PARALLEL_SIZE="${OPD_CONTEXT_PARALLEL_SIZE:-2}"
export OPD_SEQ_LENGTH="${OPD_SEQ_LENGTH:-32768}"
export OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-31744}"
export OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-16384}"
export OPD_OPTIMIZER_CPU_OFFLOAD="${OPD_OPTIMIZER_CPU_OFFLOAD:-1}"
export GPU_GRES="${GPU_GRES:-gpu:h200:4}"
export EVAL_GPU_GRES="${EVAL_GPU_GRES:-gpu:h200:4}"
export EVAL_ROLLOUT_NUM_GPUS="${EVAL_ROLLOUT_NUM_GPUS:-4}"
export EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-128}"
export EVAL_SGLANG_SERVER_CONCURRENCY="${EVAL_SGLANG_SERVER_CONCURRENCY:-4}"
export CORRECTED_AFTERANY_DEPENDENCY="${CORRECTED_AFTERANY_DEPENDENCY:-156277:156278:156279}"
export REPORT_EXPERIMENT_NOTE="${REPORT_EXPERIMENT_NOTE:-Corrected SFT -> OPD run: OPD loaded the full SFT Megatron optimizer checkpoint at rollout 96. The earlier 1k_32k run is an accidental base -> OPD test because it used an HF snapshot as Megatron --load and fell back to base.}"

source "${SCRIPT_DIR}/env.sh"
export BASE_EVAL_REUSE_DIR="${BASE_EVAL_REUSE_DIR:-${OUTPUT_ROOT}/math500_eval_base_25k_opd_1k_32k}"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_TRAIN=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${GPU_GRES}" --cpus-per-task=32)
SBATCH_EVAL=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${EVAL_GPU_GRES}" --cpus-per-task=32)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")

printf -v SFT_FINAL_STAGE "sft_%06d" "$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))"
SFT_FINAL_SUMMARY="${SFT_EVAL_OUTPUT_DIR}/${SFT_FINAL_STAGE}/summary.json"

for required_path in \
  "${OPD_JSONL}" \
  "${SPLIT_METADATA}" \
  "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" \
  "${SFT_FINAL_FULL_CKPT_DIR}/.metadata" \
  "${SFT_FINAL_FULL_CKPT_DIR}/common.pt" \
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
if [[ "${OPD_CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
  divisor=$((2 * OPD_CONTEXT_PARALLEL_SIZE))
  if (( OPD_SEQ_LENGTH % divisor != 0 )); then
    echo "OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH} must be divisible by ${divisor} for CP=${OPD_CONTEXT_PARALLEL_SIZE}" >&2
    exit 1
  fi
fi

submit_log="${SLURM_LOG_DIR}/submit_opd_${OPD_RUN_LABEL}_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting corrected SFT-loaded OPD ${OPD_RUN_LABEL} chain"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "submit log: ${submit_log}"
echo "wait dependency: afterany:${CORRECTED_AFTERANY_DEPENDENCY}"
echo "train/eval gres: ${GPU_GRES}/${EVAL_GPU_GRES}"
echo "OPD initial load: ${SFT_SAVE_DIR}"
echo "OPD actor/rollout/teacher/ray GPUs: ${OPD_ACTOR_GPUS}/${OPD_ROLLOUT_GPUS}/${OPD_TEACHER_GPU}/${OPD_RAY_GPUS}"
echo "OPD TP/CP/max response/seq length/max tokens per GPU: ${OPD_TENSOR_MODEL_PARALLEL_SIZE}/${OPD_CONTEXT_PARALLEL_SIZE}/${OPD_MAX_RESPONSE_LEN}/${OPD_SEQ_LENGTH}/${OPD_MAX_TOKENS_PER_GPU}"
echo "Eval GPUs/batch/concurrency: ${EVAL_ROLLOUT_NUM_GPUS}/${EVAL_ROLLOUT_BATCH_SIZE}/${EVAL_SGLANG_SERVER_CONCURRENCY}"

jid_opd="$(
  sbatch --parsable "${SBATCH_TRAIN[@]}" \
    --dependency=afterany:${CORRECTED_AFTERANY_DEPENDENCY} \
    --time=18:00:00 \
    --job-name=slime-qwen3-opd1k-sft4g \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd_eval="$(
  sbatch --parsable "${SBATCH_EVAL[@]}" \
    --dependency=afterok:${jid_opd} \
    --time=05:00:00 \
    --job-name=slime-qwen3-opd1k-sft4g-eval \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1,OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_base_maybe="$(
  sbatch --parsable "${SBATCH_EVAL[@]}" \
    --dependency=afterok:${jid_opd_eval} \
    --time=05:00:00 \
    --job-name=slime-qwen3-base-math500-maybe \
    --export=ALL,EVAL_TARGETS=base,EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/08_maybe_base_eval_math500.sbatch
)"
jid_report="$(
  sbatch --parsable "${SBATCH_REPORT[@]}" \
    --dependency=afterok:${jid_base_maybe} \
    --time=00:30:00 \
    --job-name=slime-qwen3-final-report-sft4g \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "opd=${jid_opd}"
  echo "opd_eval_final=${jid_opd_eval}"
  echo "base_maybe=${jid_base_maybe}"
  echo "report=${jid_report}"
  echo "dependency=afterany:${CORRECTED_AFTERANY_DEPENDENCY}"
  echo "SFT_SAVE_DIR=${SFT_SAVE_DIR}"
  echo "SFT_FINAL_FULL_CKPT_DIR=${SFT_FINAL_FULL_CKPT_DIR}"
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
  echo "BASE_EVAL_REUSE_DIR=${BASE_EVAL_REUSE_DIR}"
  echo "COMBINED_EVAL_OUTPUT_DIR=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "CHECKPOINT_REPORT_DIR=${CHECKPOINT_REPORT_DIR}"
  echo "EVAL_ROLLOUT_NUM_GPUS=${EVAL_ROLLOUT_NUM_GPUS}"
  echo "EVAL_ROLLOUT_BATCH_SIZE=${EVAL_ROLLOUT_BATCH_SIZE}"
  echo "EVAL_SGLANG_SERVER_CONCURRENCY=${EVAL_SGLANG_SERVER_CONCURRENCY}"
} | tee "${submit_log}"

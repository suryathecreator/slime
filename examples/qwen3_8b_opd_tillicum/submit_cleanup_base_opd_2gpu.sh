#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export OPD_RUN_LABEL="${OPD_RUN_LABEL:-1k_32k}"
export EVAL_ROLLOUT_NUM_GPUS="${EVAL_ROLLOUT_NUM_GPUS:-2}"
export EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-64}"
export EVAL_SGLANG_SERVER_CONCURRENCY="${EVAL_SGLANG_SERVER_CONCURRENCY:-6}"
export EVAL_TENSOR_MODEL_PARALLEL_SIZE="${EVAL_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export EVAL_CONTEXT_PARALLEL_SIZE="${EVAL_CONTEXT_PARALLEL_SIZE:-1}"
export EVAL_MAX_TOKENS_PER_GPU="${EVAL_MAX_TOKENS_PER_GPU:-31744}"
export REPORT_INCLUDE_SFT="${REPORT_INCLUDE_SFT:-0}"
export REPORT_OPD_X_OFFSET_SAMPLES="${REPORT_OPD_X_OFFSET_SAMPLES:-0}"
export REPORT_OPD_LABEL_PREFIX="${REPORT_OPD_LABEL_PREFIX:-}"
export REPORT_EXPERIMENT_NOTE="${REPORT_EXPERIMENT_NOTE:-Accidental base -> OPD cleanup report. This run evaluates the old OPD checkpoint trained from base, not the corrected SFT -> OPD experiment.}"

source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

for required_path in \
  "${OPD_FINAL_HF_DIR}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path before cleanup submit: ${required_path}" >&2
    exit 1
  fi
done

submit_time="$(date --iso-8601=seconds)"
submit_log="${SLURM_LOG_DIR}/submit_cleanup_base_opd_${OPD_RUN_LABEL}_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting accidental base->OPD 2-GPU cleanup"
echo "submit time: ${submit_time}"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "run label: ${OPD_RUN_LABEL}"
echo "dependency: none"
echo "gres/cpus/time: gpu:h200:2/16/18:00:00"
echo "eval GPUs/batch/concurrency: ${EVAL_ROLLOUT_NUM_GPUS}/${EVAL_ROLLOUT_BATCH_SIZE}/${EVAL_SGLANG_SERVER_CONCURRENCY}"
echo "eval TP/CP/max tokens per GPU: ${EVAL_TENSOR_MODEL_PARALLEL_SIZE}/${EVAL_CONTEXT_PARALLEL_SIZE}/${EVAL_MAX_TOKENS_PER_GPU}"
echo "OPD final HF dir: ${OPD_FINAL_HF_DIR}"
echo "OPD eval dir: ${OPD_EVAL_OUTPUT_DIR}"
echo "Base eval dir: ${BASE_EVAL_OUTPUT_DIR}"
echo "Combined report dir: ${COMBINED_EVAL_OUTPUT_DIR}"
echo "submit log: ${submit_log}"

jid_cleanup="$(
  sbatch --parsable \
    -A "${ACCOUNT}" \
    -p "${PARTITION}" \
    --qos "${QOS}" \
    --gres gpu:h200:2 \
    --cpus-per-task=16 \
    --time=18:00:00 \
    --job-name=slime-qwen3-base-opd-cleanup \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/09_cleanup_base_opd_2gpu.sbatch
)"

{
  echo "cleanup=${jid_cleanup}"
  echo "submit_time=${submit_time}"
  echo "dependency=none"
  echo "run_label=${OPD_RUN_LABEL}"
  echo "slurm_request=gpu:h200:2 cpus-per-task=16 time=18:00:00"
  echo "EVAL_ROLLOUT_NUM_GPUS=${EVAL_ROLLOUT_NUM_GPUS}"
  echo "EVAL_ROLLOUT_BATCH_SIZE=${EVAL_ROLLOUT_BATCH_SIZE}"
  echo "EVAL_SGLANG_SERVER_CONCURRENCY=${EVAL_SGLANG_SERVER_CONCURRENCY}"
  echo "EVAL_TENSOR_MODEL_PARALLEL_SIZE=${EVAL_TENSOR_MODEL_PARALLEL_SIZE}"
  echo "EVAL_CONTEXT_PARALLEL_SIZE=${EVAL_CONTEXT_PARALLEL_SIZE}"
  echo "OPD_FINAL_HF_DIR=${OPD_FINAL_HF_DIR}"
  echo "OPD_EVAL_OUTPUT_DIR=${OPD_EVAL_OUTPUT_DIR}"
  echo "BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR}"
  echo "COMBINED_EVAL_OUTPUT_DIR=${COMBINED_EVAL_OUTPUT_DIR}"
} | tee "${submit_log}"

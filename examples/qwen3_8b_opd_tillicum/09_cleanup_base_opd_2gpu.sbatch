#!/bin/bash
#SBATCH --job-name=slime-qwen3-base-opd-cleanup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=0
#SBATCH --time=18:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi

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

mkdir -p "${SLURM_LOG_DIR}" "${OPD_EVAL_OUTPUT_DIR}" "${BASE_EVAL_OUTPUT_DIR}" "${COMBINED_EVAL_OUTPUT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"

echo "Starting accidental base->OPD cleanup on $(hostname) at $(date)"
echo "Run label: ${OPD_RUN_LABEL}"
echo "OPD final HF dir: ${OPD_FINAL_HF_DIR}"
echo "OPD eval dir: ${OPD_EVAL_OUTPUT_DIR}"
echo "Base eval dir: ${BASE_EVAL_OUTPUT_DIR}"
echo "Combined report dir: ${COMBINED_EVAL_OUTPUT_DIR}"
echo "Eval GPUs/batch/concurrency: ${EVAL_ROLLOUT_NUM_GPUS}/${EVAL_ROLLOUT_BATCH_SIZE}/${EVAL_SGLANG_SERVER_CONCURRENCY}"
echo "Eval TP/CP/max tokens per GPU: ${EVAL_TENSOR_MODEL_PARALLEL_SIZE}/${EVAL_CONTEXT_PARALLEL_SIZE}/${EVAL_MAX_TOKENS_PER_GPU}"
echo "Report include SFT: ${REPORT_INCLUDE_SFT}"

for required_path in \
  "${OPD_FINAL_HF_DIR}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required cleanup path: ${required_path}" >&2
    exit 1
  fi
done

run_inner_eval() {
  local target="$1"
  local output_dir="$2"
  local expected_summary="$3"
  local ray_tmp="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID:-cleanup}-${target}"
  local status=0
  mkdir -p "${ray_tmp}"
  echo "Running inner eval target=${target} output=${output_dir} at $(date)"
  env -u SLURM_JOB_ID -u SLURM_JOB_NAME \
    RAY_TMPDIR="${ray_tmp}" \
    EVAL_TARGETS="${target}" \
    EVAL_OUTPUT_DIR="${output_dir}" \
    EVAL_SKIP_COMPLETED=1 \
    OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}" \
    bash "${SCRIPT_DIR}/06_eval_math500_greedy_1x.sbatch" || status=$?
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi

  if [[ -s "${expected_summary}" ]]; then
    echo "INNER_EVAL_SUMMARY_SALVAGE_OK target=${target} summary=${expected_summary} after status=${status}"
    return 0
  fi
  echo "INNER_EVAL_FAILED target=${target} status=${status} missing expected summary=${expected_summary}" >&2
  return "${status}"
}

if [[ -s "${OPD_EVAL_OUTPUT_DIR}/opd_001024/summary.json" ]]; then
  echo "OPD cleanup summary already exists: ${OPD_EVAL_OUTPUT_DIR}/opd_001024/summary.json"
else
  run_inner_eval opd "${OPD_EVAL_OUTPUT_DIR}" "${OPD_EVAL_OUTPUT_DIR}/opd_001024/summary.json"
fi

if [[ -s "${BASE_EVAL_OUTPUT_DIR}/base/summary.json" ]]; then
  echo "Base cleanup summary already exists: ${BASE_EVAL_OUTPUT_DIR}/base/summary.json"
else
  run_inner_eval base "${BASE_EVAL_OUTPUT_DIR}" "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"
fi

for required_summary in \
  "${OPD_EVAL_OUTPUT_DIR}/opd_001024/summary.json" \
  "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"; do
  if [[ ! -s "${required_summary}" ]]; then
    echo "Missing required summary before combined report: ${required_summary}" >&2
    exit 1
  fi
done

if [[ -s "${COMBINED_EVAL_OUTPUT_DIR}/summary_all.json" ]]; then
  echo "Existing accidental base->OPD combined report will be refreshed: ${COMBINED_EVAL_OUTPUT_DIR}/summary_all.json"
fi

echo "Writing accidental base->OPD combined report at $(date)"
env -u SLURM_JOB_ID -u SLURM_JOB_NAME \
  EVAL_TARGETS=report \
  EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
  bash "${SCRIPT_DIR}/07_report_math500.sbatch"

echo "Finished accidental base->OPD cleanup at $(date)"

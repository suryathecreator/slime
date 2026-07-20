#!/bin/bash
#SBATCH --job-name=q8b32b-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
: "${EVAL_STAGE:?Set EVAL_STAGE}"
: "${EVAL_MODEL_DIR:?Set EVAL_MODEL_DIR}"
EVAL_STAGE_DIR="${EVAL_OUTPUT_ROOT}/${EVAL_STAGE}"
mkdir -p "${SLURM_LOG_DIR}" "${EVAL_STAGE_DIR}/problems" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

requeue_eval() {
  trap - USR1
  echo "EVAL_REQUEUE stage=${EVAL_STAGE} job=${SLURM_JOB_ID} at=$(date -Is)"
  local pid
  for pid in $(jobs -pr); do kill -TERM "${pid}" 2>/dev/null || true; done
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_eval USR1

[[ -f "${EVAL_MODEL_DIR}/config.json" ]] || { echo "Missing eval model: ${EVAL_MODEL_DIR}" >&2; exit 1; }
VLLM_TARGET_LIBS="$(find "${VLLM_EVAL_SITE}" -path '*/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${VLLM_TARGET_LIBS}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${VLLM_EVAL_SITE}:${MATH_VERIFY_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
export SLIME_VLLM_PATCH_SITE=0
export SLIME_SGLANG_PATCH_SITE=0

cd "${SLIME_REPO_ROOT}"
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/evaluate_math500.py" shard \
      --model "${EVAL_MODEL_DIR}" \
      --stage "${EVAL_STAGE}" \
      --output-dir "${EVAL_STAGE_DIR}" \
      --shard-index "${shard}" \
      --num-shards 4 \
      --temperature "${EVAL_TEMPERATURE}" \
      --top-p "${EVAL_TOP_P}" \
      --top-k "${EVAL_TOP_K}" \
      --max-new-tokens "${EVAL_MAX_RESPONSE_LEN}" \
      --safety-margin "${EVAL_SAFETY_MARGIN}" \
      --seed "${EVAL_SEED}" \
      --stop-token-ids ${EVAL_STOP_TOKEN_IDS} \
      --resume
  ) &
done
wait

AGGREGATE_ARGS=(
  aggregate
  --stage "${EVAL_STAGE}"
  --output-dir "${EVAL_STAGE_DIR}"
  --expected "${EVAL_EXPECTED_SAMPLES}"
  --bootstrap-seed "${EVAL_SEED}"
)
if [[ -n "${EVAL_COMPARE_TO:-}" ]]; then
  AGGREGATE_ARGS+=(--compare-to "${EVAL_COMPARE_TO}")
fi
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/evaluate_math500.py" "${AGGREGATE_ARGS[@]}"

if [[ "${BUILD_FINAL_REPORT:-0}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/build_results.py"
fi
echo "EVAL_COMPLETE stage=${EVAL_STAGE} summary=${EVAL_STAGE_DIR}/summary.json"

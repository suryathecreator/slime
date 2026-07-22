#!/bin/bash
#SBATCH --job-name=q8b32b-eval-cap32k
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
export TMPDIR="/tmp/${USER:-suryadv}/q8b32b-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
unset LD_LIBRARY_PATH
: "${CAP_PROXY_STAGE:?Set CAP_PROXY_STAGE}"
: "${CAP_PROXY_SOURCE_DIR:?Set CAP_PROXY_SOURCE_DIR}"
: "${CAP_PROXY_MODEL_DIR:?Set CAP_PROXY_MODEL_DIR}"
CAP_PROXY_STAGE_DIR="${EVAL_OUTPUT_ROOT}/${CAP_PROXY_STAGE}"
mkdir -p "${SLURM_LOG_DIR}" "${CAP_PROXY_STAGE_DIR}/problems" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

requeue_eval() {
  trap - USR1
  echo "CAP_PROXY_REQUEUE stage=${CAP_PROXY_STAGE} job=${SLURM_JOB_ID} at=$(date -Is)"
  local pid
  for pid in $(jobs -pr); do kill -TERM "${pid}" 2>/dev/null || true; done
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_eval USR1

[[ -f "${CAP_PROXY_SOURCE_DIR}/summary.json" ]] || { echo "Missing source summary: ${CAP_PROXY_SOURCE_DIR}/summary.json" >&2; exit 1; }
[[ -f "${CAP_PROXY_MODEL_DIR}/config.json" ]] || { echo "Missing proxy model: ${CAP_PROXY_MODEL_DIR}" >&2; exit 1; }
VLLM_TARGET_LIBS="$(find "${VLLM_EVAL_SITE}" -path '*/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${VLLM_TARGET_LIBS}"
export PYTHONPATH="${VLLM_EVAL_SITE}:${MATH_VERIFY_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
export SLIME_VLLM_PATCH_SITE=0
export SLIME_SGLANG_PATCH_SITE=0
export CC="${EVAL_ZIG_CC}"
export CPATH="${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}${CPATH:+:${CPATH}}"

for path in \
  "${CC}" \
  "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h" \
  "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"; do
  [[ -e "${path}" ]] || { echo "Missing proxy compiler prerequisite: ${path}" >&2; exit 1; }
done
[[ -x "${CC}" ]] || { echo "Proxy compiler is not executable: ${CC}" >&2; exit 1; }
"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
test -x "${CC}"
test -f "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h"
test -f "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"
echo "CAP_PROXY_COMPILER_READY cc=${CC}"
"${CC}" --version
'

cd "${SLIME_REPO_ROOT}"
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${SCRIPT_DIR}/container_exec.sh" bash -lc '
      set -euo pipefail
      expected_visible_gpu="$1"
      shift
      if [[ "${CUDA_VISIBLE_DEVICES:-}" != "${expected_visible_gpu}" ]]; then
        echo "CUDA visibility mismatch: expected=${expected_visible_gpu} actual=${CUDA_VISIBLE_DEVICES:-unset}" >&2
        exit 1
      fi
      echo "CAP_PROXY_SHARD_GPU_READY shard=${expected_visible_gpu} visible=${CUDA_VISIBLE_DEVICES}"
      exec "$@"
    ' bash "${shard}" python3 "${SCRIPT_DIR}/evaluate_math500.py" cap-rerun-shard \
      --model "${CAP_PROXY_MODEL_DIR}" \
      --stage "${CAP_PROXY_STAGE}" \
      --source-dir "${CAP_PROXY_SOURCE_DIR}" \
      --output-dir "${CAP_PROXY_STAGE_DIR}" \
      --shard-index "${shard}" \
      --num-shards 4 \
      --expected "${EVAL_EXPECTED_SAMPLES}" \
      --target-context "${EVAL_CAP_PROXY_TARGET_CONTEXT}" \
      --safety-margin "${EVAL_SAFETY_MARGIN}" \
      --max-num-seqs "${EVAL_CAP_PROXY_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${EVAL_CAP_PROXY_MAX_NUM_BATCHED_TOKENS}" \
      --resume
  ) &
done
wait

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/evaluate_math500.py" aggregate-cap-proxy \
  --stage "${CAP_PROXY_STAGE}" \
  --source-dir "${CAP_PROXY_SOURCE_DIR}" \
  --output-dir "${CAP_PROXY_STAGE_DIR}" \
  --expected "${EVAL_EXPECTED_SAMPLES}" \
  --target-context "${EVAL_CAP_PROXY_TARGET_CONTEXT}" \
  --safety-margin "${EVAL_SAFETY_MARGIN}" \
  --bootstrap-seed "${EVAL_SEED}"

if [[ "${BUILD_FINAL_REPORT:-0}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/build_results.py"
fi
echo "CAP_PROXY_COMPLETE stage=${CAP_PROXY_STAGE} summary=${CAP_PROXY_STAGE_DIR}/summary.json"

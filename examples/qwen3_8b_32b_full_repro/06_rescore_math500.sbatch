#!/bin/bash
#SBATCH --job-name=q8b32b-rescore-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
export TMPDIR="/tmp/${USER:-suryadv}/q8b32b-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
unset LD_LIBRARY_PATH
mkdir -p "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

[[ "${RESCORE_MODE:-}" == "draft" || "${RESCORE_MODE:-}" == "final" ]] || {
  echo "RESCORE_MODE must be draft or final" >&2
  exit 1
}
[[ -n "${EXPECTED_LAUNCH_COMMIT:-}" ]] || {
  echo "EXPECTED_LAUNCH_COMMIT is required" >&2
  exit 1
}
actual_commit="$(git -C "${SLIME_REPO_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${EXPECTED_LAUNCH_COMMIT}" ]] || {
  echo "Checkout moved after submission: expected=${EXPECTED_LAUNCH_COMMIT} actual=${actual_commit}" >&2
  exit 1
}

SCORER_VERSION=math500_strict_boxed_scorer_v3
V2_ROOT="${EVAL_OUTPUT_ROOT}/rescored/math500_event_scorer_v2"
REPORT_ROOT="${OUTPUT_ROOT}/final_report"
short_commit="${EXPECTED_LAUNCH_COMMIT:0:12}"
if [[ "${RESCORE_MODE}" == "draft" ]]; then
  RESCORE_ROOT="${EVAL_OUTPUT_ROOT}/rescored/${SCORER_VERSION}_draft_${short_commit}"
  AUDIT_JSON="${REPORT_ROOT}/MATH500_SCORER_V3_DRAFT_${short_commit}.json"
  AUDIT_MD="${REPORT_ROOT}/MATH500_SCORER_V3_DRAFT_${short_commit}.md"
  expected_args=()
else
  RESCORE_ROOT="${EVAL_OUTPUT_ROOT}/rescored/${SCORER_VERSION}"
  AUDIT_JSON="${REPORT_ROOT}/MATH500_SCORER_V3_AUDIT.json"
  AUDIT_MD="${REPORT_ROOT}/MATH500_SCORER_V3_AUDIT.md"
  EXPECTED_METRICS="${SCRIPT_DIR}/MATH500_SCORER_V3_EXPECTED.json"
  [[ -f "${EXPECTED_METRICS}" ]] || {
    echo "Missing final expected metrics: ${EXPECTED_METRICS}" >&2
    exit 1
  }
  expected_args=(--expected-metrics "${EXPECTED_METRICS}")
fi

echo "RESCORE_START job=${SLURM_JOB_ID} mode=${RESCORE_MODE} source=${EVAL_OUTPUT_ROOT} output=${RESCORE_ROOT} at=$(date -Is)"
nvidia-smi -L
export PYTHONPATH="${MATH_VERIFY_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/rescore_math500.py" \
  --source-root "${EVAL_OUTPUT_ROOT}" \
  --v2-root "${V2_ROOT}" \
  --output-root "${RESCORE_ROOT}" \
  --audit-json "${AUDIT_JSON}" \
  --audit-md "${AUDIT_MD}" \
  --mode "${RESCORE_MODE}" \
  "${expected_args[@]}" \
  --expected 500 \
  --bootstrap-seed "${EVAL_SEED}"

if [[ "${RESCORE_MODE}" == "final" ]]; then
  PYTHONPATH="${SLIME_REPO_ROOT}" \
  SLIME_SGLANG_PATCH_SITE=0 \
  SLIME_VLLM_PATCH_SITE=0 \
  python3 "${SCRIPT_DIR}/build_results.py" \
    --eval-root "${RESCORE_ROOT}" \
    --v1-eval-root "${EVAL_OUTPUT_ROOT}" \
    --v2-eval-root "${V2_ROOT}" \
    --scorer-audit "${AUDIT_JSON}"
fi
echo "RESCORE_COMPLETE job=${SLURM_JOB_ID} mode=${RESCORE_MODE} audit=${AUDIT_JSON} at=$(date -Is)"

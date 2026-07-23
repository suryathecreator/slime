#!/bin/bash
#SBATCH --job-name=q8b32b-rescore-v2
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

SCORER_VERSION=math500_event_scorer_v2
RESCORE_ROOT="${EVAL_OUTPUT_ROOT}/rescored/${SCORER_VERSION}"
REPORT_ROOT="${OUTPUT_ROOT}/final_report"
AUDIT_JSON="${REPORT_ROOT}/MATH500_SCORER_AUDIT.json"
AUDIT_MD="${REPORT_ROOT}/MATH500_SCORER_AUDIT.md"

echo "RESCORE_START job=${SLURM_JOB_ID} source=${EVAL_OUTPUT_ROOT} output=${RESCORE_ROOT} at=$(date -Is)"
nvidia-smi -L
export PYTHONPATH="${MATH_VERIFY_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/rescore_math500.py" \
  --source-root "${EVAL_OUTPUT_ROOT}" \
  --output-root "${RESCORE_ROOT}" \
  --audit-json "${AUDIT_JSON}" \
  --audit-md "${AUDIT_MD}" \
  --expected 500 \
  --bootstrap-seed "${EVAL_SEED}"

python3 "${SCRIPT_DIR}/build_results.py" \
  --eval-root "${RESCORE_ROOT}" \
  --historical-eval-root "${EVAL_OUTPUT_ROOT}" \
  --scorer-audit "${AUDIT_JSON}"
echo "RESCORE_COMPLETE job=${SLURM_JOB_ID} audit=${AUDIT_JSON} at=$(date -Is)"

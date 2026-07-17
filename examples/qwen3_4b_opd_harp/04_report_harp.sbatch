#!/bin/bash
#SBATCH --job-name=q3-harp-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
: "${REPORT_MODE:?Set REPORT_MODE=base, opd1k, or final}"
mkdir -p "${SLURM_LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0
"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export PYTHONPATH="${HARP_EVAL_SITE}:${HARP_EXAMPLE_DIR}:${PYTHONPATH:-}"
if [[ "${REPORT_MODE}" == "base" ]]; then
  python3 examples/qwen3_4b_opd_harp/report_harp.py \
    --entry "Qwen3-4B base=${EVAL_4B_BASE_DIR}" \
    --entry "Qwen3-32B base=${EVAL_32B_BASE_DIR}" \
    --output-dir "${BASE_REPORT_DIR}" \
    --title "Qwen3 HARP V2 base comparison"
elif [[ "${REPORT_MODE}" == "opd1k" ]]; then
  python3 examples/qwen3_4b_opd_harp/report_harp.py \
    --entry "Qwen3-4B base=${EVAL_4B_BASE_DIR}" \
    --entry "Qwen3-32B base=${EVAL_32B_BASE_DIR}" \
    --entry "Qwen3-4B OPD-1K=${EVAL_OPD1K_DIR}" \
    --output-dir "${OPD1K_REPORT_DIR}" \
    --title "Qwen3-4B OPD-1K HARP V2 results"
elif [[ "${REPORT_MODE}" == "final" ]]; then
  python3 examples/qwen3_4b_opd_harp/report_harp.py \
    --entry "Qwen3-4B base=${EVAL_4B_BASE_DIR}" \
    --entry "Qwen3-32B base=${EVAL_32B_BASE_DIR}" \
    --entry "Qwen3-4B OPD-1K=${EVAL_OPD1K_DIR}" \
    --entry "Qwen3-4B OPD-5.1K=${EVAL_OPD5120_DIR}" \
    --output-dir "${FINAL_REPORT_DIR}" \
    --title "Qwen3-4B OPD HARP V2 results"
else
  echo "Unknown REPORT_MODE=${REPORT_MODE}" >&2
  exit 2
fi
'

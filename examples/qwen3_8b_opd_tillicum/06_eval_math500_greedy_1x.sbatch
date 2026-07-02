#!/bin/bash
#SBATCH --job-name=slime-qwen3-math500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${DATA_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting MATH-500 greedy eval on $(hostname) at $(date)"
echo "Eval targets: ${EVAL_TARGETS}"
echo "Eval output: ${EVAL_OUTPUT_DIR}"
echo "Eval skip completed: ${EVAL_SKIP_COMPLETED}"

"${SCRIPT_DIR}/container_exec.sh" bash "${SCRIPT_DIR}/06_eval_math500_greedy_inner.sh"

echo "Finished MATH-500 eval at $(date)"

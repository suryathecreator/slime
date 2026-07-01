#!/bin/bash
#SBATCH --job-name=slime-qwen3-base-math500-maybe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"

echo "Starting maybe-base MATH-500 eval on $(hostname) at $(date)"
echo "Base eval dir: ${BASE_EVAL_OUTPUT_DIR}"
echo "Base reuse dir: ${BASE_EVAL_REUSE_DIR:-}"

if [[ -s "${BASE_EVAL_OUTPUT_DIR}/base/summary.json" ]]; then
  echo "BASE_EVAL_REUSE_OK existing corrected base summary: ${BASE_EVAL_OUTPUT_DIR}/base/summary.json"
  exit 0
fi

if [[ -n "${BASE_EVAL_REUSE_DIR:-}" && -s "${BASE_EVAL_REUSE_DIR}/base/summary.json" ]]; then
  if [[ ! -e "${BASE_EVAL_OUTPUT_DIR}" ]]; then
    ln -s "${BASE_EVAL_REUSE_DIR}" "${BASE_EVAL_OUTPUT_DIR}"
    echo "BASE_EVAL_REUSE_OK linked ${BASE_EVAL_OUTPUT_DIR} -> ${BASE_EVAL_REUSE_DIR}"
    exit 0
  fi
  echo "Corrected base dir exists but lacks summary, so not linking over it: ${BASE_EVAL_OUTPUT_DIR}"
fi

echo "BASE_EVAL_REUSE_MISS running base eval into ${BASE_EVAL_OUTPUT_DIR}"
export EVAL_TARGETS=base
export EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}"
export EVAL_SKIP_COMPLETED=1
exec bash "${SCRIPT_DIR}/06_eval_math500_greedy_1x.sbatch"

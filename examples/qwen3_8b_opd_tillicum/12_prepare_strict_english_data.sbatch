#!/bin/bash
#SBATCH --job-name=slime-qwen3-strict-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${STRICT_EN_DATASET_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting strict-English OpenThoughts preparation at $(date)"
echo "Overall corpus: ${STRICT_EN_DATASET_DIR}"
echo "SFT selected output: ${SFT_PARQUET}"
echo "OPD reserve output: ${CLEANED_OPD_RESERVE_JSONL}"
echo "Count metadata: ${SPLIT_METADATA}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
export PYTHONPATH="${STRICT_LANGUAGE_SITE}:${SLIME_REPO_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}"
python3 examples/qwen3_8b_opd_tillicum/prepare_strict_english_openthoughts3.py
'

test -f "${STRICT_EN_DATASET_DIR}/_SUCCESS"
test -f "${SPLIT_METADATA}"
echo "Finished strict-English OpenThoughts preparation at $(date)"

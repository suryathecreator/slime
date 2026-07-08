#!/bin/bash
#SBATCH --job-name=slime-qwen3-clean-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting cleaned OpenThoughts3 prep on $(hostname) at $(date)"
echo "Dataset: ${OT3_DATASET} split=${OT3_SPLIT}"
echo "Seed: ${DATA_SEED}"
echo "SFT output: ${SFT_PARQUET}"
echo "OPD reserve: ${CLEANED_OPD_RESERVE_JSONL}"
echo "OPD 1k: ${CLEANED_OPD_1K_JSONL}"
echo "OPD next 4k: ${CLEANED_OPD_4K_JSONL}"
echo "Metadata: ${SPLIT_METADATA}"

"${SCRIPT_DIR}/container_exec.sh" python3 examples/qwen3_8b_opd_tillicum/02_prepare_cleaned_openthoughts3.py \
  --force

echo "Finished cleaned OpenThoughts3 prep at $(date)"

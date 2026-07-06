#!/bin/bash
#SBATCH --job-name=slime-qwen3-opd25k-cont-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
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
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Preparing OPD 25k continuation data and MATH-500 val100 at $(date)"
echo "Continuation OPD JSONL: ${OPD_JSONL}"
echo "Continuation metadata: ${OPD_CONTINUATION_METADATA}"
echo "Previous rollout logs: ${OPD_PREVIOUS_ROLLOUT_LOG_DIR}"
echo "Already-trained OPD samples: ${OPD_ALREADY_TRAINED_SAMPLES}"
echo "Target total effective OPD samples: ${OPD_EFFECTIVE_TRAIN_SAMPLES}"
echo "Val100 JSONL/config/metadata: ${MATH500_VAL100_JSONL} / ${MATH500_VAL100_CONFIG} / ${MATH500_VAL100_METADATA}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

python3 examples/qwen3_8b_opd_tillicum/prepare_math500_subset.py
python3 examples/qwen3_8b_opd_tillicum/prepare_opd_continuation_pool.py --force
'

echo "Finished OPD continuation data prep at $(date)"

#!/bin/bash
#SBATCH --job-name=q8b32b-sft200k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
bash examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch

IMPLEMENTATION_COMMIT="$(python3 -c 'import json,os; print(json.load(open(os.environ["CONTRACT_FILE"]))["repository_commit"])')"
export IMPLEMENTATION_COMMIT
python3 "${SCRIPT_DIR}/write_checkpoint_manifest.py" \
  --stage sft \
  --weights-root "${SFT_HF_SNAPSHOT_DIR}" \
  --state-root "${SFT_SAVE_DIR}" \
  --split-metadata "${SPLIT_METADATA}" \
  --out "${CHECKPOINT_REPORT_DIR}/sft_manifest.json"

#!/bin/bash
#SBATCH --job-name=q8b32b-opd1472
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
if [[ "$(realpath "${OPD_INITIAL_LOAD_DIR}")" != "$(realpath "${OPD_REF_LOAD_DIR}")" ]]; then
  echo "Initial actor/reference synchronization failed: ${OPD_INITIAL_LOAD_DIR} != ${OPD_REF_LOAD_DIR}" >&2
  exit 1
fi
bash examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch

PYTHONPATH="${MATH_VERIFY_SITE}:${PYTHONPATH:-}" \
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/audit_opd_rollouts.py"

IMPLEMENTATION_COMMIT="$(python3 -c 'import json,os; print(json.load(open(os.environ["CONTRACT_FILE"]))["repository_commit"])')"
export IMPLEMENTATION_COMMIT
python3 "${SCRIPT_DIR}/write_checkpoint_manifest.py" \
  --stage opd \
  --weights-root "${OPD_HF_SNAPSHOT_DIR}" \
  --state-root "${OPD_SAVE_DIR}" \
  --split-metadata "${SPLIT_METADATA}" \
  --out "${CHECKPOINT_REPORT_DIR}/opd_manifest.json"

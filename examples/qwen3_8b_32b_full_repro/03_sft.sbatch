#!/bin/bash
#SBATCH --job-name=q8b32b-sft200k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
export TMPDIR="/tmp/${USER:-suryadv}/q8b32b-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
unset LD_LIBRARY_PATH
cd "${SLIME_REPO_ROOT}"
sft_pid=""
requeue_sft() {
  trap - USR1
  echo "SFT_REQUEUE job=${SLURM_JOB_ID} at=$(date -Is)"
  if [[ -n "${sft_pid}" ]]; then
    kill -TERM "${sft_pid}" 2>/dev/null || true
    wait "${sft_pid}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_sft USR1
for path in \
  "${SFT_TORCH_COMPILE_CC}" \
  "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h" \
  "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"; do
  [[ -e "${path}" ]] || { echo "Missing SFT compiler prerequisite: ${path}" >&2; exit 1; }
done
[[ -x "${SFT_TORCH_COMPILE_CC}" ]] || { echo "SFT compiler is not executable: ${SFT_TORCH_COMPILE_CC}" >&2; exit 1; }
bash examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch &
sft_pid="$!"
wait "${sft_pid}"
sft_pid=""

IMPLEMENTATION_COMMIT="$(python3 -c 'import json,os; print(json.load(open(os.environ["CONTRACT_FILE"]))["repository_commit"])')"
export IMPLEMENTATION_COMMIT
python3 "${SCRIPT_DIR}/write_checkpoint_manifest.py" \
  --stage sft \
  --weights-root "${SFT_HF_SNAPSHOT_DIR}" \
  --state-root "${SFT_SAVE_DIR}" \
  --split-metadata "${SPLIT_METADATA}" \
  --out "${CHECKPOINT_REPORT_DIR}/sft_manifest.json"

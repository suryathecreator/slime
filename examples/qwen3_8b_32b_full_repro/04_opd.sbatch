#!/bin/bash
#SBATCH --job-name=q8b32b-opd1472
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
export CC="${OPD_TORCH_COMPILE_CC}"
export CPATH="${OPD_TORCH_COMPILE_CPATH}${CPATH:+:${CPATH}}"
mkdir -p "${TMPDIR}"
cd "${SLIME_REPO_ROOT}"
if [[ "$(realpath "${OPD_INITIAL_LOAD_DIR}")" != "$(realpath "${OPD_REF_LOAD_DIR}")" ]]; then
  echo "Initial actor/reference synchronization failed: ${OPD_INITIAL_LOAD_DIR} != ${OPD_REF_LOAD_DIR}" >&2
  exit 1
fi
opd_pid=""
requeue_opd() {
  trap - USR1
  echo "OPD_REQUEUE job=${SLURM_JOB_ID} at=$(date -Is)"
  if [[ -n "${opd_pid}" ]]; then
    kill -TERM "${opd_pid}" 2>/dev/null || true
    wait "${opd_pid}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_opd USR1
for path in \
  "${CC}" \
  "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h" \
  "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"; do
  [[ -e "${path}" ]] || { echo "Missing OPD compiler prerequisite: ${path}" >&2; exit 1; }
done
[[ -x "${CC}" ]] || { echo "OPD compiler is not executable: ${CC}" >&2; exit 1; }
"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
test -x "${CC}"
test -f "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h"
test -f "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"
export TRITON_CACHE_DIR="${TMPDIR}/opd-triton-compiler-preflight"
python3 -c "from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print(\"OPD_TRITON_COMPILER_READY\")"
'
bash examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch &
opd_pid="$!"
wait "${opd_pid}"
opd_pid=""

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

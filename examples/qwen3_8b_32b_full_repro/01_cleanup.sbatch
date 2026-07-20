#!/bin/bash
#SBATCH --job-name=q8b32b-cleanup
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
mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}" "${CLEANUP_WORK_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

cd "${SLIME_REPO_ROOT}"
cleanup_pid=""
requeue_cleanup() {
  trap - USR1
  echo "CLEANUP_REQUEUE job=${SLURM_JOB_ID} at=$(date -Is)"
  if [[ -n "${cleanup_pid}" ]]; then
    kill -TERM "${cleanup_pid}" 2>/dev/null || true
    wait "${cleanup_pid}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_cleanup USR1
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/clean_openthoughts3.py" --workers 32 &
cleanup_pid="$!"
wait "${cleanup_pid}"
cleanup_pid=""

python3 - "${SFT_PARQUET}" "${CLEANED_OPD_RESERVE_JSONL}" "${OPD_JSONL}" <<'PY'
import sys
from pathlib import Path

expected = (200_000, 100_000, 1_472)
if len(sys.argv[1:]) != len(expected):
    raise SystemExit("internal output-count argument mismatch")
for raw, count in zip(sys.argv[1:], expected):
    path = Path(raw)
    actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if actual != count:
        raise SystemExit(f"row-count mismatch {path}: {actual} != {count}")
print("CLEANUP_OUTPUT_COUNTS_OK")
PY

echo "CLEANUP_COMPLETE audit=${CLEANUP_AUDIT_JSON} split=${SPLIT_METADATA}"

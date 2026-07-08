#!/bin/bash
#SBATCH --job-name=slime-qwen3-vllm-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
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

mkdir -p "${SLURM_LOG_DIR}" "$(dirname "${VLLM_EVAL_VENV}")"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting vLLM eval environment setup on $(hostname) at $(date)"
echo "VLLM_EVAL_VENV=${VLLM_EVAL_VENV}"
echo "VLLM_EVAL_INSTALL_SPEC=${VLLM_EVAL_INSTALL_SPEC}"
echo "VLLM_EVAL_PIP_EXTRA_INDEX_URL=${VLLM_EVAL_PIP_EXTRA_INDEX_URL}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
if [[ ! -x "${VLLM_EVAL_VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VLLM_EVAL_VENV}"
fi
"${VLLM_EVAL_VENV}/bin/python" -m pip install --upgrade pip wheel setuptools
if ! "${VLLM_EVAL_VENV}/bin/python" -c "import vllm" >/dev/null 2>&1; then
  "${VLLM_EVAL_VENV}/bin/python" -m pip install "${VLLM_EVAL_INSTALL_SPEC}" --extra-index-url "${VLLM_EVAL_PIP_EXTRA_INDEX_URL}"
fi
"${VLLM_EVAL_VENV}/bin/python" - <<PY
import importlib.metadata as metadata
import vllm
print("vllm", metadata.version("vllm"))
PY
'

echo "Finished vLLM eval environment setup at $(date)"

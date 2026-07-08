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

mkdir -p "${SLURM_LOG_DIR}" "$(dirname "${VLLM_EVAL_VENV}")" "${VLLM_EVAL_SITE}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting vLLM eval environment setup on $(hostname) at $(date)"
echo "VLLM_EVAL_VENV=${VLLM_EVAL_VENV}"
echo "VLLM_EVAL_SITE=${VLLM_EVAL_SITE}"
echo "VLLM_EVAL_PYTHON=${VLLM_EVAL_PYTHON}"
echo "VLLM_EVAL_INSTALL_SPEC=${VLLM_EVAL_INSTALL_SPEC}"
echo "VLLM_EVAL_PIP_EXTRA_INDEX_URL=${VLLM_EVAL_PIP_EXTRA_INDEX_URL}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
vllm_target_libs() {
  find "${VLLM_EVAL_SITE}" -path "*/nvidia/*/lib" -type d -printf "%p:" 2>/dev/null || true
}

eval_python="${VLLM_EVAL_VENV}/bin/python"
if [[ ! -x "${eval_python}" ]]; then
  if python3 -m venv --system-site-packages "${VLLM_EVAL_VENV}"; then
    echo "Created vLLM venv: ${VLLM_EVAL_VENV}"
  else
    echo "python3 -m venv failed, likely because ensurepip/python3-venv is unavailable in the container." >&2
    echo "Falling back to scratch-local pip --target install at ${VLLM_EVAL_SITE}." >&2
    eval_python="${VLLM_EVAL_PYTHON}"
  fi
fi
if [[ "${eval_python}" != "${VLLM_EVAL_PYTHON}" ]] && ! "${eval_python}" -m pip --version >/dev/null 2>&1; then
  echo "Ignoring incomplete vLLM venv without pip: ${VLLM_EVAL_VENV}" >&2
  echo "Falling back to scratch-local pip --target install at ${VLLM_EVAL_SITE}." >&2
  eval_python="${VLLM_EVAL_PYTHON}"
fi

if [[ "${eval_python}" == "${VLLM_EVAL_PYTHON}" ]]; then
  export PYTHONPATH="${VLLM_EVAL_SITE}:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$(vllm_target_libs)${LD_LIBRARY_PATH:-}"
  export SLIME_SGLANG_PATCH_SITE=0
  if ! "${eval_python}" -c "import vllm" >/dev/null 2>&1; then
    "${eval_python}" -m pip install --target "${VLLM_EVAL_SITE}" "${VLLM_EVAL_INSTALL_SPEC}" --extra-index-url "${VLLM_EVAL_PIP_EXTRA_INDEX_URL}"
    export LD_LIBRARY_PATH="$(vllm_target_libs)${LD_LIBRARY_PATH:-}"
  fi
else
  "${eval_python}" -m pip install --upgrade pip wheel setuptools
  if ! "${eval_python}" -c "import vllm" >/dev/null 2>&1; then
    "${eval_python}" -m pip install "${VLLM_EVAL_INSTALL_SPEC}" --extra-index-url "${VLLM_EVAL_PIP_EXTRA_INDEX_URL}"
  fi
fi
"${eval_python}" - <<PY
import importlib.metadata as metadata
import vllm
print("vllm", metadata.version("vllm"))
PY
'

echo "Finished vLLM eval environment setup at $(date)"

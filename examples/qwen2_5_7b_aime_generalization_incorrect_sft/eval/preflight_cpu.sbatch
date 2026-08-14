#!/usr/bin/env bash
#SBATCH --job-name=q25-aime-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/aime_eval.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
: "${HELD_IN_EVAL_JSONL:?Set HELD_IN_EVAL_JSONL}"
: "${HELD_OUT_EVAL_JSONL:?Set HELD_OUT_EVAL_JSONL}"
"${RUNTIME_PYTHON}" "${EVAL_DIR}/../runtime_repo_gate.py" \
  --repo-root "${REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
CUDA_TOOLKIT_ROOT=/sw/cuda/12.8.1
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}" CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
[[ -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ]] || {
  echo "Missing executable CUDA 12.8 compiler: ${CUDA_TOOLKIT_ROOT}/bin/nvcc" >&2
  exit 2
}
"${CUDA_TOOLKIT_ROOT}/bin/nvcc" --version | grep -Fq "release 12.8" || {
  echo "Unexpected CUDA compiler version: ${CUDA_TOOLKIT_ROOT}/bin/nvcc" >&2
  exit 2
}
echo "AIME_CUDA_TOOLCHAIN_OK root=${CUDA_TOOLKIT_ROOT} nvcc=$(command -v nvcc)"
CONTROL_ROOT="$("${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" path --field control)"
mkdir -p "${CONTROL_ROOT}"
cd "${REPO_ROOT}"

"${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" prepare \
  --held-in "${HELD_IN_EVAL_JSONL}" --held-out "${HELD_OUT_EVAL_JSONL}"
"${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" verify-checkpoints \
  --full --status "${CONTROL_ROOT}/checkpoint_verification.json"
"${RUNTIME_PYTHON}" -m pytest -q \
  "${EVAL_DIR}/tests/test_aime_scorer.py" \
  "${EVAL_DIR}/tests/test_aime_eval.py"
echo "AIME_EVAL_PREFLIGHT_COMPLETE job=${SLURM_JOB_ID:?} at=$(date -Is)"

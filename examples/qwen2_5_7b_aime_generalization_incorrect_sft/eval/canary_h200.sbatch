#!/usr/bin/env bash
#SBATCH --job-name=q25-aime-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=00:30:00
#SBATCH --exclude=g3130
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/aime_eval.py"
GENERATOR="${EVAL_DIR}/generate_aime.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
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

common=(--repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}")
CANARY_TARGET=aime_correct_3000
model="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field model --target "${CANARY_TARGET}")"
tokenizer="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field tokenizer --target "${CANARY_TARGET}")"
identity="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field checkpoint-identity --target "${CANARY_TARGET}")"
smoke="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field smoke --dataset held_in)"
control_root="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field control)"
output="${control_root}/canary/${CANARY_TARGET}/held_in/repeat_00"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-aime-eval}/job_${SLURM_JOB_ID:?}/canary"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="$(mktemp -d "/tmp/q25aime-${SLURM_JOB_ID:?}.XXXXXX")"
mkdir -p "${output}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
cd "${REPO_ROOT}"

child_pid=""
requeue_job() {
  trap - USR1
  [[ -z "${child_pid}" ]] || { kill -TERM "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; }
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_job USR1
"${RUNTIME_PYTHON}" "${GENERATOR}" \
  --model "${model}" --tokenizer "${tokenizer}" \
  --checkpoint-manifest-sha256 "${identity}" --target "${CANARY_TARGET}" \
  --dataset held_in --repeat 0 --eval-file "${smoke}" --output-dir "${output}" \
  --chunk-size 1 --max-num-seqs 1 &
child_pid="$!"; wait "${child_pid}"; child_pid=""
echo "AIME_EVAL_CANARY_COMPLETE job=${SLURM_JOB_ID} at=$(date -Is)"

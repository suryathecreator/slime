#!/usr/bin/env bash
#SBATCH --job-name=q25-aime-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=02:00:00
#SBATCH --exclude=g3130
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/aime_eval.py"
GENERATOR="${EVAL_DIR}/generate_aime.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${EVAL_TARGET:?Set EVAL_TARGET}"
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
task="${SLURM_ARRAY_TASK_ID:?Run as array 0-23}"
[[ "${task}" =~ ^([0-9]|1[0-9]|2[0-3])$ ]] || { echo "Invalid array task: ${task}" >&2; exit 2; }

split_index=$((task / 12)); local_task=$((task % 12))
repeat=$((local_task / 4)); shard=$((local_task % 4))
if [[ "${split_index}" -eq 0 ]]; then dataset=held_in; else dataset=held_out_problem; fi
common=(--repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}")
model="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field model --target "${EVAL_TARGET}")"
tokenizer="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field tokenizer --target "${EVAL_TARGET}")"
identity="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field checkpoint-identity --target "${EVAL_TARGET}")"
eval_file="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field shard --dataset "${dataset}" --shard "${shard}")"
pass_output="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field pass-output --target "${EVAL_TARGET}" --dataset "${dataset}" --repeat "${repeat}")"
output="${pass_output}/shards/shard_$(printf '%02d' "${shard}")"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-aime-eval}/array_${SLURM_ARRAY_JOB_ID:?}/${EVAL_TARGET}/task_$(printf '%02d' "${task}")"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="$(mktemp -d "/tmp/q25aime-${SLURM_ARRAY_JOB_ID:?}-${task}.XXXXXX")"
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
  --checkpoint-manifest-sha256 "${identity}" --target "${EVAL_TARGET}" \
  --dataset "${dataset}" --repeat "${repeat}" --eval-file "${eval_file}" --output-dir "${output}" &
child_pid="$!"; wait "${child_pid}"; child_pid=""
echo "AIME_EVAL_SHARD_COMPLETE target=${EVAL_TARGET} dataset=${dataset} repeat=${repeat} shard=${shard} job=${SLURM_JOB_ID} at=$(date -Is)"

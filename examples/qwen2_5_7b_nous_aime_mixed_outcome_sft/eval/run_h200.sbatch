#!/usr/bin/env bash
#SBATCH --job-name=q25-nous-aime-eval
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
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/mixed_aime_eval.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${EVAL_TARGET:?Set EVAL_TARGET}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
draw="${SLURM_ARRAY_TASK_ID:?Run as array 0-15}"
[[ "${draw}" =~ ^([0-9]|1[0-5])$ ]] || { echo "Invalid draw index: ${draw}" >&2; exit 2; }
"${RUNTIME_PYTHON}" "${EVAL_DIR}/../../qwen2_5_7b_aime_generalization_incorrect_sft/runtime_repo_gate.py" \
  --repo-root "${REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"

CUDA_TOOLKIT_ROOT=/sw/cuda/12.8.1
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}" CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
[[ -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ]]
"${CUDA_TOOLKIT_ROOT}/bin/nvcc" --version | grep -Fq "release 12.8"

common=(--repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}")
model="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field model --target "${EVAL_TARGET}")"
tokenizer="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field tokenizer --target "${EVAL_TARGET}")"
identity="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field identity --target "${EVAL_TARGET}")"
output="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field draw-output --target "${EVAL_TARGET}" --draw "${draw}")"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-nous-aime-eval}/array_${SLURM_ARRAY_JOB_ID:?}/${EVAL_TARGET}/draw_$(printf '%02d' "${draw}")"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="$(mktemp -d "/tmp/q25-nous-aime-${SLURM_ARRAY_JOB_ID}-${draw}.XXXXXX")"
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
"${RUNTIME_PYTHON}" "${EVAL_DIR}/generate_mixed_aime.py" \
  --model "${model}" --tokenizer "${tokenizer}" --checkpoint-identity "${identity}" \
  --contract-file "${EVAL_EXPERIMENT_ROOT}/handoff/eval_contract.json" \
  --target "${EVAL_TARGET}" --draw-index "${draw}" \
  --eval-file "${EVAL_EXPERIMENT_ROOT}/data/eval/aime24_30.jsonl" \
  --eval-file "${EVAL_EXPERIMENT_ROOT}/data/eval/aime25_30.jsonl" --output-dir "${output}" &
child_pid="$!"; wait "${child_pid}"; child_pid=""
echo "MIXED_AIME_DRAW_COMPLETE target=${EVAL_TARGET} draw=${draw} job=${SLURM_JOB_ID} at=$(date -Is)"

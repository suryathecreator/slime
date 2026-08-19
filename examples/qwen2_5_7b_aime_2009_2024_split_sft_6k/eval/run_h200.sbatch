#!/usr/bin/env bash
#SBATCH --job-name=q25-split-aime-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=08:00:00
#SBATCH --exclude=g3130
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/split_aime_eval.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${EVAL_TARGET:?Set EVAL_TARGET}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"

# task_id = 2 * draw_index + split_index, over 4 draws x 2 splits.
task="${SLURM_ARRAY_TASK_ID:?Run as array 0-7}"
[[ "${task}" =~ ^[0-7]$ ]] || { echo "Invalid array task: ${task}" >&2; exit 2; }
draw=$(( task / 2 ))
split_index=$(( task % 2 ))
SPLITS=(held_in held_out)
split="${SPLITS[${split_index}]}"

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
output="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field draw-output \
  --target "${EVAL_TARGET}" --split "${split}" --draw "${draw}")"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-split-aime-eval}/array_${SLURM_ARRAY_JOB_ID:?}/${EVAL_TARGET}/${split}/draw_$(printf '%02d' "${draw}")"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="$(mktemp -d "/tmp/q25-split-aime-${SLURM_ARRAY_JOB_ID}-${task}.XXXXXX")"
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
"${RUNTIME_PYTHON}" "${EVAL_DIR}/generate_split_aime.py" \
  --model "${model}" --tokenizer "${tokenizer}" --checkpoint-identity "${identity}" \
  --target "${EVAL_TARGET}" --split "${split}" --draw-index "${draw}" \
  --eval-file "${EVAL_EXPERIMENT_ROOT}/data/eval/${split}_106.jsonl" --output-dir "${output}" &
child_pid="$!"; wait "${child_pid}"; child_pid=""
echo "SPLIT_AIME_DRAW_COMPLETE target=${EVAL_TARGET} split=${split} draw=${draw} job=${SLURM_JOB_ID} at=$(date -Is)"

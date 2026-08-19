#!/usr/bin/env bash
#SBATCH --job-name=q25-split-aime-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=00:30:00
#SBATCH --exclude=g3130
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/split_aime_eval.py"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
CANARY_TARGET="${EVAL_CANARY_TARGET:-held_in_correct_6000}"
"${RUNTIME_PYTHON}" "${EVAL_DIR}/../../qwen2_5_7b_aime_generalization_incorrect_sft/runtime_repo_gate.py" \
  --repo-root "${REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"

CUDA_TOOLKIT_ROOT=/sw/cuda/12.8.1
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}" CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
export PATH="${CUDA_TOOLKIT_ROOT}/bin:${PATH}"
[[ -x "${CUDA_TOOLKIT_ROOT}/bin/nvcc" ]]
"${CUDA_TOOLKIT_ROOT}/bin/nvcc" --version | grep -Fq "release 12.8"

common=(--repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}")
model="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field model --target "${CANARY_TARGET}")"
tokenizer="$("${RUNTIME_PYTHON}" "${CONTROL}" "${common[@]}" path --field tokenizer --target "${CANARY_TARGET}")"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-split-aime-eval}/canary_${SLURM_JOB_ID:?}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg" VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor" TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="$(mktemp -d "/tmp/q25-split-aime-canary-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
cd "${REPO_ROOT}"

"${RUNTIME_PYTHON}" "${EVAL_DIR}/canary_split_aime.py" \
  --model "${model}" --tokenizer "${tokenizer}" \
  --eval-file "${EVAL_EXPERIMENT_ROOT}/data/eval/held_in_106.jsonl"
echo "SPLIT_AIME_CANARY_COMPLETE target=${CANARY_TARGET} job=${SLURM_JOB_ID} at=$(date -Is)"

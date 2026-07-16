#!/bin/bash
#SBATCH --job-name=slime-qwen3-math500-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

requeue_before_timeout() {
  trap - USR1
  echo "Received USR1 at $(date); requeueing eval job ${SLURM_JOB_ID} with atomic sample resume."
  local pid
  for pid in $(jobs -pr); do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  if ! scontrol requeue "${SLURM_JOB_ID}"; then
    echo "Failed to requeue eval job ${SLURM_JOB_ID}." >&2
    exit 1
  fi
  exit 0
}
trap requeue_before_timeout USR1

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"

echo "Starting vLLM MATH-500 eval on $(hostname) at $(date)"
echo "Eval targets: ${EVAL_TARGETS}"
echo "Eval output: ${EVAL_OUTPUT_DIR}"
echo "vLLM venv: ${VLLM_EVAL_VENV}"
echo "vLLM target site: ${VLLM_EVAL_SITE}"
echo "vLLM workers/seqs/batched tokens: ${VLLM_EVAL_NUM_GPUS}/${VLLM_EVAL_MAX_NUM_SEQS}/${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}"
echo "vLLM memory/KV/load: utilization=${VLLM_EVAL_GPU_MEMORY_UTILIZATION} kv_dtype=${VLLM_EVAL_KV_CACHE_DTYPE} safetensors=${VLLM_EVAL_SAFETENSORS_LOAD_STRATEGY}"
echo "vLLM scheduler: direct_offline=1 chunked_prefill=${VLLM_EVAL_ENABLE_CHUNKED_PREFILL} prefix_cache=${VLLM_EVAL_ENABLE_PREFIX_CACHING}"
echo "vLLM speculative: method=${VLLM_EVAL_SPECULATIVE_METHOD} tokens=${VLLM_EVAL_NUM_SPECULATIVE_TOKENS} lookup=${VLLM_EVAL_PROMPT_LOOKUP_MIN}:${VLLM_EVAL_PROMPT_LOOKUP_MAX}"
echo "vLLM force native sampler: ${VLLM_EVAL_FORCE_NATIVE_SAMPLER}"
echo "vLLM per-sample resume: legacy_chunk_size=${VLLM_EVAL_CHUNK_SIZE} resume_completed=${VLLM_EVAL_RESUME_COMPLETED}"
echo "Eval max response/context: ${EVAL_MAX_RESPONSE_LEN}/${EVAL_MAX_CONTEXT_LEN}"
echo "Qwen3 thinking enabled: ${QWEN3_ENABLE_THINKING}"
echo "Qwen3 stop token ids: ${EVAL_STOP_TOKEN_IDS} (im_end=${QWEN3_IM_END_TOKEN_ID}, endoftext=${QWEN3_ENDOFTEXT_TOKEN_ID})"
echo "vLLM preserve special tokens: ${VLLM_EVAL_PRESERVE_SPECIAL_TOKENS}"
export SLIME_VLLM_PATCH_SITE="${VLLM_EVAL_FORCE_NATIVE_SAMPLER}"
echo "vLLM startup site patch: ${SLIME_VLLM_PATCH_SITE}"

if [[ "${QWEN3_ENABLE_THINKING}" != "1" ]]; then
  echo "This Qwen3 eval requires QWEN3_ENABLE_THINKING=1." >&2
  exit 2
fi
if [[ "${VLLM_EVAL_PRESERVE_SPECIAL_TOKENS}" != "1" ]]; then
  echo "This Qwen3 eval requires VLLM_EVAL_PRESERVE_SPECIAL_TOKENS=1." >&2
  exit 2
fi

if [[ -d "${VLLM_EVAL_SITE}/vllm" ]]; then
  export VLLM_EVAL_PYTHON_CMD="${VLLM_EVAL_PYTHON}"
  export PYTHONPATH="${VLLM_EVAL_SITE}:${PYTHONPATH:-}"
  VLLM_EVAL_TARGET_LIBS="$(find "${VLLM_EVAL_SITE}" -path '*/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null || true)"
  export LD_LIBRARY_PATH="${VLLM_EVAL_TARGET_LIBS}${LD_LIBRARY_PATH:-}"
  export SLIME_SGLANG_PATCH_SITE=0
elif [[ -x "${VLLM_EVAL_VENV}/bin/python" ]]; then
  export VLLM_EVAL_PYTHON_CMD="${VLLM_EVAL_VENV}/bin/python"
else
  echo "Missing vLLM eval environment: ${VLLM_EVAL_VENV} or ${VLLM_EVAL_SITE}" >&2
  echo "Run 10_setup_vllm_eval_env.sbatch first." >&2
  exit 1
fi
echo "vLLM python: ${VLLM_EVAL_PYTHON_CMD}"

"${SCRIPT_DIR}/container_exec.sh" "${VLLM_EVAL_PYTHON_CMD}" \
  examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py prepare-math500 \
  --out "${MATH500_JSONL}" \
  --dataset "${MATH500_DATASET}" \
  --hf-home "${HF_HOME}"

aggregate_if_available() {
  local dir="$1"
  local label="$2"
  if [[ -d "${dir}" ]] && find "${dir}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit | grep -q .; then
    echo "Aggregating ${label}: ${dir}"
    "${SCRIPT_DIR}/container_exec.sh" python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
      --aggregate-dir "${dir}" \
      --out-json "${dir}/summary_all.json"
  else
    echo "No per-stage summaries to aggregate for ${label}: ${dir}"
  fi
}

run_vllm_eval() {
  local stage="$1"
  local load_dir="$2"
  local train_samples="$3"
  local output_root="$4"
  local stage_dir="${output_root}/${stage}"
  local summary_file="${stage_dir}/summary.json"
  local debug_file="${stage_dir}/debug_eval_0.pt"
  mkdir -p "${stage_dir}"

  if [[ "${EVAL_SKIP_COMPLETED}" == "1" && -s "${summary_file}" ]]; then
    echo "VLLM_EVAL_SKIP_COMPLETED stage=${stage} summary=${summary_file}"
    aggregate_if_available "${output_root}" "${stage}"
    return
  fi

  if [[ ! -d "${load_dir}" ]]; then
    echo "Missing vLLM eval model dir for ${stage}: ${load_dir}" >&2
    exit 1
  fi

  echo "VLLM_EVAL_STAGE stage=${stage} load=${load_dir} train_samples=${train_samples} output=${stage_dir}"
"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
shard=0
while [[ "${shard}" -lt "${VLLM_EVAL_NUM_GPUS}" ]]; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${VLLM_EVAL_PYTHON_CMD}" examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py shard \
      --model "'"${load_dir}"'" \
      --data "${MATH500_JSONL}" \
      --out "'"${stage_dir}"'/debug_eval_shard_${shard}.pt" \
      --stage "'"${stage}"'" \
      --train-samples "'"${train_samples}"'" \
      --shard-index "${shard}" \
      --num-shards "${VLLM_EVAL_NUM_GPUS}" \
      --max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
      --max-context-len "${EVAL_MAX_CONTEXT_LEN}" \
      --max-model-len "${VLLM_EVAL_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}" \
      --max-num-seqs "${VLLM_EVAL_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}" \
      --dtype "${VLLM_EVAL_DTYPE}" \
      --kv-cache-dtype "${VLLM_EVAL_KV_CACHE_DTYPE}" \
      --safetensors-load-strategy "${VLLM_EVAL_SAFETENSORS_LOAD_STRATEGY}" \
      --speculative-method "${VLLM_EVAL_SPECULATIVE_METHOD}" \
      --num-speculative-tokens "${VLLM_EVAL_NUM_SPECULATIVE_TOKENS}" \
      --prompt-lookup-min "${VLLM_EVAL_PROMPT_LOOKUP_MIN}" \
      --prompt-lookup-max "${VLLM_EVAL_PROMPT_LOOKUP_MAX}" \
      --chunk-size "${VLLM_EVAL_CHUNK_SIZE}" \
      --enable-thinking \
      --stop-token-ids ${EVAL_STOP_TOKEN_IDS} \
      --im-end-token-id "${QWEN3_IM_END_TOKEN_ID}" \
      --endoftext-token-id "${QWEN3_ENDOFTEXT_TOKEN_ID}" \
      --preserve-special-tokens \
      '"$([[ "${VLLM_EVAL_ENABLE_PREFIX_CACHING}" == "1" ]] && printf "%s" "--enable-prefix-caching")"' \
      '"$([[ "${VLLM_EVAL_ENABLE_CHUNKED_PREFILL}" == "1" ]] && printf "%s" "--enable-chunked-prefill")"' \
      '"$([[ "${VLLM_EVAL_RESUME_COMPLETED}" == "1" ]] && printf "%s" "--resume-completed")"' \
      '"$([[ "${VLLM_EVAL_TRUST_REMOTE_CODE}" == "1" ]] && printf "%s" "--trust-remote-code")"' \
      --hf-home "${HF_HOME}"
  ) &
  shard=$((shard + 1))
done
wait
"${VLLM_EVAL_PYTHON_CMD}" examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py merge \
  --stage-dir "'"${stage_dir}"'" \
  --out "'"${debug_file}"'" \
  --expected-samples "${EVAL_EXPECTED_SAMPLES}"
'

  "${SCRIPT_DIR}/container_exec.sh" python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
    --stage "${stage}" \
    --debug-file "${debug_file}" \
    --out-json "${summary_file}" \
    --max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
    --expected-samples "${EVAL_EXPECTED_SAMPLES}" \
    --train-samples "${train_samples}"
  aggregate_if_available "${output_root}" "${stage}"
}

hf_snapshot_dir() {
  local root="$1"
  local rollout_id="$2"
  printf "%s/iter_%07d" "${root}" "${rollout_id}"
}

sample_count_for_rollout() {
  local rollout_id="$1"
  local batch_size="$2"
  local total_size="$3"
  local count=$(((rollout_id + 1) * batch_size))
  if [[ "${count}" -gt "${total_size}" ]]; then
    count="${total_size}"
  fi
  printf "%s" "${count}"
}

for target in ${EVAL_TARGETS}; do
  case "${target}" in
    base)
      run_vllm_eval base "${STUDENT_HF_DIR}" 0 "${BASE_EVAL_OUTPUT_DIR}"
      ;;
    sft)
      sft_samples="$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))"
      sft_stage="$(printf "sft_%06d" "${sft_samples}")"
      run_vllm_eval "${sft_stage}" "${SFT_FINAL_HF_DIR}" "${sft_samples}" "${SFT_EVAL_OUTPUT_DIR}"
      ;;
    opd)
      for rid in ${OPD_MILESTONE_ROLLOUT_IDS}; do
        samples="$(sample_count_for_rollout "${rid}" "${OPD_ROLLOUT_BATCH_SIZE}" "${OPD_TRAIN_SIZE}")"
        stage="$(printf "opd_%06d" "${samples}")"
        run_vllm_eval "${stage}" "$(hf_snapshot_dir "${OPD_HF_SNAPSHOT_DIR}" "${rid}")" "${samples}" "${OPD_EVAL_OUTPUT_DIR}"
      done
      ;;
    *)
      echo "Unknown vLLM EVAL_TARGETS entry: ${target}" >&2
      exit 2
      ;;
  esac
done

echo "Finished vLLM MATH-500 eval at $(date)"

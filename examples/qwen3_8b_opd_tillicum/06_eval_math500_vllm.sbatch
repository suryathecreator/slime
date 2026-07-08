#!/bin/bash
#SBATCH --job-name=slime-qwen3-math500-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

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
echo "Eval max response/context: ${EVAL_MAX_RESPONSE_LEN}/${EVAL_MAX_CONTEXT_LEN}"

if [[ -d "${VLLM_EVAL_SITE}/vllm" ]]; then
  export VLLM_EVAL_PYTHON_CMD="${VLLM_EVAL_PYTHON}"
  export PYTHONPATH="${VLLM_EVAL_SITE}:${PYTHONPATH:-}"
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
for shard in $(seq 0 $((VLLM_EVAL_NUM_GPUS - 1))); do
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
      '"$([[ "${VLLM_EVAL_TRUST_REMOTE_CODE}" == "1" ]] && printf "%s" "--trust-remote-code")"' \
      --hf-home "${HF_HOME}"
  ) &
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

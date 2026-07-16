#!/bin/bash
#SBATCH --job-name=q3-harp-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

requeue_before_timeout() {
  trap - USR1
  for pid in $(jobs -pr); do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_before_timeout USR1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
: "${EVAL_MODE:?Set EVAL_MODE=tune or eval}"
: "${EVAL_MODEL_DIR:?Set EVAL_MODEL_DIR}"
: "${EVAL_TOKENIZER_DIR:?Set EVAL_TOKENIZER_DIR}"
: "${EVAL_STAGE_DIR:?Set EVAL_STAGE_DIR}"
: "${EVAL_TUNING_FILE:?Set EVAL_TUNING_FILE}"

mkdir -p "${SLURM_LOG_DIR}" "${EVAL_STAGE_DIR}" "$(dirname "${EVAL_TUNING_FILE}")" "${VLLM_CACHE_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"
echo "Starting direct offline-vLLM ${EVAL_MODE} for ${EVAL_MODEL_DIR} on $(hostname) at $(date)"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0

"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export PYTHONPATH="${HARP_EVAL_SITE}:${VLLM_EVAL_SITE}:${HARP_EXAMPLE_DIR}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(find "${VLLM_EVAL_SITE}" -path "*/nvidia/*/lib" -type d -printf "%p:" 2>/dev/null || true)${LD_LIBRARY_PATH:-}"
export VLLM_CACHE_ROOT
evaluator="examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py"

run_candidate() {
  local name="$1"
  local max_num_seqs="$2"
  local eager="$3"
  local candidate_dir="${EVAL_STAGE_DIR}/${name}"
  local summary="${candidate_dir}/benchmark.json"
  if [[ -s "${summary}" ]]; then
    echo "Reusing completed tuning candidate ${summary}"
    return 0
  fi
  mkdir -p "${candidate_dir}"
  complete_shards=0
  for shard in 0 1 2 3; do
    if [[ -s "${candidate_dir}/shard_$(printf "%02d" "${shard}")_manifest.json" && -s "${candidate_dir}/shard_$(printf "%02d" "${shard}")_benchmark.jsonl" ]]; then
      complete_shards=$((complete_shards + 1))
    fi
  done
  if (( complete_shards == 4 )); then
    python3 "${evaluator}" merge-benchmark --stage-dir "${candidate_dir}" --output "${summary}"
    return 0
  fi
  local eager_arg=()
  if [[ "${eager}" == "1" ]]; then
    eager_arg+=(--enforce-eager)
  fi
  local pids=()
  local ready_files=()
  local shard
  for shard in 0 1 2 3; do
    ready="${candidate_dir}/ready_${SLURM_JOB_ID:-manual}_${shard}"
    ready_files+=("${ready}")
    CUDA_VISIBLE_DEVICES="${shard}" python3 "${evaluator}" shard \
      --model "${EVAL_MODEL_DIR}" \
      --tokenizer "${EVAL_TOKENIZER_DIR}" \
      --data "${HARP_JSONL}" \
      --output-dir "${candidate_dir}" \
      --shard-index "${shard}" \
      --num-shards 4 \
      --expected-samples "${EVAL_EXPECTED_SAMPLES}" \
      --max-model-len "${VLLM_EVAL_MAX_MODEL_LEN}" \
      --max-response-len "${VLLM_EVAL_MAX_RESPONSE_LEN}" \
      --max-num-seqs "${max_num_seqs}" \
      --max-num-batched-tokens "${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}" \
      --benchmark \
      --benchmark-total-size "${VLLM_EVAL_BENCHMARK_SIZE}" \
      --benchmark-max-tokens "${VLLM_EVAL_BENCHMARK_MAX_TOKENS}" \
      --ready-file "${ready}" \
      --resume \
      "${eager_arg[@]}" \
      >"${candidate_dir}/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done

  local elapsed=0
  while (( elapsed < 900 )); do
    local ready_count=0
    local dead_unready=0
    local index
    for index in 0 1 2 3; do
      [[ -s "${ready_files[${index}]}" ]] && ready_count=$((ready_count + 1))
      if [[ ! -s "${ready_files[${index}]}" ]] && ! kill -0 "${pids[${index}]}" 2>/dev/null; then
        dead_unready=$((dead_unready + 1))
      fi
    done
    if (( ready_count == 4 )); then
      break
    fi
    if (( dead_unready > 0 )); then
      echo "Candidate ${name} lost a worker during engine initialization" >&2
      for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
      wait || true
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if (( elapsed >= 900 )); then
    echo "Candidate ${name} exceeded the 15-minute CUDA-graph startup limit" >&2
    for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
    wait || true
    return 1
  fi

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed != 0 )); then
    echo "Candidate ${name} failed generation" >&2
    return 1
  fi
  python3 "${evaluator}" merge-benchmark --stage-dir "${candidate_dir}" --output "${summary}"
}

if [[ "${EVAL_MODE}" == "tune" ]]; then
  valid_graph=0
  run_candidate graph_maxseq4 4 0 && valid_graph=$((valid_graph + 1)) || true
  run_candidate graph_maxseq8 8 0 && valid_graph=$((valid_graph + 1)) || true
  if (( valid_graph == 0 )); then
    run_candidate graph_maxseq2 2 0 && valid_graph=1 || true
  fi
  if (( valid_graph == 0 )); then
    echo "Every CUDA-graph candidate failed; testing eager fallback."
    run_candidate eager_maxseq4 4 1 || true
    run_candidate eager_maxseq8 8 1 || true
  fi
  candidate_args=()
  shopt -s nullglob
  for summary in "${EVAL_STAGE_DIR}"/*/benchmark.json; do
    candidate_args+=(--candidate "${summary}")
  done
  shopt -u nullglob
  python3 "${evaluator}" select-tuning "${candidate_args[@]}" --output "${EVAL_TUNING_FILE}"
elif [[ "${EVAL_MODE}" == "eval" ]]; then
  read -r max_num_seqs eager < <(python3 - "${EVAL_TUNING_FILE}" <<"PY"
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))["selected_engine_config"]
print(config["max_num_seqs"], int(bool(config["enforce_eager"])))
PY
  )
  eager_arg=()
  [[ "${eager}" == "1" ]] && eager_arg+=(--enforce-eager)
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${shard}" python3 "${evaluator}" shard \
      --model "${EVAL_MODEL_DIR}" \
      --tokenizer "${EVAL_TOKENIZER_DIR}" \
      --data "${HARP_JSONL}" \
      --output-dir "${EVAL_STAGE_DIR}" \
      --shard-index "${shard}" \
      --num-shards 4 \
      --expected-samples "${EVAL_EXPECTED_SAMPLES}" \
      --max-model-len "${VLLM_EVAL_MAX_MODEL_LEN}" \
      --max-response-len "${VLLM_EVAL_MAX_RESPONSE_LEN}" \
      --max-num-seqs "${max_num_seqs}" \
      --max-num-batched-tokens "${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}" \
      --resume \
      "${eager_arg[@]}" \
      >"${EVAL_STAGE_DIR}/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || exit 1
  python3 "${evaluator}" merge --stage-dir "${EVAL_STAGE_DIR}" --data "${HARP_JSONL}" --output-dir "$(dirname "${EVAL_STAGE_DIR}")"
else
  echo "Unknown EVAL_MODE=${EVAL_MODE}" >&2
  exit 2
fi
'

echo "Finished direct offline-vLLM ${EVAL_MODE} at $(date)"

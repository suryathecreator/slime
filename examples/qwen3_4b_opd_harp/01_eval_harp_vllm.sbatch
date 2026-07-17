#!/bin/bash
#SBATCH --job-name=q3-harp-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=384G
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

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
: "${EVAL_MODE:?Set EVAL_MODE=tune or eval}"
: "${EVAL_MODEL_DIR:?Set EVAL_MODEL_DIR}"
: "${EVAL_TOKENIZER_DIR:?Set EVAL_TOKENIZER_DIR}"
: "${EVAL_STAGE_DIR:?Set EVAL_STAGE_DIR}"
if [[ "${EVAL_MODE}" == "tune" ]]; then
  : "${EVAL_TUNING_FILE:?Set EVAL_TUNING_FILE for tuning}"
elif [[ "${EVAL_MODE}" == "eval" ]]; then
  : "${EVAL_MAX_NUM_SEQS:?Set EVAL_MAX_NUM_SEQS for evaluation}"
  : "${EVAL_PREFER_CUDA_GRAPH:?Set EVAL_PREFER_CUDA_GRAPH=0 or 1}"
fi

mkdir -p "${SLURM_LOG_DIR}" "${EVAL_STAGE_DIR}" "${VLLM_CACHE_ROOT}"
if [[ "${EVAL_MODE}" == "tune" ]]; then
  mkdir -p "$(dirname "${EVAL_TUNING_FILE}")"
fi
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
export CC="${HARP_ZIG_CC}"
export CPATH="${HARP_ZIG_INCLUDE_ROOT}/python3.12:${HARP_ZIG_INCLUDE_ROOT}${CPATH:+:${CPATH}}"
test -x "${CC}"
test -f "${HARP_ZIG_INCLUDE_ROOT}/python3.12/Python.h"
test -f "${HARP_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"
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
  if [[ "${EVAL_PREFER_CUDA_GRAPH}" != "0" && "${EVAL_PREFER_CUDA_GRAPH}" != "1" ]]; then
    echo "EVAL_PREFER_CUDA_GRAPH must be 0 or 1, found ${EVAL_PREFER_CUDA_GRAPH}" >&2
    exit 2
  fi

  run_eval_attempt() {
    local attempt_name="$1"
    local eager="$2"
    local eager_arg=()
    local pids=()
    local ready_files=()
    local shard ready
    [[ "${eager}" == "1" ]] && eager_arg+=(--enforce-eager)

    echo "Launching production HARP evaluation attempt=${attempt_name} max_num_seqs=${EVAL_MAX_NUM_SEQS} epoch=${EPOCHSECONDS}"
    for shard in 0 1 2 3; do
      ready="${EVAL_STAGE_DIR}/ready_${SLURM_JOB_ID:-manual}_${attempt_name}_${shard}"
      ready_files+=("${ready}")
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
        --max-num-seqs "${EVAL_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${VLLM_EVAL_GPU_MEMORY_UTILIZATION}" \
        --ready-file "${ready}" \
        --resume \
        "${eager_arg[@]}" \
        >"${EVAL_STAGE_DIR}/shard_${shard}_${attempt_name}.log" 2>&1 &
      pids+=("$!")
    done

    local elapsed=0
    while (( elapsed < VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS )); do
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
        echo "All production engines are ready for attempt=${attempt_name}; continuing the real evaluation."
        break
      fi
      if (( dead_unready > 0 )); then
        echo "Evaluation attempt=${attempt_name} lost a worker during engine initialization" >&2
        for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
        wait || true
        return 1
      fi
      sleep 1
      elapsed=$((elapsed + 1))
    done
    if (( elapsed >= VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS )); then
      echo "Evaluation attempt=${attempt_name} exceeded the engine initialization timeout" >&2
      for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
      wait || true
      return 1
    fi

    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "Evaluation attempt=${attempt_name} failed after every engine reported ready" >&2
      return 2
    fi
    return 0
  }

  selected_mode=eager
  if [[ "${EVAL_PREFER_CUDA_GRAPH}" == "1" ]]; then
    if run_eval_attempt cuda_graph 0; then
      selected_mode=cuda_graph
    else
      graph_status=$?
      if (( graph_status != 1 )); then
        exit "${graph_status}"
      fi
      echo "CUDA-graph engine initialization failed; retrying the real evaluation in eager mode." >&2
      run_eval_attempt eager 1
    fi
  else
    run_eval_attempt eager 1
  fi
  echo "Production HARP evaluation selected_mode=${selected_mode} max_num_seqs=${EVAL_MAX_NUM_SEQS}"
  python3 "${evaluator}" merge --stage-dir "${EVAL_STAGE_DIR}" --data "${HARP_JSONL}" --output-dir "${EVAL_STAGE_DIR%/*}"
else
  echo "Unknown EVAL_MODE=${EVAL_MODE}" >&2
  exit 2
fi
'

echo "Finished direct offline-vLLM ${EVAL_MODE} at $(date)"

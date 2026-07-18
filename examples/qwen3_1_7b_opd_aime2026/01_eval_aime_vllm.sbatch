#!/bin/bash
#SBATCH --job-name=q3-aime26-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
: "${EVAL_MODEL_DIR:?Set EVAL_MODEL_DIR}"
: "${EVAL_TOKENIZER_DIR:?Set EVAL_TOKENIZER_DIR}"
: "${EVAL_STAGE_DIR:?Set EVAL_STAGE_DIR}"
: "${EVAL_MAX_NUM_SEQS:?Set EVAL_MAX_NUM_SEQS}"
mkdir -p "${SLURM_LOG_DIR}" "${EVAL_STAGE_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"
for required in "${AIME_JSONL}" "${EVAL_MODEL_DIR}/config.json" "${EVAL_TOKENIZER_DIR}/config.json" "${VLLM_EVAL_SITE}/vllm"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing evaluation prerequisite: ${required}" >&2
    exit 1
  fi
done
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0

"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export PYTHONPATH="${VLLM_EVAL_SITE}:${AIME_EXAMPLE_DIR}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(find "${VLLM_EVAL_SITE}" -path "*/nvidia/*/lib" -type d -printf "%p:" 2>/dev/null || true)${LD_LIBRARY_PATH:-}"
evaluator=examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
python3 - <<"PY"
import importlib.metadata as metadata
import vllm
print(f"VLLM_IMPORT_PREFLIGHT_OK version={metadata.version('vllm')} module={vllm.__file__}", flush=True)
PY

run_attempt() {
  local attempt_name="$1"
  local eager="$2"
  local eager_arg=()
  local pids=()
  local ready_files=()
  local shard ready
  [[ "${eager}" == "1" ]] && eager_arg+=(--enforce-eager)
  echo "Launching AIME 2026 attempt=${attempt_name} max_num_seqs=${EVAL_MAX_NUM_SEQS} epoch=${EPOCHSECONDS}"
  for shard in 0 1 2 3; do
    ready="${EVAL_STAGE_DIR}/ready_${SLURM_JOB_ID:-manual}_${attempt_name}_${shard}"
    ready_files+=("${ready}")
    CUDA_VISIBLE_DEVICES="${shard}" python3 "${evaluator}" shard \
      --model "${EVAL_MODEL_DIR}" \
      --tokenizer "${EVAL_TOKENIZER_DIR}" \
      --data "${AIME_JSONL}" \
      --output-dir "${EVAL_STAGE_DIR}" \
      --shard-index "${shard}" \
      --num-shards 4 \
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
      echo "All AIME engines ready for attempt=${attempt_name}"
      break
    fi
    if (( dead_unready > 0 )); then
      for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
      wait || true
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if (( elapsed >= VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS )); then
    for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
    wait || true
    return 1
  fi
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "AIME attempt=${attempt_name} failed after engine readiness" >&2
    return 2
  fi
}

selected_mode=eager
if [[ "${VLLM_EVAL_PREFER_CUDA_GRAPH}" == "1" ]]; then
  if run_attempt cuda_graph 0; then
    selected_mode=cuda_graph
  else
    graph_status=$?
    if (( graph_status != 1 )); then exit "${graph_status}"; fi
    echo "CUDA-graph initialization failed; retrying AIME evaluation eagerly." >&2
    run_attempt eager 1
  fi
else
  run_attempt eager 1
fi
echo "AIME 2026 evaluation selected_mode=${selected_mode}"
python3 "${evaluator}" merge \
  --stage-dir "${EVAL_STAGE_DIR}" \
  --data "${AIME_JSONL}" \
  --output-dir "${EVAL_STAGE_DIR%/*}"
'

echo "Finished sampled AIME 2026 evaluation at $(date)"

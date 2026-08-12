#!/usr/bin/env bash

run_token_audit() {
  local data_path="${1:?data path required}"
  local model_path="${2:?model path required}"
  local output_path="${3:?output path required}"
  local phase="${4:?phase required}"
  local details_root="${5:-}"
  local -a command=(
    python3 "${AIME_GENERALIZATION_DIR}/runtime_token_audit.py"
    --data "${data_path}" --model "${model_path}"
    --output "${output_path}" --phase "${phase}"
    --max-sequence-length "${SFT_SEQ_LENGTH}"
    --tensor-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"
    --data-parallel-size "$((SFT_ACTOR_GPUS / (SFT_TENSOR_MODEL_PARALLEL_SIZE * SFT_CONTEXT_PARALLEL_SIZE * SFT_PIPELINE_MODEL_PARALLEL_SIZE)))"
  )
  if [[ -n "${details_root}" ]]; then
    local path
    local train_count=0
    for path in "${details_root}/rollout_data/0.pt" "${details_root}/rollout_data/4.pt"; do
      [[ -f "${path}" ]] || { echo "Missing runtime rollout evidence: ${path}" >&2; return 1; }
      command+=(--rollout-data "${path}")
    done
    while IFS= read -r path; do
      command+=(--train-data "${path}")
      train_count=$((train_count + 1))
    done < <(find "${details_root}/train_data" -maxdepth 1 -type f \( -name '0_*.pt' -o -name '4_*.pt' \) -print | sort)
    local expected_train_count=$((2 * SFT_ACTOR_GPUS))
    [[ "${train_count}" -eq "${expected_train_count}" ]] || {
      echo "Expected ${expected_train_count} actor train dumps for rollouts 0 and 4, found ${train_count}" >&2
      return 1
    }
  fi
  "${AIME_GENERALIZATION_DIR}/../qwen3_8b_opd_tillicum/container_exec.sh" \
    bash -lc 'cd "${SLIME_REPO_ROOT}"; exec "$@"' bash "${command[@]}"
}

run_copy_diagnostic_nonblocking() {
  local model_path="${1:?model path required}"
  local output_path="${2:?output path required}"
  local phase="${3:?phase required}"
  set +e
  "${AIME_GENERALIZATION_DIR}/../qwen3_8b_opd_tillicum/container_exec.sh" \
    bash -lc 'cd "${SLIME_REPO_ROOT}"; exec python3 "${AIME_GENERALIZATION_DIR}/copy_regression_diagnostic.py" --problems "${HELD_IN_EVAL_JSONL}" --model "$1" --output "$2" --phase "$3"' \
    bash "${model_path}" "${output_path}" "${phase}"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "NONBLOCKING_COPY_DIAGNOSTIC_FAILED phase=${phase} status=${status}" >&2
  fi
  return 0
}

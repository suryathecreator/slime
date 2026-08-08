#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }

validate_args=(--example-dir "${SCRIPT_DIR}")
if [[ "${DRY_RUN}" == 0 ]]; then
  validate_args+=(--require-pushed-clean)
fi
python3 "${SCRIPT_DIR}/validate_config.py" "${validate_args[@]}"

if [[ "${DRY_RUN}" == 1 ]]; then
  for name in \
    prepare_model_qwen2_5_3b prepare_model_qwen2_5_7b \
    prepare_model_qwen3_4b prepare_model_qwen3_8b prepare_shared_data \
    canary_qwen2_5_3b_8k canary_qwen2_5_3b_16k canary_qwen2_5_7b_8k \
    canary_qwen3_4b_8k canary_qwen3_8b_8k \
    train_qwen2_5_3b_8k train_qwen2_5_3b_16k train_qwen2_5_7b_8k \
    train_qwen3_4b_8k train_qwen3_8b_8k finalize_handoff; do
    echo "DRY_RUN_SERIAL_AFTEROK name=${name}"
  done
  echo "DRY_RUN_COMPLETE jobs=16 maximum_concurrent_gpus=4 eval_jobs=0 transfer_jobs=0"
  exit 0
fi

SUBMISSION_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
[[ ! -e "${EXPERIMENT_ROOT}" ]] || {
  echo "Refusing to reuse existing experiment root ${EXPERIMENT_ROOT}" >&2
  exit 1
}
mkdir -p "${MANIFEST_ROOT}"

submitted_jobs=()
manifest_jobs=()
submission_complete=0
tail_job=""
cancel_partial() {
  local status=$?
  trap - EXIT
  if [[ "${submission_complete}" != 1 && "${#submitted_jobs[@]}" -gt 0 ]]; then
    echo "Submission failed; canceling partial chain ${submitted_jobs[*]}" >&2
    scancel "${submitted_jobs[@]}" || true
  fi
  exit "${status}"
}
trap cancel_partial EXIT

submit_serial() {
  local name="$1"
  shift
  local args=(
    --parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}"
    --partition="${PARTITION}" --qos="${QOS}"
  )
  if [[ -n "${tail_job}" ]]; then
    args+=(--dependency="afterok:${tail_job}")
  fi
  local raw
  raw="$(sbatch "${args[@]}" "$@")"
  tail_job="${raw%%;*}"
  [[ "${tail_job}" =~ ^[0-9]+$ ]]
  submitted_jobs+=("${tail_job}")
  manifest_jobs+=("${name}=${tail_job}")
  echo "SUBMITTED name=${name} job=${tail_job}"
}

for model_key in qwen2_5_3b qwen2_5_7b qwen3_4b qwen3_8b; do
  submit_serial "prepare_model_${model_key}" \
    --job-name="qco-model-${model_key}" --export=ALL,MODEL_KEY="${model_key}" \
    "${SCRIPT_DIR}/00_prepare_model.sbatch"
done
submit_serial prepare_shared_data "${SCRIPT_DIR}/01_prepare_data.sbatch"
for run_key in \
  qwen2_5_3b_8k qwen2_5_3b_16k qwen2_5_7b_8k qwen3_4b_8k qwen3_8b_8k; do
  submit_serial "canary_${run_key}" \
    --job-name="qco-canary-${run_key}" --export=ALL,RUN_KEY="${run_key}" \
    "${SCRIPT_DIR}/02_canary.sbatch"
done
for run_key in \
  qwen2_5_3b_8k qwen2_5_3b_16k qwen2_5_7b_8k qwen3_4b_8k qwen3_8b_8k; do
  submit_serial "train_${run_key}" \
    --job-name="qco-train-${run_key}" --export=ALL,RUN_KEY="${run_key}" \
    "${SCRIPT_DIR}/03_train.sbatch"
done
submit_serial finalize_handoff "${SCRIPT_DIR}/04_finalize.sbatch"

python3 "${SCRIPT_DIR}/write_submission_manifest.py" \
  --output "${SUBMISSION_MANIFEST}" --contract "${CONTRACT_HASH}" \
  --branch qwen-correct-only-openr1-sft-2k "${manifest_jobs[@]}"
submission_complete=1
trap - EXIT
echo "SUBMISSION_COMPLETE jobs=${#submitted_jobs[@]} first=${submitted_jobs[0]} last=${tail_job}"

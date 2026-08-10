#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

MODE="${1:-}"
[[ "${MODE}" == "--dry-run" || "${MODE}" == "--submit" ]] || {
  echo "Usage: $0 --dry-run|--submit" >&2
  exit 2
}
[[ $# -eq 1 ]] || { echo "Usage: $0 --dry-run|--submit" >&2; exit 2; }

validate_args=(--example-dir "${SCRIPT_DIR}" --require-source-artifacts)
if [[ "${MODE}" == "--submit" ]]; then
  validate_args+=(--require-pushed-clean)
fi
python3 "${SCRIPT_DIR}/validate_config.py" "${validate_args[@]}"

for batch_script in \
  01_prepare_data.sbatch 02_score_and_build.sbatch 03_prefix_canary.sbatch \
  04_train.sbatch 05_finalize.sbatch; do
  sbatch --test-only --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
    --partition="${PARTITION}" --qos="${QOS}" "${SCRIPT_DIR}/${batch_script}"
done
echo "SBATCH_PREFLIGHT_COMPLETE scripts=5"

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "DRY_RUN_SERIAL_AFTEROK name=prepare_data"
  echo "DRY_RUN_SERIAL_AFTEROK name=score_and_build"
  echo "DRY_RUN_SERIAL_AFTEROK name=prefix_canary"
  for variant in "${VARIANTS[@]}"; do
    echo "DRY_RUN_SERIAL_AFTEROK name=train_${variant}"
  done
  echo "DRY_RUN_SERIAL_AFTEROK name=finalize_handoff"
  echo "DRY_RUN_COMPLETE jobs=14 maximum_concurrent_gpus=4 eval_jobs=0 transfer_jobs=0"
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

submit_serial prepare_data "${SCRIPT_DIR}/01_prepare_data.sbatch"
submit_serial score_and_build "${SCRIPT_DIR}/02_score_and_build.sbatch"
submit_serial prefix_canary "${SCRIPT_DIR}/03_prefix_canary.sbatch"
for variant in "${VARIANTS[@]}"; do
  submit_serial "train_${variant}" \
    --job-name="q25crm-${variant}" --export=ALL,SFT_VARIANT="${variant}" \
    "${SCRIPT_DIR}/04_train.sbatch"
done
submit_serial finalize_handoff "${SCRIPT_DIR}/05_finalize.sbatch"

python3 "${SCRIPT_DIR}/write_submission_manifest.py" \
  --output "${SUBMISSION_MANIFEST}" --contract "${CONTRACT_HASH}" \
  --branch qwen-correct-only-openr1-sft-2k "${manifest_jobs[@]}"
submission_complete=1
trap - EXIT
echo "SUBMISSION_COMPLETE jobs=${#submitted_jobs[@]} first=${submitted_jobs[0]} last=${tail_job}"

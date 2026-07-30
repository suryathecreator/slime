#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
python3 "${SCRIPT_DIR}/validate_config.py" \
  --example-dir "${SCRIPT_DIR}" --require-pushed-clean
SUBMISSION_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
[[ ! -e "${SUBMISSION_MANIFEST}" ]] || {
  echo "Refusing to overwrite prior submission ${SUBMISSION_MANIFEST}" >&2
  exit 1
}
for target in "${STUDENT_HF_DIR}" "${STUDENT_TORCH_DIST_DIR}" "${DATA_ROOT}" "${OUTPUT_ROOT}"; do
  [[ ! -e "${target}" ]] || {
    echo "Refusing to overwrite prior experiment artifact ${target}" >&2
    exit 1
  }
done
mkdir -p "${MANIFEST_ROOT}"
submitted_jobs=()
manifest_jobs=()
submission_complete=0
tail_job=""
cancel_partial() {
  local status=$?
  trap - EXIT
  if [[ "${submission_complete}" != "1" && "${#submitted_jobs[@]}" -gt 0 ]]; then
    echo "Submission failed; canceling partial chain ${submitted_jobs[*]}" >&2
    scancel "${submitted_jobs[@]}" || true
  fi
  exit "${status}"
}
trap cancel_partial EXIT
submit_serial() {
  local name="$1"
  shift
  local args=(--parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}")
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
submit_serial prepare_qwen25_7b_model "${SCRIPT_DIR}/00_prepare_model.sbatch"
submit_serial prepare_qwen25_2k_data "${SCRIPT_DIR}/01_prepare.sbatch"
submit_serial smoke_qwen25_full_sft "${SCRIPT_DIR}/02_smoke.sbatch"
submit_serial score_qwen25_margins_and_build "${SCRIPT_DIR}/03_score_and_build.sbatch"
for variant in \
  correct_only unmasked random_mask_70 inverse_tau_0p20 inverse_tau_0p05 \
  random_mask_25 random_mask_50 random_mask_80 random_mask_90 margin_mask prob_ratio_mask; do
  submit_serial "train_qwen25_2k_${variant}" \
    --job-name="q25-2k-${variant}" \
    --export=ALL,SFT_VARIANT="${variant}" \
    "${SCRIPT_DIR}/04_train.sbatch"
done
python3 "${SCRIPT_DIR}/write_submission_manifest.py" \
  --output "${SUBMISSION_MANIFEST}" --contract "${CONTRACT_HASH}" "${manifest_jobs[@]}"
submission_complete=1
trap - EXIT
echo "SUBMISSION_COMPLETE jobs=${#submitted_jobs[@]} first=${submitted_jobs[0]} last=${tail_job}"

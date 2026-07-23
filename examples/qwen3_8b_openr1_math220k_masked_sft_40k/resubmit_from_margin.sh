#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
python3 "${SCRIPT_DIR}/validate_config.py" --example-dir "${SCRIPT_DIR}" --require-pushed-clean

for path in "${PREP_STATS_JSON}" "${SELECTED_TRACES_JSONL}" "${CORRECT_ONLY_JSONL}" "${UNMASKED_JSONL}"; do
  [[ -s "${path}" ]] || { echo "Missing completed preparation artifact ${path}" >&2; exit 1; }
done
REWIRE_MANIFEST="${MANIFEST_ROOT}/resubmission_from_margin.json"
[[ ! -e "${REWIRE_MANIFEST}" ]] || { echo "Refusing to overwrite ${REWIRE_MANIFEST}" >&2; exit 1; }

submitted_jobs=()
submission_complete=0
cancel_partial() {
  local status=$?
  trap - EXIT
  if [[ "${submission_complete}" != "1" && "${#submitted_jobs[@]}" -gt 0 ]]; then
    scancel "${submitted_jobs[@]}" || true
  fi
  exit "${status}"
}
trap cancel_partial EXIT

submit_job() {
  local output_name="$1"
  local dependency="$2"
  shift 2
  local args=(--parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="afterok:${dependency}"); fi
  local raw job_id
  raw="$(sbatch "${args[@]}" "$@")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response ${raw}" >&2; return 1; }
  submitted_jobs+=("${job_id}")
  printf -v "${output_name}" '%s' "${job_id}"
}

submit_job margin_job "" "${SCRIPT_DIR}/02_score_and_build.sbatch"
submit_job correct_train_job "${margin_job}" --job-name=q8b-sft-correct --export=ALL,SFT_VARIANT=correct_only "${SCRIPT_DIR}/03_train.sbatch"
submit_job weighted_train_job "${correct_train_job}" --job-name=q8b-sft-weighted --export=ALL,SFT_VARIANT=weighted_tau_0p20 "${SCRIPT_DIR}/03_train.sbatch"
submit_job unmasked_train_job "${weighted_train_job}" --job-name=q8b-sft-unmasked --export=ALL,SFT_VARIANT=unmasked "${SCRIPT_DIR}/03_train.sbatch"
submit_job correct_eval_job "${unmasked_train_job}" --job-name=q8b-eval-correct \
  --export=ALL,EVAL_STAGE=correct_only,EVAL_MODEL_DIR="${OUTPUT_ROOT}/training/correct_only/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}" \
  "${SCRIPT_DIR}/04_eval_math500.sbatch"
submit_job weighted_eval_job "${correct_eval_job}" --job-name=q8b-eval-weighted \
  --export=ALL,EVAL_STAGE=weighted_tau_0p20,EVAL_MODEL_DIR="${OUTPUT_ROOT}/training/weighted_tau_0p20/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}" \
  "${SCRIPT_DIR}/04_eval_math500.sbatch"
submit_job unmasked_eval_job "${weighted_eval_job}" --job-name=q8b-eval-unmasked \
  --export=ALL,EVAL_STAGE=unmasked,EVAL_MODEL_DIR="${OUTPUT_ROOT}/training/unmasked/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}",BUILD_FINAL_REPORT=1 \
  "${SCRIPT_DIR}/04_eval_math500.sbatch"

python3 "${SCRIPT_DIR}/write_submission_manifest.py" \
  --output "${REWIRE_MANIFEST}" \
  --contract-hash "${CONTRACT_HASH}" \
  "margin_score_and_weighted_build=${margin_job}" \
  "train_correct_only=${correct_train_job}" \
  "train_weighted_tau_0p20=${weighted_train_job}" \
  "train_unmasked=${unmasked_train_job}" \
  "eval_correct_only=${correct_eval_job}" \
  "eval_weighted_tau_0p20=${weighted_eval_job}" \
  "eval_unmasked_and_report=${unmasked_eval_job}"
submission_complete=1
trap - EXIT
echo "Resubmitted from atomic margin records; chain ends in ${unmasked_eval_job}."

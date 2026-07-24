#!/usr/bin/env bash
set -euo pipefail

SCRIPT_2K="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SCRIPT_40K="${SCRIPT_2K}/../qwen3_8b_openr1_math220k_masked_sft_40k"

source "${SCRIPT_40K}/env.sh"
CONTRACT_HASH_40K="${CONTRACT_HASH}"
OUTPUT_ROOT_40K="${OUTPUT_ROOT}"
MANIFEST_ROOT_40K="${MANIFEST_ROOT}"
source "${SCRIPT_2K}/env.sh"
CONTRACT_HASH_2K="${CONTRACT_HASH}"
OUTPUT_ROOT_2K="${OUTPUT_ROOT}"
MANIFEST_ROOT_2K="${MANIFEST_ROOT}"

cd "${SLIME_REPO_ROOT}"
python3 "${SCRIPT_40K}/validate_config.py" \
  --example-dir "${SCRIPT_40K}" --require-pushed-clean
python3 "${SCRIPT_2K}/validate_config.py" \
  --example-dir "${SCRIPT_2K}" --require-pushed-clean

SUBMISSION_MANIFEST="${MANIFEST_ROOT_2K}/completion_chain.json"
[[ ! -e "${SUBMISSION_MANIFEST}" ]] || {
  echo "Refusing to overwrite prior submission ${SUBMISSION_MANIFEST}" >&2
  exit 1
}
for root in "${OUTPUT_ROOT_40K}" "${OUTPUT_ROOT_2K}"; do
  if find "${root}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing submission into nonempty output root ${root}" >&2
    exit 1
  fi
done
mkdir -p \
  "${MANIFEST_ROOT_40K}" "${MANIFEST_ROOT_2K}" \
  "${OUTPUT_ROOT_40K}" "${OUTPUT_ROOT_2K}"

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
  local args=(
    --parsable
    --chdir="${SLIME_REPO_ROOT}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --qos="${QOS}"
  )
  if [[ -n "${tail_job}" ]]; then
    args+=(--dependency="afterok:${tail_job}")
  fi
  local raw
  raw="$(sbatch "${args[@]}" "$@")"
  tail_job="${raw%%;*}"
  [[ "${tail_job}" =~ ^[0-9]+$ ]] || {
    echo "Unexpected sbatch response ${raw}" >&2
    return 1
  }
  submitted_jobs+=("${tail_job}")
  manifest_jobs+=("${name}=${tail_job}")
  echo "SUBMITTED name=${name} job=${tail_job}"
}

# One new base evaluation is reused by both the 40K table and all 2K tables.
submit_serial base_8b_eval \
  --job-name=q8b-base-greedy32k \
  --export=ALL,EVAL_STAGE=base_8b,EVAL_MODEL_DIR="${STUDENT_HF_DIR}" \
  "${SCRIPT_2K}/04_eval_math500.sbatch"

# Complete the already-started 40K work from the immutable finished datasets.
submit_serial train_40k_correct_only \
  --job-name=q8b-40k-train-correct \
  --export=ALL,SFT_VARIANT=correct_only \
  "${SCRIPT_40K}/03_train.sbatch"
submit_serial eval_40k_correct_only \
  --job-name=q8b-40k-eval-correct \
  --export=ALL,EVAL_STAGE=correct_only,EVAL_MODEL_DIR="${OUTPUT_ROOT_40K}/training/correct_only/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}" \
  "${SCRIPT_40K}/04_eval_math500.sbatch"
submit_serial train_40k_weighted_tau_0p20 \
  --job-name=q8b-40k-train-weight \
  --export=ALL,SFT_VARIANT=weighted_tau_0p20 \
  "${SCRIPT_40K}/03_train.sbatch"
submit_serial eval_40k_weighted_tau_0p20 \
  --job-name=q8b-40k-eval-weight \
  --export=ALL,EVAL_STAGE=weighted_tau_0p20,EVAL_MODEL_DIR="${OUTPUT_ROOT_40K}/training/weighted_tau_0p20/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}" \
  "${SCRIPT_40K}/04_eval_math500.sbatch"
submit_serial train_40k_unmasked \
  --job-name=q8b-40k-train-unmask \
  --export=ALL,SFT_VARIANT=unmasked \
  "${SCRIPT_40K}/03_train.sbatch"
submit_serial eval_40k_unmasked_and_report \
  --job-name=q8b-40k-eval-unmask \
  --export=ALL,EVAL_STAGE=unmasked,EVAL_MODEL_DIR="${OUTPUT_ROOT_40K}/training/unmasked/weights/iter_0000199",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}",BUILD_FINAL_REPORT=1 \
  "${SCRIPT_40K}/04_eval_math500.sbatch"

# Freeze/retokenize the canonical Axolotl traces, then score the 1K wrong set once.
submit_serial prepare_2k_traces "${SCRIPT_2K}/01_prepare.sbatch"
submit_serial score_2k_margins_and_build_variants \
  "${SCRIPT_2K}/02_score_and_build.sbatch"

# Controls first, followed by the requested priority and remaining variants.
for variant in \
  correct_only \
  unmasked \
  random_mask_70 \
  inverse_tau_0p20 \
  inverse_tau_0p05 \
  random_mask_25 \
  random_mask_50 \
  random_mask_80 \
  random_mask_90 \
  margin_mask \
  prob_ratio_mask; do
  submit_serial "train_2k_${variant}" \
    --job-name="q8b-2k-t-${variant}" \
    --export=ALL,SFT_VARIANT="${variant}" \
    "${SCRIPT_2K}/03_train.sbatch"
  submit_serial "eval_2k_${variant}" \
    --job-name="q8b-2k-e-${variant}" \
    --export=ALL,EVAL_STAGE="${variant}",EVAL_MODEL_DIR="${OUTPUT_ROOT_2K}/training/${variant}/weights/iter_0000009",EVAL_COMPARE_TO="${BASE_EVAL_SUMMARY}" \
    "${SCRIPT_2K}/04_eval_math500.sbatch"
done
submit_serial report_2k_tables "${SCRIPT_2K}/05_report.sbatch"

python3 "${SCRIPT_2K}/write_submission_manifest.py" \
  --output "${SUBMISSION_MANIFEST}" \
  --contract-40k "${CONTRACT_HASH_40K}" \
  --contract-2k "${CONTRACT_HASH_2K}" \
  "${manifest_jobs[@]}"
submission_complete=1
trap - EXIT
echo "SUBMISSION_COMPLETE jobs=${#submitted_jobs[@]} first=${submitted_jobs[0]} last=${tail_job}"

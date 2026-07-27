#!/usr/bin/env bash
set -euo pipefail

SCRIPT_2K="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SCRIPT_40K="${SCRIPT_2K}/../qwen3_8b_openr1_math220k_masked_sft_40k"
HANDOFF_DIR="${SCRIPT_2K}/../qwen3_8b_openr1_math220k_masked_sft_eval_handoff"

source "${SCRIPT_40K}/env.sh"
CONTRACT_HASH_40K="${CONTRACT_HASH}"
EXPERIMENT_ROOT_40K="${EXPERIMENT_ROOT}"
OUTPUT_ROOT_40K="${OUTPUT_ROOT}"
source "${SCRIPT_2K}/env.sh"
CONTRACT_HASH_2K="${CONTRACT_HASH}"
EXPERIMENT_ROOT_2K="${EXPERIMENT_ROOT}"
OUTPUT_ROOT_2K="${OUTPUT_ROOT}"
MANIFEST_ROOT_2K="${MANIFEST_ROOT}"

cd "${SLIME_REPO_ROOT}"
python3 "${SCRIPT_40K}/validate_config.py" \
  --example-dir "${SCRIPT_40K}" --require-pushed-clean
python3 "${SCRIPT_2K}/validate_config.py" \
  --example-dir "${SCRIPT_2K}" --require-pushed-clean

SUBMISSION_MANIFEST="${MANIFEST_ROOT_2K}/training_only_chain.json"
[[ ! -e "${SUBMISSION_MANIFEST}" ]] || {
  echo "Refusing to overwrite prior submission ${SUBMISSION_MANIFEST}" >&2
  exit 1
}

CORRECT_40K_CHECKPOINT="${OUTPUT_ROOT_40K}/training/correct_only/weights/iter_0000199"
[[ -f "${CORRECT_40K_CHECKPOINT}/config.json" ]] || {
  echo "Missing completed 40K correct-only checkpoint ${CORRECT_40K_CHECKPOINT}" >&2
  exit 1
}
[[ "$(<"${OUTPUT_ROOT_40K}/training/correct_only/full_state/latest_checkpointed_iteration.txt")" == "199" ]] || {
  echo "40K correct-only full-state checkpoint is not at iteration 199." >&2
  exit 1
}

for variant in \
  correct_only unmasked random_mask_70 inverse_tau_0p20 inverse_tau_0p05 \
  random_mask_25 random_mask_50 random_mask_80 random_mask_90 \
  margin_mask prob_ratio_mask; do
  target="${OUTPUT_ROOT_2K}/training/${variant}"
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite 2K training output ${target}" >&2
    exit 1
  fi
done
for variant in weighted_tau_0p20 unmasked; do
  target="${OUTPUT_ROOT_40K}/training/${variant}"
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite 40K training output ${target}" >&2
    exit 1
  fi
done

# Record the two already-existing immutable checkpoints before submission.
python3 "${HANDOFF_DIR}/checkpoint_manifest.py" record \
  --checkpoint "${STUDENT_HF_DIR}" \
  --output "${EXPERIMENT_ROOT_2K}/handoff/checkpoints/base_8b.json" \
  --family shared \
  --variant base_8b \
  --contract-hash "${CONTRACT_HASH_2K}" \
  --repo-root "${SLIME_REPO_ROOT}" \
  --no-full-sft
python3 "${HANDOFF_DIR}/checkpoint_manifest.py" record \
  --checkpoint "${CORRECT_40K_CHECKPOINT}" \
  --output "${EXPERIMENT_ROOT_40K}/handoff/checkpoints/correct_only.json" \
  --family 40k \
  --variant correct_only \
  --final-iteration 199 \
  --contract-hash "${CONTRACT_HASH_40K}" \
  --repo-root "${SLIME_REPO_ROOT}" \
  --full-sft

mkdir -p "${MANIFEST_ROOT_2K}"
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

# The entire 2K suite is first. Preparation freezes one canonical unpaired
# trace set; every mixed variant reuses the exact same ordered 1K+1K rows.
submit_serial prepare_2k_traces "${SCRIPT_2K}/01_prepare.sbatch"
submit_serial score_2k_margins_and_build_variants \
  "${SCRIPT_2K}/02_score_and_build.sbatch"
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
done

# The completed 40K correct-only checkpoint is reused. Only its two remaining
# full-SFT variants are appended after all 2K checkpoints have been produced.
submit_serial train_40k_weighted_tau_0p20 \
  --job-name=q8b-40k-train-weight \
  --export=ALL,SFT_VARIANT=weighted_tau_0p20 \
  "${SCRIPT_40K}/03_train.sbatch"
submit_serial train_40k_unmasked \
  --job-name=q8b-40k-train-unmask \
  --export=ALL,SFT_VARIANT=unmasked \
  "${SCRIPT_40K}/03_train.sbatch"

python3 "${SCRIPT_2K}/write_submission_manifest.py" \
  --output "${SUBMISSION_MANIFEST}" \
  --contract-40k "${CONTRACT_HASH_40K}" \
  --contract-2k "${CONTRACT_HASH_2K}" \
  --reused-checkpoint "40k_correct_only=${CORRECT_40K_CHECKPOINT}" \
  "${manifest_jobs[@]}"
submission_complete=1
trap - EXIT
echo "SUBMISSION_COMPLETE jobs=${#submitted_jobs[@]} first=${submitted_jobs[0]} last=${tail_job}"

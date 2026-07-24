#!/bin/bash
#SBATCH --job-name=q8b-sft2k-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
base_sha256="$(<"${BASE_EVAL_SUMMARY_SHA256_FILE}")"
python3 "${SCRIPT_DIR}/report_results.py" \
  --summary "base_8b=${BASE_EVAL_SUMMARY}" \
  --summary "correct_only=${EVAL_OUTPUT_ROOT}/correct_only/summary.json" \
  --summary "unmasked=${EVAL_OUTPUT_ROOT}/unmasked/summary.json" \
  --summary "random_mask_25=${EVAL_OUTPUT_ROOT}/random_mask_25/summary.json" \
  --summary "random_mask_50=${EVAL_OUTPUT_ROOT}/random_mask_50/summary.json" \
  --summary "random_mask_70=${EVAL_OUTPUT_ROOT}/random_mask_70/summary.json" \
  --summary "random_mask_80=${EVAL_OUTPUT_ROOT}/random_mask_80/summary.json" \
  --summary "random_mask_90=${EVAL_OUTPUT_ROOT}/random_mask_90/summary.json" \
  --summary "inverse_tau_0p20=${EVAL_OUTPUT_ROOT}/inverse_tau_0p20/summary.json" \
  --summary "inverse_tau_0p05=${EVAL_OUTPUT_ROOT}/inverse_tau_0p05/summary.json" \
  --summary "margin_mask=${EVAL_OUTPUT_ROOT}/margin_mask/summary.json" \
  --summary "prob_ratio_mask=${EVAL_OUTPUT_ROOT}/prob_ratio_mask/summary.json" \
  --base-sha256 "${base_sha256}" \
  --output-json "${EXPERIMENT_ROOT}/RESULTS.json" \
  --output-md "${EXPERIMENT_ROOT}/RESULTS.md"
echo "REPORT_2K_COMPLETE results=${EXPERIMENT_ROOT}/RESULTS.json"

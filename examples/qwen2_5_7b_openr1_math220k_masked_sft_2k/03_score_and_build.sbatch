#!/bin/bash
#SBATCH --job-name=q25-7b-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k"
SCORE_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_40k"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${MARGIN_SHARD_DIR}/records" "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${SCRIPT_DIR}/container_exec.sh" python3 "${SCORE_DIR}/score_margins.py" \
      --selected-traces "${SELECTED_TRACES_JSONL}" \
      --output-dir "${MARGIN_SHARD_DIR}" \
      --model "${STUDENT_HF_DIR}" \
      --shard-index "${shard}" --num-shards 4 --batch-size 2 \
      --chat-template-kwargs '{}' \
      --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
      --expected-wrong "${MIXED_WRONG_ROWS}"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/build_variants.py" \
  --unmasked-data "${UNMASKED_JSONL}" --margin-dir "${MARGIN_SHARD_DIR}" \
  --stats-output "${VARIANT_STATS_JSON}" --mask-seed "${MASK_SEED}" \
  --output "random_mask_25=${RANDOM_MASK_25_JSONL}" \
  --output "random_mask_50=${RANDOM_MASK_50_JSONL}" \
  --output "random_mask_70=${RANDOM_MASK_70_JSONL}" \
  --output "random_mask_80=${RANDOM_MASK_80_JSONL}" \
  --output "random_mask_90=${RANDOM_MASK_90_JSONL}" \
  --output "inverse_tau_0p20=${INVERSE_TAU_0P20_JSONL}" \
  --output "inverse_tau_0p05=${INVERSE_TAU_0P05_JSONL}" \
  --output "margin_mask=${MARGIN_MASK_JSONL}" \
  --output "prob_ratio_mask=${PROB_RATIO_MASK_JSONL}"
mixed_hash="$(jq -r '.trace_sets.mixed_ordered_sha256' "${PREP_STATS_JSON}")"
for variant in random_mask_25 random_mask_50 random_mask_70 random_mask_80 random_mask_90 inverse_tau_0p20 inverse_tau_0p05 margin_mask prob_ratio_mask; do
  case "${variant}" in
    random_mask_25) data="${RANDOM_MASK_25_JSONL}" ;;
    random_mask_50) data="${RANDOM_MASK_50_JSONL}" ;;
    random_mask_70) data="${RANDOM_MASK_70_JSONL}" ;;
    random_mask_80) data="${RANDOM_MASK_80_JSONL}" ;;
    random_mask_90) data="${RANDOM_MASK_90_JSONL}" ;;
    inverse_tau_0p20) data="${INVERSE_TAU_0P20_JSONL}" ;;
    inverse_tau_0p05) data="${INVERSE_TAU_0P05_JSONL}" ;;
    margin_mask) data="${MARGIN_MASK_JSONL}" ;;
    prob_ratio_mask) data="${PROB_RATIO_MASK_JSONL}" ;;
  esac
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
    --data "${data}" --variant "${variant}" \
    --inventory "${TOKENIZER_INVENTORY_JSON}" --expected-trace-hash "${mixed_hash}"
done
echo "QWEN25_SCORE_BUILD_COMPLETE at=$(date -Is)"

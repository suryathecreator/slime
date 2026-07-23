#!/bin/bash
#SBATCH --job-name=q8b-masksft-margin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_40k"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${MARGIN_SHARD_DIR}/records" "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
[[ -s "${SELECTED_TRACES_JSONL}" ]] || { echo "Missing selected traces" >&2; exit 1; }

pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/score_margins.py" \
      --selected-traces "${SELECTED_TRACES_JSONL}" \
      --output-dir "${MARGIN_SHARD_DIR}" \
      --model "${STUDENT_HF_DIR}" \
      --shard-index "${shard}" \
      --num-shards 4 \
      --batch-size 2 \
      --max-sequence-length "${MAX_SEQUENCE_LENGTH}" \
      --expected-wrong "${MIXED_WRONG_ROWS}"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/build_weighted.py" \
  --unmasked-data "${UNMASKED_JSONL}" \
  --margin-dir "${MARGIN_SHARD_DIR}" \
  --merged-margins "${MARGIN_MERGED_JSONL}" \
  --weighted-output "${WEIGHTED_JSONL}" \
  --stats-output "${WEIGHTED_STATS_JSON}" \
  --tau "${WEIGHT_TAU}" \
  --expected-rows 40000 \
  --expected-correct 20000 \
  --expected-wrong 20000
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${WEIGHTED_JSONL}" --variant weighted_tau_0p20
echo "MARGIN_BUILD_JOB_COMPLETE at=$(date -Is)"

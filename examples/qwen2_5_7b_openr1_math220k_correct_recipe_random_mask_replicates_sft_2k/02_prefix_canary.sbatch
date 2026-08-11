#!/bin/bash
#SBATCH --job-name=q25rr-prefix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Prefix canary must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q25rr-prefix-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export SFT_VARIANT=random_mask_15_seed42
export SFT_PARQUET="$(variant_data_path "${SFT_VARIANT}")"
export SFT_NUM_ROLLOUT=5
export SFT_FINAL_ROLLOUT_ID=4
export SFT_MILESTONE_ROLLOUT_IDS=4
export SFT_SAVE_INTERVAL=5
export PREFIX_ROOT="${OUTPUT_ROOT}/prefix_canary/${SFT_VARIANT}"
export SFT_SAVE_DIR="${PREFIX_ROOT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${PREFIX_ROOT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000004"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000004"
export SFT_DETAILS_DIR="${PREFIX_ROOT}/details"
export CHECKPOINT_REPORT_DIR="${PREFIX_ROOT}/checkpoint_reports"
export SFT_WANDB_GROUP="qwen2.5-7b-random-mask-replicate-prefix-${SFT_VARIANT}"
for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${PREFIX_ROOT}" "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
mixed_hash="$(jq -r '.ordered_mixed_trace_ids_sha256' "${PREP_STATS_JSON}")"
"${CONTAINER_EXEC}" python3 "${SCRIPT_DIR}/validate_random_mask.py" \
  --contract "${CONTRACT_FILE}" --source-unmasked "${SOURCE_UNMASKED_JSONL}" \
  --data "${SFT_PARQUET}" --variant "${SFT_VARIANT}" \
  --expected-trace-hash "${mixed_hash}" --max-sequence-length "${SFT_SEQ_LENGTH}"
test -f "${SCHEDULE_AUDIT_JSON}"
bash "${SHARED_RUNNER}"
test "$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")" = 4
test -f "${SFT_FINAL_HF_DIR}/config.json"
echo "REPLICATE_PREFIX_COMPLETE variant=${SFT_VARIANT} at=$(date -Is)"

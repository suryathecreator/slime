#!/bin/bash
#SBATCH --job-name=qco-prefix-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
source "${SCRIPT_DIR}/env.sh"
: "${RUN_KEY:?Set RUN_KEY}"
: "${PREFIX_CANARY_ATTEMPT:?Set PREFIX_CANARY_ATTEMPT}"
: "${SLURM_JOB_ID:?Training-prefix canary must run under Slurm}"
if [[ ! "${PREFIX_CANARY_ATTEMPT}" =~ ^[a-z0-9_]+$ ]]; then
  echo "PREFIX_CANARY_ATTEMPT must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi
case "${RUN_KEY}" in
  qwen2_5_3b_8k|qwen2_5_3b_16k) ;;
  *) echo "Training-prefix canary only supports Qwen2.5-3B runs" >&2; exit 2 ;;
esac
resolve_run "${RUN_KEY}"
export TMPDIR="/tmp/${USER:-suryadv}/qco-prefix-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export SFT_SKIP_LOG_REDIRECT=1
export SFT_NUM_ROLLOUT=5
export SFT_FINAL_ROLLOUT_ID=4
export SFT_MILESTONE_ROLLOUT_IDS=4
export SFT_SAVE_INTERVAL=5
export PREFIX_ROOT="${OUTPUT_ROOT}/training_prefix_canaries/${RUN_KEY}/${PREFIX_CANARY_ATTEMPT}"
export SFT_SAVE_DIR="${PREFIX_ROOT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${PREFIX_ROOT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000004"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000004"
export SFT_DETAILS_DIR="${PREFIX_ROOT}/details"
export CHECKPOINT_REPORT_DIR="${PREFIX_ROOT}/checkpoint_reports"
export SFT_WANDB_GROUP="${RUN_KEY}-prefix-${PREFIX_CANARY_ATTEMPT}"
for target in \
  "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" \
  "${CHECKPOINT_REPORT_DIR}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${PREFIX_ROOT}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${SFT_PARQUET}" --run-key "${RUN_KEY}" --stats "${PREP_STATS_JSON}"
test -f "${SCHEDULE_AUDIT_DIR}/${RUN_KEY}.json"
echo "TRAINING_PREFIX_CANARY run=${RUN_KEY} attempt=${PREFIX_CANARY_ATTEMPT} rollouts=5"
bash "${SHARED_RUNNER}"
test "$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")" = 4
test -f "${SFT_FINAL_HF_DIR}/config.json"
echo "TRAINING_PREFIX_CANARY_COMPLETE run=${RUN_KEY} attempt=${PREFIX_CANARY_ATTEMPT} at=$(date -Is)"

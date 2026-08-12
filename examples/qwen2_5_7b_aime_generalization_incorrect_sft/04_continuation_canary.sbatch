#!/bin/bash
#SBATCH --job-name=q25aig-cont-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
HANDOFF_TOOL="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
source "${SCRIPT_DIR}/env.sh"
: "${AIME_EXPECTED_GIT_COMMIT:?Missing submit-time Git commit pin}"
python3 -B "${SCRIPT_DIR}/runtime_repo_gate.py" \
  --repo-root "${SLIME_REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
source "${SCRIPT_DIR}/canary_helpers.sh"
: "${SLURM_JOB_ID:?Continuation canary must run under Slurm}"
configure_sft_variant continue_random_mask_90
export SFT_NUM_ROLLOUT=5
export SFT_FINAL_ROLLOUT_ID=4
export SFT_MILESTONE_ROLLOUT_IDS=4
export SFT_SAVE_INTERVAL=5
export SFT_LR_DECAY_ITERS=5
export TMPDIR="/tmp/${USER:-suryadv}/q25aig-cont-canary-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export PREFIX_ROOT="${OUTPUT_ROOT}/canaries/continuation_from_stage1"
export SFT_SAVE_DIR="${PREFIX_ROOT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${PREFIX_ROOT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000004"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000004"
export SFT_DETAILS_DIR="${PREFIX_ROOT}/details"
export CHECKPOINT_REPORT_DIR="${PREFIX_ROOT}/checkpoint_reports"
export CANARY_EVIDENCE_ROOT="${PREFIX_ROOT}/evidence"
export SFT_WANDB_GROUP="qwen2.5-7b-aime-continuation-canary"
for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${PREFIX_ROOT}" "${CANARY_EVIDENCE_ROOT}" "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
test "${SFT_INITIAL_HF_DIR}" = "${STAGE1_FINAL_HF_DIR}"
test -f "${STAGE1_FINAL_HF_DIR}/config.json"
test -f "${STAGE1_MANIFEST}"
python3 "${SCRIPT_DIR}/parent_checkpoint_gate.py" record \
  --tool "${HANDOFF_TOOL}" --manifest "${STAGE1_MANIFEST}" \
  --checkpoint "${STAGE1_FINAL_HF_DIR}" --contract-hash "${CONTRACT_HASH}" \
  --marker "${STAGE1_PARENT_GATE}"
run_token_audit "${SFT_PARQUET}" "${STAGE1_FINAL_HF_DIR}" \
  "${CANARY_EVIDENCE_ROOT}/before_runtime_token_audit.json" before_continuation
run_copy_diagnostic_nonblocking "${STAGE1_FINAL_HF_DIR}" \
  "${CANARY_EVIDENCE_ROOT}/before_copy_diagnostic.json" before_continuation
bash "${SHARED_RUNNER}"
test "$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")" = 4
test -f "${SFT_FINAL_HF_DIR}/config.json"
run_token_audit "${SFT_PARQUET}" "${STAGE1_FINAL_HF_DIR}" \
  "${CANARY_EVIDENCE_ROOT}/after_runtime_token_audit.json" after_continuation \
  "${SFT_DETAILS_DIR}"
run_copy_diagnostic_nonblocking "${SFT_FINAL_HF_DIR}" \
  "${CANARY_EVIDENCE_ROOT}/after_copy_diagnostic.json" after_continuation
echo "AIME_CONTINUATION_CANARY_COMPLETE parent=${STAGE1_FINAL_HF_DIR} at=$(date -Is)"

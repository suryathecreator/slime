#!/bin/bash
#SBATCH --job-name=q25aig-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_aime_generalization_incorrect_sft"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
HANDOFF_TOOL="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
source "${SFT_EXPERIMENT_ENV_FILE:-${SCRIPT_DIR}/env.sh}"
source "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/checkpoint_utils.sh"
: "${AIME_EXPECTED_GIT_COMMIT:?Missing submit-time Git commit pin}"
python3 -B "${SCRIPT_DIR}/runtime_repo_gate.py" \
  --repo-root "${SLIME_REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
: "${SFT_VARIANT:?Set SFT_VARIANT}"
: "${SLURM_JOB_ID:?Training must run under Slurm}"
configure_sft_variant "${SFT_VARIANT}"
export TMPDIR="/tmp/${USER:-suryadv}/q25aig-train-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export TRAINING_ROOT="${OUTPUT_ROOT}/training/${SFT_VARIANT}"
export SFT_SAVE_DIR="${TRAINING_ROOT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${TRAINING_ROOT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
printf -v final_tag "iter_%07d" "${SFT_FINAL_ROLLOUT_ID}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/${final_tag}"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/${final_tag}"
export SFT_DETAILS_DIR="${TRAINING_ROOT}/details"
export CHECKPOINT_REPORT_DIR="${TRAINING_ROOT}/checkpoint_reports"
export SFT_WANDB_GROUP="${SFT_WANDB_GROUP_PREFIX:-qwen2.5-7b-aime-generalization}-${SFT_VARIANT}"
export TRAINED_MANIFEST="${HANDOFF_ROOT}/checkpoints/${SFT_VARIANT}.json"
HF_GATE=(python3 "${SCRIPT_DIR}/hf_checkpoint_gate.py")
manifest_already_present=0
[[ ! -e "${TRAINED_MANIFEST}" ]] || manifest_already_present=1
resume_iteration=""
if [[ -f "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  tracked_iteration="$(<"${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")"
  tracker_is_malformed=0
  if [[ "${tracked_iteration}" =~ ^[0-9]+$ ]]; then
    (( tracked_iteration <= SFT_FINAL_ROLLOUT_ID )) || {
      echo "Local checkpoint is beyond contracted final iteration: ${tracked_iteration}" >&2
      exit 1
    }
    resume_upper_bound="${tracked_iteration}"
  else
    tracker_is_malformed=1
    resume_upper_bound="${SFT_FINAL_ROLLOUT_ID}"
    echo "MALFORMED_TRACKER_RECOVERY variant=${SFT_VARIANT} value=${tracked_iteration@Q}"
  fi
  if resume_iteration="$(checkpoint_highest_durable_iteration "${SFT_SAVE_DIR}" "${resume_upper_bound}")"; then
    resume_tag="$(printf 'iter_%07d' "${resume_iteration}")"
    if [[ "${tracker_is_malformed}" == "1" || "${resume_iteration}" != "${tracked_iteration}" ]]; then
      tracker_tmp="${SFT_SAVE_DIR}/.latest_checkpointed_iteration.rollback.$$"
      printf "%s\n" "${resume_iteration}" >"${tracker_tmp}"
      mv -- "${tracker_tmp}" "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt"
      echo "TORN_TRACKER_ROLLED_BACK variant=${SFT_VARIANT} tracked=${tracked_iteration} durable=${resume_iteration}"
    fi
    echo "OWN_VARIANT_RESUME_ACCEPTED variant=${SFT_VARIANT} iteration=${resume_iteration}"
  else
    [[ "${manifest_already_present}" == "0" ]] || {
      echo "Final manifest exists but no durable local checkpoint is recoverable: ${TRAINED_MANIFEST}" >&2
      exit 1
    }
    restart_count="${SLURM_RESTART_COUNT:-0}"
    restart_quarantine="${TRAINING_ROOT}/restart_from_zero_job_${SLURM_JOB_ID}_restart_${restart_count}"
    [[ ! -e "${restart_quarantine}" ]] || {
      echo "Restart-from-zero quarantine already exists: ${restart_quarantine}" >&2
      exit 1
    }
    mkdir -p "${restart_quarantine}"
    for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
      if [[ -e "${target}" ]]; then
        mv -- "${target}" "${restart_quarantine}/$(basename "${target}")"
      fi
    done
    resume_iteration=""
    echo "NO_DURABLE_CHECKPOINT_RESTART_FROM_ZERO variant=${SFT_VARIANT} preserved=${restart_quarantine}"
  fi
else
  for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
    [[ ! -e "${target}" ]] || {
      echo "Partial/foreign output exists without a local resume tracker: ${target}" >&2
      exit 1
    }
  done
fi
if [[ "${manifest_already_present}" == "1" && "${resume_iteration:-}" != "${SFT_FINAL_ROLLOUT_ID}" ]]; then
  echo "Final manifest exists without the exact final durable tracker: ${TRAINED_MANIFEST}" >&2
  exit 1
fi
if [[ -n "${resume_iteration}" && "${resume_iteration}" != "${SFT_FINAL_ROLLOUT_ID}" ]]; then
  restart_count="${SLURM_RESTART_COUNT:-0}"
  resume_quarantine="${TRAINING_ROOT}/resume_quarantine_job_${SLURM_JOB_ID}_restart_${restart_count}"
  [[ ! -e "${resume_quarantine}" ]] || {
    echo "Resume quarantine already exists: ${resume_quarantine}" >&2
    exit 1
  }
  mkdir -p "${resume_quarantine}/full_state/rollout" "${resume_quarantine}/weights"
  while IFS= read -r directory; do
    [[ "$(basename "${directory}")" == "${resume_tag}" ]] || \
      mv -- "${directory}" "${resume_quarantine}/full_state/"
  done < <(find "${SFT_SAVE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print)
  while IFS= read -r directory; do
    [[ "$(basename "${directory}")" == "${resume_tag}" ]] || \
      mv -- "${directory}" "${resume_quarantine}/weights/"
  done < <(find "${SFT_HF_SNAPSHOT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print 2>/dev/null || true)
  while IFS= read -r state_file; do
    [[ "$(basename "${state_file}")" == "global_dataset_state_dict_${resume_iteration}.pt" ]] || \
      mv -- "${state_file}" "${resume_quarantine}/full_state/rollout/"
  done < <(find "${SFT_SAVE_DIR}/rollout" -mindepth 1 -maxdepth 1 -type f -name 'global_dataset_state_dict_*.pt' -print 2>/dev/null || true)
  for target in "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
    if [[ -e "${target}" ]]; then
      mv -- "${target}" "${resume_quarantine}/"
    fi
  done
  echo "RESUME_FORWARD_STATE_QUARANTINED variant=${SFT_VARIANT} path=${resume_quarantine}"
fi
mkdir -p "${TRAINING_ROOT}" "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}" "${HANDOFF_ROOT}/checkpoints"
exec >>"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
runner_pid=""
hf_pruner_pid=""
prune_hf_snapshots() {
  local final_mode="${1:-0}"
  local keep=""
  if [[ "${final_mode}" == "1" ]]; then
    keep="${SFT_FINAL_HF_DIR}"
    "${HF_GATE[@]}" --checkpoint "${keep}" >/dev/null
  elif [[ -f "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    local latest
    latest="$(<"${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")"
    local candidate
    candidate="${SFT_HF_SNAPSHOT_DIR}/$(printf 'iter_%07d' "${latest}")"
    "${HF_GATE[@]}" --checkpoint "${candidate}" >/dev/null 2>&1 || return 0
    keep="${candidate}"
  else
    return 0
  fi
  local directory
  while IFS= read -r directory; do
    if [[ "${directory}" != "${keep}" ]]; then
      echo "HF_SNAPSHOT_PRUNE_REMOVE variant=${SFT_VARIANT} path=${directory}"
      rm -rf -- "${directory}"
    fi
  done < <(find "${SFT_HF_SNAPSHOT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print)
}
start_hf_pruner() {
  (
    set +e
    while true; do
      prune_hf_snapshots 0
      sleep 60
    done
  ) &
  hf_pruner_pid=$!
}
stop_hf_pruner() {
  if [[ -n "${hf_pruner_pid}" ]]; then
    kill "${hf_pruner_pid}" 2>/dev/null || true
    wait "${hf_pruner_pid}" 2>/dev/null || true
    hf_pruner_pid=""
  fi
}
handle_preemption() {
  echo "PREEMPTION_SIGNAL variant=${SFT_VARIANT} at=$(date -Is)" >&2
  if [[ -n "${runner_pid}" ]]; then
    kill -TERM "${runner_pid}" 2>/dev/null || true
    wait "${runner_pid}" 2>/dev/null || true
  fi
  stop_hf_pruner
  if [[ ! -f "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    local quarantine_root="${TRAINING_ROOT}/interrupted_before_first_checkpoint_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}"
    mkdir -p "${quarantine_root}"
    local target
    for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"; do
      if [[ -e "${target}" ]]; then
        mv -- "${target}" "${quarantine_root}/$(basename "${target}")"
      fi
    done
    echo "NO_DURABLE_CHECKPOINT exact restart after requeue; preserved=${quarantine_root}" >&2
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap handle_preemption USR1
trap stop_hf_pruner EXIT
recover_final_hf_from_full_state() {
  local recovery="${SFT_HF_SNAPSHOT_DIR}/.recovery_${SFT_VARIANT}_${SLURM_JOB_ID}"
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  local quarantine="${SFT_HF_SNAPSHOT_DIR}/.incomplete_$(basename "${SFT_FINAL_HF_DIR}")_${SLURM_JOB_ID}_restart_${restart_count}"
  local failed_recovery="${SFT_HF_SNAPSHOT_DIR}/.failed_recovery_${SFT_VARIANT}_${SLURM_JOB_ID}_restart_${restart_count}"
  if "${HF_GATE[@]}" --checkpoint "${SFT_FINAL_HF_DIR}" >/dev/null 2>&1; then
    return 0
  fi
  if [[ -e "${recovery}" ]] && \
     ! "${HF_GATE[@]}" --checkpoint "${recovery}" >/dev/null 2>&1; then
    [[ ! -e "${failed_recovery}" ]] || {
      echo "Failed-recovery quarantine already exists: ${failed_recovery}" >&2
      return 1
    }
    mv -- "${recovery}" "${failed_recovery}"
  fi
  if [[ ! -e "${recovery}" ]]; then
    echo "RECOVER_FINAL_HF_FROM_FULL_STATE variant=${SFT_VARIANT} source=${SFT_FINAL_FULL_CKPT_DIR}"
    "${AIME_GENERALIZATION_DIR}/../qwen3_8b_opd_tillicum/container_exec.sh" \
      bash -lc 'cd "${SLIME_REPO_ROOT}"; exec python3 tools/convert_torch_dist_to_hf_bridge.py --input-dir "$1" --output-dir "$2" --origin-hf-dir "$3"' \
      bash "${SFT_FINAL_FULL_CKPT_DIR}" "${recovery}" "${STUDENT_HF_DIR}"
  else
    echo "REUSE_COMPLETE_RECOVERED_HF variant=${SFT_VARIANT} path=${recovery}"
  fi
  "${HF_GATE[@]}" --checkpoint "${recovery}"
  if [[ -e "${SFT_FINAL_HF_DIR}" ]]; then
    [[ ! -e "${quarantine}" ]] || {
      echo "Final-HF quarantine already exists: ${quarantine}" >&2
      return 1
    }
    mv -- "${SFT_FINAL_HF_DIR}" "${quarantine}"
  fi
  mv -- "${recovery}" "${SFT_FINAL_HF_DIR}"
  "${HF_GATE[@]}" --checkpoint "${SFT_FINAL_HF_DIR}"
}
test -f "${SFT_PARQUET}"
test -f "${SCHEDULE_AUDIT_JSON}"
if [[ -n "${SFT_INITIAL_HF_DIR:-}" ]]; then
  test -f "${STAGE1_FINAL_HF_DIR}/config.json"
  test -f "${STAGE1_MANIFEST}"
  test "${SFT_INITIAL_HF_DIR}" = "${STAGE1_FINAL_HF_DIR}"
  python3 "${SCRIPT_DIR}/parent_checkpoint_gate.py" check \
    --tool "${HANDOFF_TOOL}" --manifest "${STAGE1_MANIFEST}" \
    --checkpoint "${STAGE1_FINAL_HF_DIR}" --contract-hash "${CONTRACT_HASH}" \
    --marker "${STAGE1_PARENT_GATE}"
fi
echo "TRAIN_SFT_VARIANT variant=${SFT_VARIANT} updates=${SFT_NUM_ROLLOUT} initial_hf=${SFT_INITIAL_HF_DIR:-base_torch_dist}"
if [[ "${resume_iteration:-}" == "${SFT_FINAL_ROLLOUT_ID}" ]] && \
   ! "${HF_GATE[@]}" --checkpoint "${SFT_FINAL_HF_DIR}" >/dev/null 2>&1; then
  recover_final_hf_from_full_state
fi
if [[ "${resume_iteration:-}" != "${SFT_FINAL_ROLLOUT_ID}" ]]; then
  start_hf_pruner
  bash "${SHARED_RUNNER}" &
  runner_pid=$!
  wait "${runner_pid}"
  runner_pid=""
  stop_hf_pruner
else
  echo "FINAL_CHECKPOINT_ALREADY_DURABLE variant=${SFT_VARIANT}; recording manifest only"
fi
test "$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")" = "${SFT_FINAL_ROLLOUT_ID}"
"${HF_GATE[@]}" --checkpoint "${SFT_FINAL_HF_DIR}"
prune_hf_snapshots 1
manifest_args=(
  record --checkpoint "${SFT_FINAL_HF_DIR}" --output "${TRAINED_MANIFEST}"
  --family "${SFT_MANIFEST_FAMILY:-aime_generalization}" --variant "${SFT_VARIANT}"
  --final-iteration "${SFT_FINAL_ROLLOUT_ID}" --contract-hash "${CONTRACT_HASH}"
  --repo-root "${SLIME_REPO_ROOT}" --full-sft
)
if [[ -n "${SFT_INITIAL_HF_DIR:-}" ]]; then
  parent_identity="$(jq -r '.checkpoint_manifest_sha256' "${STAGE1_MANIFEST}")"
  [[ "${parent_identity}" =~ ^[0-9a-f]{64}$ ]]
  manifest_args+=(--parent-checkpoint-manifest-sha256 "${parent_identity}" --fresh-optimizer)
fi
python3 "${HANDOFF_TOOL}" "${manifest_args[@]}"
echo "TRAIN_SFT_VARIANT_COMPLETE variant=${SFT_VARIANT} at=$(date -Is)"

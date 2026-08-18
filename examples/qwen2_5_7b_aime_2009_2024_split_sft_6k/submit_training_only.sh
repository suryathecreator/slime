#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --dry-run or --submit}"
[[ "${MODE}" == "--dry-run" || "${MODE}" == "--submit" ]] || {
  echo "Usage: $0 --dry-run|--submit" >&2
  exit 2
}
[[ $# -eq 1 ]] || { echo "Usage: $0 --dry-run|--submit" >&2; exit 2; }

PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${PACKAGE}/env.sh"
cd "${SLIME_REPO_ROOT}"
submit_commit="$(git rev-parse HEAD)"

VALIDATE=(
  python3 "${PACKAGE}/validate_config.py"
  --repo-root "${SLIME_REPO_ROOT}" --contract "${CONTRACT_FILE}"
  --require-source-artifacts
)
if [[ "${MODE}" == "--submit" ]]; then
  VALIDATE+=(--require-pushed-clean)
fi
"${VALIDATE[@]}"

DRY_COMMON=(
  --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}"
  --chdir="${SLIME_REPO_ROOT}"
)
test_script() {
  local script="${1:?script required}"
  local export_tail="${2:-}"
  sbatch --test-only "${DRY_COMMON[@]}" \
    --export="ALL,AIME_EXPECTED_GIT_COMMIT=${submit_commit}${export_tail}" \
    "${PACKAGE}/${script}" >/dev/null
}

test_script 01_prepare_data.sbatch
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  test_script 02_canary.sbatch ",SFT_CANARY_VARIANT=${variant}"
done
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  test_script 03_train.sbatch ",SFT_VARIANT=${variant}"
done
test_script 05_finalize.sbatch
echo "AIME_SPLIT_6K_SBATCH_TEST_ONLY_OK jobs=10 maximum_concurrent_h200=4"

temporary_root="$(mktemp -d "/tmp/${USER:-suryadv}.aime-split-6k-submit.XXXXXX")"
trap 'rm -rf -- "${temporary_root}"' EXIT
planned_tsv="${temporary_root}/planned_jobs.tsv"
printf 'name\tjob_id\tdependency\tscript\tvariant\n' >"${planned_tsv}"
previous=""
append_plan() {
  local name="$1" script="$2" variant="${3:-}"
  local id="DRYRUN_$(printf '%02d' "$(wc -l <"${planned_tsv}")")"
  local dependency=""
  [[ -z "${previous}" ]] || dependency="afterok:${previous}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${id}" "${dependency}" "${script}" "${variant}" >>"${planned_tsv}"
  previous="${id}"
}
append_plan prepare_data 01_prepare_data.sbatch
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  append_plan "canary_${variant}" 02_canary.sbatch "${variant}"
done
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  append_plan "train_${variant}" 03_train.sbatch "${variant}"
done
append_plan finalize_handoff 05_finalize.sbatch

if [[ "${MODE}" == "--dry-run" ]]; then
  dry_manifest="/tmp/${USER:-suryadv}.aime-split-6k-${CONTRACT_HASH}-dry-run.json"
  [[ ! -e "${dry_manifest}" ]] || dry_manifest="${temporary_root}/dry_run_manifest.json"
  python3 "${PACKAGE}/write_submission_manifest.py" \
    --output "${dry_manifest}" --jobs-tsv "${planned_tsv}" \
    --contract-hash "${CONTRACT_HASH}" --expected-commit "${submit_commit}" \
    --repo-root "${SLIME_REPO_ROOT}" --execution-worktree "${SLIME_REPO_ROOT}" \
    --mode dry_run
  echo "AIME_SPLIT_6K_TRAINING_DRY_RUN_OK manifest=${dry_manifest} jobs=10"
  exit 0
fi

[[ ! -e "${EXPERIMENT_ROOT}" ]] || {
  echo "Experiment root already exists; refusing a second launch: ${EXPERIMENT_ROOT}" >&2
  exit 1
}
mkdir -p "${MANIFEST_ROOT}" "${SLURM_LOG_DIR}"
export EXECUTION_WORKTREE="${EXPERIMENT_ROOT}/execution_repo"
git worktree add --detach "${EXECUTION_WORKTREE}" "${submit_commit}"
test "$(git -C "${EXECUTION_WORKTREE}" rev-parse HEAD)" = "${submit_commit}"
test -z "$(git -C "${EXECUTION_WORKTREE}" status --porcelain --untracked-files=all)"
EXEC_PACKAGE="${EXECUTION_WORKTREE}/examples/qwen2_5_7b_aime_2009_2024_split_sft_6k"
cd "${EXECUTION_WORKTREE}"
COMMON=(
  --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}"
  --chdir="${EXECUTION_WORKTREE}"
)

jobs_tsv="${MANIFEST_ROOT}/submitted_jobs.tsv"
printf 'name\tjob_id\tdependency\tscript\tvariant\n' >"${jobs_tsv}"
submitted_ids=()
submission_complete=0
cancel_partial_chain() {
  local status=$?
  if [[ "${submission_complete}" -eq 0 && "${#submitted_ids[@]}" -gt 0 ]]; then
    echo "Submission failed; cancelling new jobs: ${submitted_ids[*]}" >&2
    scancel "${submitted_ids[@]}" || true
  fi
  echo "Immutable execution worktree preserved for audit: ${EXECUTION_WORKTREE}" >&2
  exit "${status}"
}
trap cancel_partial_chain ERR INT TERM

previous=""
submit_job() {
  local name="$1" script="$2" variant="${3:-}" kind="${4:-}"
  local dependency=""
  [[ -z "${previous}" ]] || dependency="afterok:${previous}"
  local export_tail=""
  if [[ "${kind}" == "canary" ]]; then
    export_tail=",SFT_CANARY_VARIANT=${variant}"
  elif [[ "${kind}" == "train" ]]; then
    export_tail=",SFT_VARIANT=${variant}"
  fi
  local -a args=(
    sbatch --parsable "${COMMON[@]}"
    --output="${SLURM_LOG_DIR}/%x_%j.bootstrap.out"
  )
  [[ -z "${dependency}" ]] || args+=(--dependency="${dependency}")
  args+=(--export="ALL,AIME_EXPECTED_GIT_COMMIT=${submit_commit}${export_tail}")
  local raw job_id
  raw="$("${args[@]}" "${EXEC_PACKAGE}/${script}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "sbatch returned a nonnumeric job ID: ${raw}" >&2
    return 1
  }
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${job_id}" "${dependency}" "${script}" "${variant}" >>"${jobs_tsv}"
  submitted_ids+=("${job_id}")
  previous="${job_id}"
  python3 "${EXEC_PACKAGE}/write_submission_manifest.py" \
    --output "${MANIFEST_ROOT}/submission_in_progress.json" \
    --jobs-tsv "${jobs_tsv}" --contract-hash "${CONTRACT_HASH}" \
    --repo-root "${EXECUTION_WORKTREE}" --execution-worktree "${EXECUTION_WORKTREE}" \
    --expected-commit "${submit_commit}" --mode in_progress --replace >/dev/null
  echo "AIME_SPLIT_6K_JOB_SUBMITTED name=${name} id=${job_id} dependency=${dependency:-none} variant=${variant:-none}"
}

submit_job prepare_data 01_prepare_data.sbatch
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  submit_job "canary_${variant}" 02_canary.sbatch "${variant}" canary
done
for variant in "${ALL_TRAINED_VARIANTS[@]}"; do
  submit_job "train_${variant}" 03_train.sbatch "${variant}" train
done
submit_job finalize_handoff 05_finalize.sbatch

python3 "${EXEC_PACKAGE}/write_submission_manifest.py" \
  --output "${MANIFEST_ROOT}/training_chain.json" --jobs-tsv "${jobs_tsv}" \
  --contract-hash "${CONTRACT_HASH}" --expected-commit "${submit_commit}" \
  --repo-root "${EXECUTION_WORKTREE}" --execution-worktree "${EXECUTION_WORKTREE}" \
  --mode submitted
python3 "${EXEC_PACKAGE}/write_submission_manifest.py" \
  --output "${PACKAGE}/SUBMISSION.json" --jobs-tsv "${jobs_tsv}" \
  --contract-hash "${CONTRACT_HASH}" --expected-commit "${submit_commit}" \
  --repo-root "${EXECUTION_WORKTREE}" --execution-worktree "${EXECUTION_WORKTREE}" \
  --mode submitted
submission_complete=1
trap - ERR INT TERM
echo "AIME_SPLIT_6K_TRAINING_CHAIN_SUBMITTED first=${submitted_ids[0]} final=${submitted_ids[-1]} jobs=${#submitted_ids[@]} contract=${CONTRACT_HASH} execution_commit=${submit_commit}"

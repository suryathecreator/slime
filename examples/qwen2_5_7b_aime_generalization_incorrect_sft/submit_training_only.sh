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

temporary_root="$(mktemp -d "/tmp/${USER:-suryadv}.aime-generalization-submit.XXXXXX")"
trap 'rm -rf -- "${temporary_root}"' EXIT
gate="${temporary_root}/upstream_gate.json"
python3 "${PACKAGE}/prerequisite_gate.py" \
  --job-id "${UPSTREAM_AFTEROK_JOB_ID}" --output "${gate}"
if [[ "$(jq -r '.controller_known' "${gate}")" == "true" ]]; then
  first_dependency="afterok:${UPSTREAM_AFTEROK_JOB_ID}"
else
  first_dependency=""
fi

COMMON=(
  --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}"
  --chdir="${SLIME_REPO_ROOT}"
)
test_script() {
  local script="${1:?script required}"
  local variant="${2:-}"
  local -a args=(
    sbatch --test-only "${COMMON[@]}"
    --export="ALL,AIME_EXPECTED_GIT_COMMIT=${submit_commit}${variant:+,SFT_VARIANT=${variant}}"
  )
  "${args[@]}" "${script}" >/dev/null
}

test_script "${PACKAGE}/01_prepare_data.sbatch"
test_script "${PACKAGE}/02_stage1_canary.sbatch"
test_script "${PACKAGE}/03_train.sbatch" "${STAGE1_VARIANT}"
test_script "${PACKAGE}/04_continuation_canary.sbatch"
for variant in "${CHILD_VARIANTS[@]}"; do
  test_script "${PACKAGE}/03_train.sbatch" "${variant}"
done
test_script "${PACKAGE}/05_finalize.sbatch"
echo "AIME_SBATCH_TEST_ONLY_OK jobs=16"

planned_tsv="${temporary_root}/planned_jobs.tsv"
printf 'name\tjob_id\tdependency\tscript\tvariant\n' >"${planned_tsv}"
append_plan() {
  local name="$1" id="$2" dependency="$3" script="$4" variant="${5:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${id}" "${dependency}" "${script}" "${variant}" >>"${planned_tsv}"
}
previous=""
plan_job() {
  local name="$1" script="$2" variant="${3:-}"
  local number=$(( $(wc -l <"${planned_tsv}") ))
  local id="DRYRUN_$(printf '%02d' "${number}")"
  local dependency=""
  if [[ -z "${previous}" ]]; then
    dependency="${first_dependency}"
  else
    dependency="afterok:${previous}"
  fi
  append_plan "${name}" "${id}" "${dependency}" "${script}" "${variant}"
  previous="${id}"
}
plan_job prepare_data "01_prepare_data.sbatch"
plan_job stage1_canary "02_stage1_canary.sbatch"
plan_job train_aime_correct_3000 "03_train.sbatch" "${STAGE1_VARIANT}"
plan_job continuation_canary "04_continuation_canary.sbatch"
for variant in "${CHILD_VARIANTS[@]}"; do
  plan_job "train_${variant}" "03_train.sbatch" "${variant}"
done
plan_job finalize_handoff "05_finalize.sbatch"

if [[ "${MODE}" == "--dry-run" ]]; then
  dry_manifest="/tmp/${USER:-suryadv}.aime-generalization-${CONTRACT_HASH}-dry-run.json"
  [[ ! -e "${dry_manifest}" ]] || {
    dry_manifest="${temporary_root}/dry_run_manifest.json"
  }
  python3 "${PACKAGE}/write_submission_manifest.py" \
    --output "${dry_manifest}" --jobs-tsv "${planned_tsv}" \
    --upstream-gate "${gate}" --contract-hash "${CONTRACT_HASH}" \
    --expected-commit "${submit_commit}" \
    --repo-root "${SLIME_REPO_ROOT}" --mode dry_run
  echo "AIME_TRAINING_DRY_RUN_OK manifest=${dry_manifest} jobs=16"
  exit 0
fi

[[ ! -e "${EXPERIMENT_ROOT}" ]] || {
  echo "Experiment root already exists; refusing a second launch: ${EXPERIMENT_ROOT}" >&2
  exit 1
}
mkdir -p "${MANIFEST_ROOT}" "${SLURM_LOG_DIR}"
mv -- "${gate}" "${MANIFEST_ROOT}/upstream_gate.json"
gate="${MANIFEST_ROOT}/upstream_gate.json"
jobs_tsv="${MANIFEST_ROOT}/submitted_jobs.tsv"
printf 'name\tjob_id\tdependency\tscript\tvariant\n' >"${jobs_tsv}"
submitted_ids=()
submission_complete=0
cancel_partial_chain() {
  local status=$?
  if [[ "${submission_complete}" -eq 0 && "${#submitted_ids[@]}" -gt 0 ]]; then
    echo "Submission failed; cancelling newly submitted jobs: ${submitted_ids[*]}" >&2
    scancel "${submitted_ids[@]}" || true
  fi
  exit "${status}"
}
trap cancel_partial_chain ERR INT TERM

previous=""
submit_job() {
  local name="$1" script="$2" variant="${3:-}"
  local dependency=""
  if [[ -z "${previous}" ]]; then
    dependency="${first_dependency}"
  else
    dependency="afterok:${previous}"
  fi
  local -a args=(
    sbatch --parsable "${COMMON[@]}"
    --output="${SLURM_LOG_DIR}/%x_%j.bootstrap.out"
  )
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="${dependency}")
  fi
  args+=(--export="ALL,AIME_EXPECTED_GIT_COMMIT=${submit_commit}${variant:+,SFT_VARIANT=${variant}}")
  local raw job_id
  raw="$("${args[@]}" "${PACKAGE}/${script}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "sbatch returned a nonnumeric job ID: ${raw}" >&2
    return 1
  }
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${job_id}" "${dependency}" "${script}" "${variant}" >>"${jobs_tsv}"
  submitted_ids+=("${job_id}")
  previous="${job_id}"
  python3 "${PACKAGE}/write_submission_manifest.py" \
    --output "${MANIFEST_ROOT}/submission_in_progress.json" \
    --jobs-tsv "${jobs_tsv}" --upstream-gate "${gate}" \
    --contract-hash "${CONTRACT_HASH}" --repo-root "${SLIME_REPO_ROOT}" \
    --expected-commit "${submit_commit}" \
    --mode in_progress --replace >/dev/null
  echo "AIME_JOB_SUBMITTED name=${name} id=${job_id} dependency=${dependency:-already_satisfied} variant=${variant:-none}"
}

submit_job prepare_data 01_prepare_data.sbatch
submit_job stage1_canary 02_stage1_canary.sbatch
submit_job train_aime_correct_3000 03_train.sbatch "${STAGE1_VARIANT}"
submit_job continuation_canary 04_continuation_canary.sbatch
for variant in "${CHILD_VARIANTS[@]}"; do
  submit_job "train_${variant}" 03_train.sbatch "${variant}"
done
submit_job finalize_handoff 05_finalize.sbatch

python3 "${PACKAGE}/write_submission_manifest.py" \
  --output "${MANIFEST_ROOT}/training_chain.json" --jobs-tsv "${jobs_tsv}" \
  --upstream-gate "${gate}" --contract-hash "${CONTRACT_HASH}" \
  --expected-commit "${submit_commit}" \
  --repo-root "${SLIME_REPO_ROOT}" --mode submitted
submission_complete=1
trap - ERR INT TERM
echo "AIME_TRAINING_CHAIN_SUBMITTED first=${submitted_ids[0]} final=${submitted_ids[-1]} jobs=${#submitted_ids[@]} contract=${CONTRACT_HASH}"

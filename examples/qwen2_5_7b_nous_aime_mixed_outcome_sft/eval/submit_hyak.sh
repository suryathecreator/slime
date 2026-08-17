#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit, optionally followed by --four-epoch}"
[[ "${MODE}" == "--test-only-shapes" || "${MODE}" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit [--four-epoch]" >&2
  exit 2
}
PROFILE="${2:-}"
[[ -z "${PROFILE}" || "${PROFILE}" == "--four-epoch" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit [--four-epoch]" >&2
  exit 2
}
[[ $# -le 2 ]] || { echo "Usage: $0 --test-only-shapes|--submit [--four-epoch]" >&2; exit 2; }
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PACKAGE="$(cd -- "${EVAL_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PACKAGE}/../.." >/dev/null 2>&1 && pwd)"
BRANCH="${EVAL_GIT_BRANCH:-qwen2.5-7b-aime-distribution-generalization}"
SUBMIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/mixed_aime_eval.py"
TARGETS=(base_qwen2_5_7b aime24_correct_3000 aime24_incorrect_3000 aime25_correct_3000 aime25_incorrect_3000)
if [[ "${PROFILE}" == "--four-epoch" ]]; then
  EVAL_EXPERIMENT_ROOT="${EVAL_EXPERIMENT_ROOT:-${REPO_ROOT}/checkpoints/qwen2_5_7b_nous_aime_mixed_outcome_sft/v1/2f2b580a2f52b1b7}"
  BASE_SOURCE_EXPERIMENT_ROOT="${REPO_ROOT}/checkpoints/qwen2_5_7b_nous_aime_mixed_outcome_sft/v1/2d556f01dbd853a1"
  BASE_SOURCE_RESULT_ROOT="${PACKAGE}/results/aime_mixed_outcome_sampled_v1"
  GENERATED_TARGETS=("${TARGETS[@]:1}")
  JOB_PREFIX=q25n4
else
  EVAL_EXPERIMENT_ROOT="${EVAL_EXPERIMENT_ROOT:-${REPO_ROOT}/checkpoints/qwen2_5_7b_nous_aime_mixed_outcome_sft/v1/2d556f01dbd853a1}"
  GENERATED_TARGETS=("${TARGETS[@]}")
  JOB_PREFIX=q25n
fi
COMMON=(
  --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue --chdir="${REPO_ROOT}"
  --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL
)
EXPORT_COMMON="ALL,REPO_ROOT=${REPO_ROOT},EVAL_EXPERIMENT_ROOT=${EVAL_EXPERIMENT_ROOT},HF_HOME=/gscratch/scrubbed/suryadv/hf_cache,HF_DATASETS_CACHE=/gscratch/scrubbed/suryadv/hf_cache/datasets,TRANSFORMERS_CACHE=/gscratch/scrubbed/suryadv/hf_cache/transformers,SCRUBBED_CACHE_ROOT=/gscratch/scrubbed/suryadv/cache/q25-nous-aime-eval,AIME_EXPECTED_GIT_COMMIT=${SUBMIT_COMMIT},PYTHONDONTWRITEBYTECODE=1"
LOG_DIR="${EVAL_EXPERIMENT_ROOT}/outputs/eval/aime_mixed_outcome_sampled_v1/_control/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ "${MODE}" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/preflight_cpu.sbatch"
  if [[ "${PROFILE}" == "--four-epoch" ]]; then
    sbatch --test-only "${COMMON[@]}" \
      --export="${EXPORT_COMMON},EVAL_BASE_SOURCE_EXPERIMENT_ROOT=${BASE_SOURCE_EXPERIMENT_ROOT},EVAL_BASE_SOURCE_RESULT_ROOT=${BASE_SOURCE_RESULT_ROOT}" \
      --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/reuse_base_cpu.sbatch"
  fi
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --gpus=h200:1 --exclude=g3130 --time=00:30:00 --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/canary_h200.sbatch"
  representative="${GENERATED_TARGETS[0]}"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=${representative}" \
    --array=0-15 --gpus=h200:1 --exclude=g3130 --time=02:00:00 \
    --output="${LOG_DIR}/%x-%A_%a.out" "${EVAL_DIR}/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=${representative}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
  echo "MIXED_AIME_SLURM_SHAPES_VALID generated_targets=${#GENERATED_TARGETS[@]} array_tasks_per_target=16 generated=$(( ${#GENERATED_TARGETS[@]} * 960 )) reused=$(( (5 - ${#GENERATED_TARGETS[@]}) * 960 )) scored=4800"
  exit 0
fi

[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "${BRANCH}" ]] || { echo "Wrong SLIME branch" >&2; exit 2; }
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || { echo "Commit evaluator before submission" >&2; exit 2; }
remote_commit="$(git -C "${REPO_ROOT}" rev-parse "refs/remotes/origin/${BRANCH}")"
[[ "${SUBMIT_COMMIT}" == "${remote_commit}" ]] || { echo "Evaluator commit is not pushed" >&2; exit 2; }
[[ -x "${RUNTIME_PYTHON}" && -f "${EVAL_EXPERIMENT_ROOT}/handoff/eval_contract.json" ]] || { echo "Runtime or contract missing" >&2; exit 2; }
JOURNAL="${EVAL_EXPERIMENT_ROOT}/outputs/eval/aime_mixed_outcome_sampled_v1/_control/submission.json"
[[ ! -e "${JOURNAL}" ]] || { echo "Submission already exists: ${JOURNAL}" >&2; exit 2; }

SUBMITTED=(); RECORDED=0; LAST_JOB=""
rollback() {
  status=$?; trap - EXIT
  if [[ "${RECORDED}" != 1 && ${#SUBMITTED[@]} -gt 0 ]]; then scancel "${SUBMITTED[@]}" 2>/dev/null || true; fi
  exit "${status}"
}
trap rollback EXIT
submit_job() {
  response="$(sbatch --parsable "$@")"; response="${response%%;*}"
  [[ "${response}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: ${response}" >&2; return 2; }
  LAST_JOB="${response}"; SUBMITTED+=("${response}")
}

submit_job "${COMMON[@]}" --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/preflight_cpu.sbatch"
preflight_job="${LAST_JOB}"
submit_job "${COMMON[@]}" --dependency="afterok:${preflight_job}" --export="${EXPORT_COMMON}" \
  --gpus=h200:1 --exclude=g3130 --time=00:30:00 --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/canary_h200.sbatch"
canary_job="${LAST_JOB}"

reuse_job=""
if [[ "${PROFILE}" == "--four-epoch" ]]; then
  submit_job "${COMMON[@]}" --dependency="afterok:${preflight_job}" --job-name=q25n4-base-reuse \
    --cpus-per-task=8 --mem=32G --time=00:30:00 \
    --export="${EXPORT_COMMON},EVAL_BASE_SOURCE_EXPERIMENT_ROOT=${BASE_SOURCE_EXPERIMENT_ROOT},EVAL_BASE_SOURCE_RESULT_ROOT=${BASE_SOURCE_RESULT_ROOT}" \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/reuse_base_cpu.sbatch"
  reuse_job="${LAST_JOB}"
fi

array_jobs=(); finalizer_jobs=(); array_assignments=(); finalizer_assignments=()
for target in "${GENERATED_TARGETS[@]}"; do
  slug="${target//_/-}"
  submit_job "${COMMON[@]}" --dependency="afterok:${canary_job}" --job-name="${JOB_PREFIX}-${slug}" \
    --array=0-15 --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --signal=B:USR1@600 --export="${EXPORT_COMMON},EVAL_TARGET=${target}" \
    --output="${LOG_DIR}/%x-%A_%a.out" "${EVAL_DIR}/run_h200.sbatch"
  array_job="${LAST_JOB}"
  submit_job "${COMMON[@]}" --dependency="afterok:${array_job}" --job-name="${JOB_PREFIX}-${slug}-final" \
    --cpus-per-task=8 --mem=32G --time=00:30:00 --export="${EXPORT_COMMON},EVAL_TARGET=${target}" \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/finalize_cpu.sbatch"
  finalizer_job="${LAST_JOB}"
  array_jobs+=("${array_job}"); finalizer_jobs+=("${finalizer_job}")
  array_assignments+=(--array-job "${target}=${array_job}")
  finalizer_assignments+=(--finalizer-job "${target}=${finalizer_job}")
done
audit_dependencies=("${finalizer_jobs[@]}")
[[ -z "${reuse_job}" ]] || audit_dependencies+=("${reuse_job}")
dependency="afterok:$(IFS=:; echo "${audit_dependencies[*]}")"
submit_job "${COMMON[@]}" --dependency="${dependency}" --job-name="${JOB_PREFIX}-audit" \
  --cpus-per-task=8 --mem=32G --time=00:30:00 --export="${EXPORT_COMMON}" \
  --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
audit_job="${LAST_JOB}"

record_args=(
  --preflight-job "${preflight_job}" --canary-job "${canary_job}"
  "${array_assignments[@]}" "${finalizer_assignments[@]}"
  --audit-job "${audit_job}" --git-commit "${SUBMIT_COMMIT}"
)
if [[ "${PROFILE}" == "--four-epoch" ]]; then
  record_args+=(
    --reuse-job "${reuse_job}" --reuse-target base_qwen2_5_7b
    --reuse-source-experiment-root "${BASE_SOURCE_EXPERIMENT_ROOT}"
    --reuse-source-result-root "${BASE_SOURCE_RESULT_ROOT}"
  )
fi
"${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" record-submission \
  "${record_args[@]}"
RECORDED=1; trap - EXIT
echo "MIXED_AIME_SUBMITTED commit=${SUBMIT_COMMIT} preflight=${preflight_job} reuse=${reuse_job:-none}" \
  "canary=${canary_job} arrays=${array_jobs[*]} finalizers=${finalizer_jobs[*]} audit=${audit_job}"

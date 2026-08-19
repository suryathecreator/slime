#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "${MODE}" == "--test-only-shapes" || "${MODE}" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}
[[ $# -le 1 ]] || { echo "Usage: $0 --test-only-shapes|--submit" >&2; exit 2; }

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PACKAGE="$(cd -- "${EVAL_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PACKAGE}/../.." >/dev/null 2>&1 && pwd)"
BRANCH="${EVAL_GIT_BRANCH:-qwen2.5-7b-aime-distribution-generalization}"
SUBMIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/split_aime_eval.py"
EVAL_NAME=aime_split_control_sampled_v1
JOB_PREFIX=q25split
TARGETS=(
  base_qwen2_5_7b
  held_in_correct_6000
  held_in_incorrect_6000
  held_out_correct_6000
  held_out_incorrect_6000
)
EVAL_EXPERIMENT_ROOT="${EVAL_EXPERIMENT_ROOT:-${REPO_ROOT}/checkpoints/qwen2_5_7b_aime_2009_2024_split_sft_6k/v1/0b22cc069c941ec5}"

COMMON=(
  --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue --chdir="${REPO_ROOT}"
  --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL
)
# record-submission writes the committed metadata file; the delayed jobs therefore
# have to accept a clean descendant commit touching exactly that one path.
GATE_EXPORT=",AIME_RUNTIME_ALLOWED_DESCENDANT_PATH=examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/eval/submission_metadata.json"
EXPORT_COMMON="ALL,REPO_ROOT=${REPO_ROOT},EVAL_EXPERIMENT_ROOT=${EVAL_EXPERIMENT_ROOT},HF_HOME=/gscratch/scrubbed/suryadv/hf_cache,HF_DATASETS_CACHE=/gscratch/scrubbed/suryadv/hf_cache/datasets,TRANSFORMERS_CACHE=/gscratch/scrubbed/suryadv/hf_cache/transformers,SCRUBBED_CACHE_ROOT=/gscratch/scrubbed/suryadv/cache/q25-split-aime-eval,AIME_EXPECTED_GIT_COMMIT=${SUBMIT_COMMIT},PYTHONDONTWRITEBYTECODE=1${GATE_EXPORT}"
LOG_DIR="${EVAL_EXPERIMENT_ROOT}/outputs/eval/${EVAL_NAME}/_control/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ "${MODE}" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/preflight_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --gpus=h200:1 --exclude=g3130 \
    --cpus-per-task=8 --mem=220G --time=00:30:00 \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/canary_h200.sbatch"
  representative="${TARGETS[0]}"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=${representative}" \
    --array=0-7 --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=220G --time=08:00:00 \
    --signal=B:USR1@600 --output="${LOG_DIR}/%x-%A_%a.out" "${EVAL_DIR}/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=${representative}" \
    --cpus-per-task=8 --mem=32G --time=00:30:00 \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --cpus-per-task=8 --mem=32G --time=00:30:00 \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
  echo "SPLIT_AIME_SLURM_SHAPES_VALID targets=${#TARGETS[@]} array_tasks_per_target=8" \
    "h200_tasks=$(( ${#TARGETS[@]} * 8 )) completions=$(( ${#TARGETS[@]} * 212 * 4 ))"
  exit 0
fi

[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "${BRANCH}" ]] || { echo "Wrong SLIME branch" >&2; exit 2; }
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || { echo "Commit evaluator before submission" >&2; exit 2; }
remote_commit="$(git -C "${REPO_ROOT}" rev-parse "refs/remotes/origin/${BRANCH}")"
[[ "${SUBMIT_COMMIT}" == "${remote_commit}" ]] || { echo "Evaluator commit is not pushed" >&2; exit 2; }
[[ -x "${RUNTIME_PYTHON}" && -f "${EVAL_EXPERIMENT_ROOT}/handoff/eval_contract.json" ]] || {
  echo "Runtime or contract missing" >&2; exit 2; }
JOURNAL="${EVAL_EXPERIMENT_ROOT}/outputs/eval/${EVAL_NAME}/_control/submission.json"
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

submit_job "${COMMON[@]}" --cpus-per-task=8 --mem=96G --time=02:00:00 \
  --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/preflight_cpu.sbatch"
preflight_job="${LAST_JOB}"
submit_job "${COMMON[@]}" --dependency="afterok:${preflight_job}" --export="${EXPORT_COMMON}" \
  --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=220G --time=00:30:00 \
  --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/canary_h200.sbatch"
canary_job="${LAST_JOB}"

array_jobs=(); finalizer_jobs=(); array_assignments=(); finalizer_assignments=()
for target in "${TARGETS[@]}"; do
  slug="${target//_/-}"
  submit_job "${COMMON[@]}" --dependency="afterok:${canary_job}" --job-name="${JOB_PREFIX}-${slug}" \
    --array=0-7 --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=220G --time=08:00:00 \
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
dependency="afterok:$(IFS=:; echo "${finalizer_jobs[*]}")"
submit_job "${COMMON[@]}" --dependency="${dependency}" --job-name="${JOB_PREFIX}-audit" \
  --cpus-per-task=8 --mem=32G --time=00:30:00 --export="${EXPORT_COMMON}" \
  --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
audit_job="${LAST_JOB}"

"${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" \
  record-submission --preflight-job "${preflight_job}" --canary-job "${canary_job}" \
  "${array_assignments[@]}" "${finalizer_assignments[@]}" \
  --audit-job "${audit_job}" --git-commit "${SUBMIT_COMMIT}"
RECORDED=1; trap - EXIT
echo "SPLIT_AIME_SUBMITTED commit=${SUBMIT_COMMIT} preflight=${preflight_job} canary=${canary_job}" \
  "arrays=${array_jobs[*]} finalizers=${finalizer_jobs[*]} audit=${audit_job}"
echo "Commit and push examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/eval/submission_metadata.json now:"
echo "  the delayed jobs accept only a clean descendant commit touching exactly that path."

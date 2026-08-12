#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "${MODE}" == "--test-only-shapes" || "${MODE}" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}
[[ $# -eq 1 ]] || { echo "Usage: $0 --test-only-shapes|--submit" >&2; exit 2; }
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PACKAGE="$(cd -- "${EVAL_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PACKAGE}/../.." >/dev/null 2>&1 && pwd)"
CONTRACT_FILE="${PACKAGE}/config/experiment_contract.json"
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
SUBMIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
CONTROL="${EVAL_DIR}/aime_eval.py"
BRANCH="${EVAL_GIT_BRANCH:-qwen2.5-7b-aime-distribution-generalization}"
EVAL_EXPERIMENT_ROOT="${EVAL_EXPERIMENT_ROOT:-${REPO_ROOT}/checkpoints/qwen2_5_7b_aime_generalization_incorrect_sft/v1/${CONTRACT_HASH}}"
HELD_IN_EVAL_JSONL="${HYAK_HELD_IN_EVAL_JSONL:-${EVAL_EXPERIMENT_ROOT}/data/eval/held_in.jsonl}"
HELD_OUT_EVAL_JSONL="${HYAK_HELD_OUT_EVAL_JSONL:-${EVAL_EXPERIMENT_ROOT}/data/eval/held_out_problem.jsonl}"
HYAK_HF_HOME="${HYAK_HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
HYAK_CACHE_ROOT="${HYAK_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25-aime-eval}"
TARGETS=(
  base_qwen2_5_7b
  aime_correct_3000
  continue_wrong_unmasked
  continue_random_mask_10
  continue_random_mask_20
  continue_random_mask_30
  continue_random_mask_40
  continue_random_mask_50
  continue_random_mask_60
  continue_random_mask_70
  continue_random_mask_80
  continue_random_mask_90
  continue_correct_only_1500
)
COMMON=(
  --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue --chdir="${REPO_ROOT}"
  --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL
)
EXPORT_COMMON="ALL,REPO_ROOT=${REPO_ROOT},EVAL_EXPERIMENT_ROOT=${EVAL_EXPERIMENT_ROOT},HELD_IN_EVAL_JSONL=${HELD_IN_EVAL_JSONL},HELD_OUT_EVAL_JSONL=${HELD_OUT_EVAL_JSONL},HF_HOME=${HYAK_HF_HOME},HF_DATASETS_CACHE=${HYAK_HF_HOME}/datasets,TRANSFORMERS_CACHE=${HYAK_HF_HOME}/transformers,SCRUBBED_CACHE_ROOT=${HYAK_CACHE_ROOT},AIME_EXPECTED_GIT_COMMIT=${SUBMIT_COMMIT},EVAL_GIT_COMMIT=${SUBMIT_COMMIT},PYTHONDONTWRITEBYTECODE=1"
LOG_DIR="${EVAL_EXPERIMENT_ROOT}/outputs/eval/aime_generalization_sampled_v1/_control/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ "${MODE}" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/preflight_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --gpus=h200:1 --exclude=g3130 --time=00:30:00 --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/canary_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=base_qwen2_5_7b" --array=0-23 --gpus=h200:1 --exclude=g3130 --time=02:00:00 --output="${LOG_DIR}/%x-%A_%a.out" "${EVAL_DIR}/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON},EVAL_TARGET=base_qwen2_5_7b" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --export="${EXPORT_COMMON}" --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
  echo "AIME_EVAL_SLURM_SHAPES_VALID targets=13 array_tasks_per_target=24 generations=31200"
  exit 0
fi

[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "${BRANCH}" ]] || { echo "Wrong SLIME branch" >&2; exit 2; }
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || { echo "Commit evaluator before submission" >&2; exit 2; }
git_commit="${SUBMIT_COMMIT}"
remote_commit="$(git -C "${REPO_ROOT}" rev-parse "refs/remotes/origin/${BRANCH}")"
[[ "${git_commit}" == "${remote_commit}" ]] || { echo "Evaluator commit is not pushed" >&2; exit 2; }
[[ -x "${RUNTIME_PYTHON}" ]] || { echo "Missing runtime: ${RUNTIME_PYTHON}" >&2; exit 2; }
[[ -f "${HELD_IN_EVAL_JSONL}" && -f "${HELD_OUT_EVAL_JSONL}" ]] || { echo "Transferred eval data are missing" >&2; exit 2; }
[[ -f "${EVAL_EXPERIMENT_ROOT}/handoff/comparison_sources.json" ]] || { echo "Transferred checkpoint inventory is missing" >&2; exit 2; }
JOURNAL="${EVAL_EXPERIMENT_ROOT}/outputs/eval/aime_generalization_sampled_v1/_control/submission.json"
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
array_jobs=(); finalizer_jobs=(); array_assignments=(); finalizer_assignments=()
for target in "${TARGETS[@]}"; do
  slug="${target//_/-}"
  submit_job "${COMMON[@]}" --dependency="afterok:${canary_job}" --job-name="q25a-${slug}" \
    --array=0-23 --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --signal=B:USR1@600 --export="${EXPORT_COMMON},EVAL_TARGET=${target}" \
    --output="${LOG_DIR}/%x-%A_%a.out" "${EVAL_DIR}/run_h200.sbatch"
  array_job="${LAST_JOB}"
  submit_job "${COMMON[@]}" --dependency="afterok:${array_job}" --job-name="q25a-${slug}-final" \
    --cpus-per-task=8 --mem=64G --time=02:00:00 \
    --export="${EXPORT_COMMON},EVAL_TARGET=${target}" \
    --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/finalize_cpu.sbatch"
  finalizer_job="${LAST_JOB}"
  array_jobs+=("${array_job}"); finalizer_jobs+=("${finalizer_job}")
  array_assignments+=(--array-job "${target}=${array_job}")
  finalizer_assignments+=(--finalizer-job "${target}=${finalizer_job}")
done
dependency="afterok:$(IFS=:; echo "${finalizer_jobs[*]}")"
submit_job "${COMMON[@]}" --dependency="${dependency}" --job-name=q25-aime-audit \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --export="${EXPORT_COMMON}" \
  --output="${LOG_DIR}/%x-%j.out" "${EVAL_DIR}/audit_cpu.sbatch"
audit_job="${LAST_JOB}"
"${RUNTIME_PYTHON}" "${CONTROL}" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" record-submission \
  --preflight-job "${preflight_job}" --canary-job "${canary_job}" \
  "${array_assignments[@]}" "${finalizer_assignments[@]}" \
  --audit-job "${audit_job}" --git-commit "${git_commit}"
RECORDED=1; trap - EXIT
echo "AIME_EVAL_SUBMITTED commit=${git_commit} preflight=${preflight_job} canary=${canary_job} arrays=${array_jobs[*]} finalizers=${finalizer_jobs[*]} audit=${audit_job}"

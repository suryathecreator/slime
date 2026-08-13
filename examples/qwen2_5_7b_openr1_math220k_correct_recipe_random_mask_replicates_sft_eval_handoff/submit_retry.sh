#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == --test-only-shapes || "$MODE" == --submit ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
BRANCH=qwen-correct-only-openr1-sft-2k
ATTEMPT=2
ORIGINAL_GIT_COMMIT=870ac7698548a31de27ee6eb73aaa46c2218c54b
TARGETS=(
  random_mask_05_seed42 random_mask_15_seed42
  random_mask_05_seed43 random_mask_15_seed43
  random_mask_05_seed44 random_mask_15_seed44
)

[[ -x "$RUNTIME_PYTHON" ]] || { echo "Missing pinned runtime: $RUNTIME_PYTHON" >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$BRANCH" ]] || { echo "Wrong SLIME branch" >&2; exit 2; }
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" validate-retry   --original-git-commit "$ORIGINAL_GIT_COMMIT"
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control)"
ORIGINAL_JOURNAL="$CONTROL_ROOT/submission.json"
PREFLIGHT_JOB="$("$RUNTIME_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["preflight_job"])' "$ORIGINAL_JOURNAL")"
LOG_DIR="$CONTROL_ROOT/slurm_logs"; mkdir -p "$LOG_DIR"
COMMON=(
  --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue
  --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL
)

if [[ "$MODE" == --test-only-shapes ]]; then
  sbatch --test-only "${COMMON[@]}" --gpus=h200:1 --exclude=g3130 --time=00:30:00 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/canary_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-11 --gpus=h200:1 --exclude=g3130 --export=ALL,EVAL_TARGET=random_mask_05_seed42 --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export=ALL,EVAL_TARGET=random_mask_05_seed42,EVAL_GIT_COMMIT=0000000000000000000000000000000000000000 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_cpu.sbatch"
  echo "Q25R_MATH500_RETRY_SLURM_SHAPES_VALID attempt=2 targets=6 tasks=72 canary_cases=1"
  exit 0
fi

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Commit retry patch before submission" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
}
git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
remote_commit="$(git -C "$REPO_ROOT" rev-parse "refs/remotes/origin/$BRANCH")"
[[ "$git_commit" == "$remote_commit" ]] || {
  echo "Retry patch is not pushed: local=$git_commit origin=$remote_commit" >&2
  exit 2
}

SUBMITTED=(); RECORDED=0; LAST_JOB=""
rollback() {
  status=$?; trap - EXIT
  if [[ "$RECORDED" != 1 && ${#SUBMITTED[@]} -gt 0 ]]; then
    scancel "${SUBMITTED[@]}" 2>/dev/null || true
  fi
  exit "$status"
}
trap rollback EXIT
submit_job() {
  response="$(sbatch --parsable "$@")"; response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || {
    echo "Invalid sbatch response: $response" >&2
    return 2
  }
  LAST_JOB="$response"; SUBMITTED+=("$response")
}

submit_job "${COMMON[@]}" --gpus=h200:1 --exclude=g3130 --time=00:30:00   --output="$LOG_DIR/%x-%j.out" "$HANDOFF/canary_h200.sbatch"
canary_job="$LAST_JOB"
array_jobs=(); finalizer_jobs=(); array_assignments=(); finalizer_assignments=()
for target in "${TARGETS[@]}"; do
  slug="${target//_/-}"
  submit_job "${COMMON[@]}" --dependency="afterok:$canary_job" --job-name="q25r2-m5-$slug"     --array=0-11 --gpus=h200:1 --exclude=g3130 --cpus-per-task=8 --mem=96G --time=02:00:00     --signal=B:USR1@600 --export=ALL,EVAL_TARGET="$target"     --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  array_job="$LAST_JOB"
  submit_job "${COMMON[@]}" --dependency="afterok:$array_job" --job-name="q25r2-m5-$slug-final"     --cpus-per-task=8 --mem=64G --time=02:00:00     --export=ALL,EVAL_TARGET="$target",EVAL_GIT_COMMIT="$git_commit"     --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_cpu.sbatch"
  finalizer_job="$LAST_JOB"
  array_jobs+=("$array_job"); finalizer_jobs+=("$finalizer_job")
  array_assignments+=(--array-job "$target=$array_job")
  finalizer_assignments+=(--finalizer-job "$target=$finalizer_job")
done
dependency="afterok:$(IFS=:; echo "${finalizer_jobs[*]}")"
submit_job "${COMMON[@]}" --dependency="$dependency" --job-name=q25r2-m500-audit   --cpus-per-task=8 --mem=64G --time=02:00:00   --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_cpu.sbatch"
audit_job="$LAST_JOB"

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" record-submission   --attempt "$ATTEMPT" --reuse-preflight --supersedes-submission "$ORIGINAL_JOURNAL"   --preflight-job "$PREFLIGHT_JOB" --canary-job "$canary_job"   "${array_assignments[@]}" "${finalizer_assignments[@]}"   --audit-job "$audit_job" --git-commit "$git_commit"
RECORDED=1; trap - EXIT
echo "Q25R_MATH500_RETRY_SUBMITTED attempt=2 commit=$git_commit preflight_reused=$PREFLIGHT_JOB canary=$canary_job arrays=${array_jobs[*]} finalizers=${finalizer_jobs[*]} audit=$audit_job"

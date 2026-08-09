#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == --test-only-shapes || "$MODE" == --submit ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
SOURCE_EVAL="${SOURCE_EVAL:-$AXOLOTL_ROOT/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/eval_benchmarks/math500.jsonl}"
BRANCH=qwen-correct-only-openr1-sft-2k
TARGETS=(
  base_qwen2_5_3b qwen2_5_3b_8k qwen2_5_3b_16k
  base_qwen2_5_7b qwen2_5_7b_8k
  base_qwen3_4b qwen3_4b_8k
  base_qwen3_8b qwen3_8b_8k
)
[[ -x "$RUNTIME_PYTHON" ]] || { echo "Missing pinned runtime: $RUNTIME_PYTHON" >&2; exit 2; }
[[ -f "$SOURCE_EVAL" ]] || { echo "Missing pinned MATH-500 source: $SOURCE_EVAL" >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$BRANCH" ]] || { echo "Wrong SLIME branch" >&2; exit 2; }
[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == 6b8f0e3314e3d162260cdc35d84741c3da163f30 ]] || { echo "Axolotl commit changed" >&2; exit 2; }
[[ -z "$(git -C "$AXOLOTL_ROOT" status --porcelain)" ]] || { echo "Axolotl worktree is dirty" >&2; exit 2; }

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" prepare --source "$SOURCE_EVAL"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" validate-gold
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control)"
LOG_DIR="$CONTROL_ROOT/slurm_logs"; mkdir -p "$LOG_DIR"
COMMON=(
  --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue
  --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL
)

if [[ "$MODE" == --test-only-shapes ]]; then
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --gpus=h200:1 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/canary_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-15 --gpus=h200:1 --export=ALL,EVAL_TARGET=base_qwen2_5_3b --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export=ALL,EVAL_TARGET=base_qwen2_5_3b,EVAL_GIT_COMMIT=0000000000000000000000000000000000000000 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_cpu.sbatch"
  echo "QCO_MATH500_SLURM_SHAPES_VALID targets=9 tasks=144"
  exit 0
fi

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "Commit evaluator before submission" >&2; git -C "$REPO_ROOT" status --short >&2; exit 2; }
git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
remote_commit="$(git -C "$REPO_ROOT" rev-parse "refs/remotes/origin/$BRANCH")"
[[ "$git_commit" == "$remote_commit" ]] || { echo "Evaluator commit is not pushed: local=$git_commit origin=$remote_commit" >&2; exit 2; }
JOURNAL="$CONTROL_ROOT/submission.json"
[[ ! -e "$JOURNAL" ]] || { echo "Submission already exists: $JOURNAL" >&2; exit 2; }

SUBMITTED=(); RECORDED=0; LAST_JOB=""
rollback() {
  status=$?; trap - EXIT
  if [[ "$RECORDED" != 1 && ${#SUBMITTED[@]} -gt 0 ]]; then scancel "${SUBMITTED[@]}" 2>/dev/null || true; fi
  exit "$status"
}
trap rollback EXIT
submit_job() {
  response="$(sbatch --parsable "$@")"; response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $response" >&2; return 2; }
  LAST_JOB="$response"; SUBMITTED+=("$response")
}

submit_job "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_cpu.sbatch"
preflight_job="$LAST_JOB"
submit_job "${COMMON[@]}" --dependency="afterok:$preflight_job" --gpus=h200:1 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/canary_h200.sbatch"
canary_job="$LAST_JOB"
array_jobs=(); finalizer_jobs=(); array_assignments=(); finalizer_assignments=()
for target in "${TARGETS[@]}"; do
  slug="${target//_/-}"
  submit_job "${COMMON[@]}" --dependency="afterok:$canary_job" --job-name="qco-m5-$slug" \
    --array=0-15 --gpus=h200:1 --cpus-per-task=8 --mem=96G --time=02:00:00 \
    --signal=B:USR1@600 --export=ALL,EVAL_TARGET="$target" \
    --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  array_job="$LAST_JOB"
  submit_job "${COMMON[@]}" --dependency="afterok:$array_job" --job-name="qco-m5-$slug-final" \
    --cpus-per-task=8 --mem=64G --time=02:00:00 \
    --export=ALL,EVAL_TARGET="$target",EVAL_GIT_COMMIT="$git_commit" \
    --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_cpu.sbatch"
  finalizer_job="$LAST_JOB"
  array_jobs+=("$array_job"); finalizer_jobs+=("$finalizer_job")
  array_assignments+=(--array-job "$target=$array_job")
  finalizer_assignments+=(--finalizer-job "$target=$finalizer_job")
done
dependency="afterok:$(IFS=:; echo "${finalizer_jobs[*]}")"
submit_job "${COMMON[@]}" --dependency="$dependency" --cpus-per-task=8 --mem=64G --time=02:00:00 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_cpu.sbatch"
audit_job="$LAST_JOB"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" --canary-job "$canary_job" \
  "${array_assignments[@]}" "${finalizer_assignments[@]}" \
  --audit-job "$audit_job" --git-commit "$git_commit"
RECORDED=1; trap - EXIT
echo "QCO_MATH500_SUBMITTED commit=$git_commit preflight=$preflight_job canary=$canary_job arrays=${array_jobs[*]} finalizers=${finalizer_jobs[*]} audit=$audit_job"

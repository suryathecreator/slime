#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes, --submit, or --resubmit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" || "$MODE" == "--resubmit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit|--resubmit" >&2
  exit 2
}

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
CONTROL="$HANDOFF/math500_eval.py"
SOURCE_EVAL="${SOURCE_EVAL:-$AXOLOTL_ROOT/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/eval_benchmarks/math500.jsonl}"
BRANCH="qwen3-0.6b-openr1-masked-sft-2k"
VARIANTS=(
  base_0p6b
  correct_only
  unmasked
  random_mask_25
  random_mask_50
  random_mask_70
  random_mask_80
  random_mask_90
  inverse_tau_0p20
  inverse_tau_0p05
  margin_mask
  prob_ratio_mask
)

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
[[ -x "$RUNTIME_PYTHON" ]] || { echo "Missing pinned runtime: $RUNTIME_PYTHON" >&2; exit 2; }
[[ -f "$SOURCE_EVAL" ]] || { echo "Missing pinned MATH-500 JSONL: $SOURCE_EVAL" >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$BRANCH" ]] || {
  echo "Wrong SLIME execution branch" >&2
  exit 2
}
[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == "6b8f0e3314e3d162260cdc35d84741c3da163f30" ]] || {
  echo "Axolotl evaluator commit changed" >&2
  exit 2
}
[[ -z "$(git -C "$AXOLOTL_ROOT" status --porcelain)" ]] || {
  echo "Axolotl evaluator worktree is dirty" >&2
  exit 2
}

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" prepare --source "$SOURCE_EVAL"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" validate-gold

CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
LOG_DIR="$CONTROL_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --qos=ckpt
  --requeue
  --mail-user=suryadv@cs.washington.edu
  --mail-type=END,FAIL
)

if [[ "$MODE" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --gpus=h200:1 --output="$LOG_DIR/%x-%j.out" "$HANDOFF/canary_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --gpus=h200:1 \
    --export=ALL,EVAL_VARIANT=base_0p6b \
    --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export=ALL,EVAL_VARIANT=base_0p6b \
    --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_cpu.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_cpu.sbatch"
  echo "MATH500_V6_SLURM_SHAPES_VALID checkpoints=12 shards=48"
  exit 0
fi

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Commit evaluator and provenance before submission" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
}
local_head="$(git -C "$REPO_ROOT" rev-parse HEAD)"
remote_head="$(git -C "$REPO_ROOT" rev-parse "refs/remotes/origin/$BRANCH")"
[[ "$local_head" == "$remote_head" ]] || {
  echo "Local HEAD is not the pushed origin branch: local=$local_head origin=$remote_head" >&2
  exit 2
}
SUBMISSION_JOURNAL="$CONTROL_ROOT/submission.json"
if [[ "$MODE" == "--submit" ]]; then
  [[ ! -e "$SUBMISSION_JOURNAL" ]] || {
    echo "Submission journal already exists: $SUBMISSION_JOURNAL" >&2
    exit 2
  }
else
  [[ -f "$SUBMISSION_JOURNAL" ]] || {
    echo "Missing submission journal to supersede: $SUBMISSION_JOURNAL" >&2
    exit 2
  }
  mapfile -t PRIOR_JOBS < <(
    "$RUNTIME_PYTHON" - "$SUBMISSION_JOURNAL" "${VARIANTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
variants = set(sys.argv[2:])
value = json.loads(path.read_text())
if value.get("checkpoint_count") != 12 or value.get("eval_task_count") != 48:
    raise RuntimeError("prior submission shape changed")
if set(value.get("array_jobs", {})) != variants:
    raise RuntimeError("prior array inventory changed")
if set(value.get("merge_jobs", {})) != variants:
    raise RuntimeError("prior merge inventory changed")
jobs = {
    str(value["preflight_job"]),
    str(value["canary_job"]),
    str(value["audit_job"]),
    *(str(item) for item in value["array_jobs"].values()),
    *(str(item) for item in value["merge_jobs"].values()),
}
for job in sorted(jobs, key=int):
    if not job.isdigit():
        raise RuntimeError(f"invalid prior job id: {job}")
    print(job)
PY
  )
  [[ "${#PRIOR_JOBS[@]}" == 27 ]] || {
    echo "Expected 27 jobs in prior submission; found ${#PRIOR_JOBS[@]}" >&2
    exit 2
  }
  INFRA_FAILURE_LOG=""
  INFRA_FAILURE_KIND=""
  for prior_job in "${PRIOR_JOBS[@]}"; do
    for candidate_log in \
      "$LOG_DIR"/*-"$prior_job".out \
      "$LOG_DIR"/*-"$prior_job"_*.out; do
      [[ -f "$candidate_log" ]] || continue
      if grep -Fq "zmq.error.ZMQError" "$candidate_log" \
        && grep -Fq "is longer than 107 characters" "$candidate_log" \
        && grep -Eq "/q06-math500-v(5|6)/tasks/job_" "$candidate_log"; then
        INFRA_FAILURE_LOG="$candidate_log"
        INFRA_FAILURE_KIND="zmq_ipc_path_too_long"
        break 2
      elif grep -Fq "torch._dynamo.exc.BackendCompilerFailed" "$candidate_log" \
        && grep -Eq "/q06-math500-v(5|6)/vllm/torch_compile_cache/" "$candidate_log"; then
        INFRA_FAILURE_LOG="$candidate_log"
        INFRA_FAILURE_KIND="shared_compile_cache_race"
        break 2
      fi
    done
  done
  [[ -n "$INFRA_FAILURE_LOG" ]] || {
    echo "No approved infrastructure failure was found in prior job logs" >&2
    exit 2
  }
  prior_csv="$(IFS=,; echo "${PRIOR_JOBS[*]}")"
  mapfile -t ACTIVE_PRIOR_JOBS < <(squeue --noheader --jobs="$prior_csv" --format='%A' | sort -u)
  if (( ${#ACTIVE_PRIOR_JOBS[@]} )); then
    scancel "${ACTIVE_PRIOR_JOBS[@]}"
  fi
  for _ in {1..60}; do
    [[ -z "$(squeue --noheader --jobs="$prior_csv" --format='%i')" ]] && break
    sleep 2
  done
  [[ -z "$(squeue --noheader --jobs="$prior_csv" --format='%i')" ]] || {
    echo "Superseded jobs did not leave the queue within 120 seconds" >&2
    exit 2
  }

  attempt_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  ATTEMPT_ARCHIVE="$CONTROL_ROOT/submission_attempts/$attempt_stamp"
  mkdir -p "$ATTEMPT_ARCHIVE"
  mv "$SUBMISSION_JOURNAL" "$ATTEMPT_ARCHIVE/submission.json"
  if [[ -f "$CONTROL_ROOT/CANARY_READY.json" ]]; then
    mv "$CONTROL_ROOT/CANARY_READY.json" "$ATTEMPT_ARCHIVE/CANARY_READY.json"
  fi
  if [[ -d "$CONTROL_ROOT/canary" ]]; then
    mv "$CONTROL_ROOT/canary" "$ATTEMPT_ARCHIVE/canary"
  fi
  echo "MATH500_V6_SUPERSEDED archive=$ATTEMPT_ARCHIVE failure_kind=$INFRA_FAILURE_KIND failure_log=$INFRA_FAILURE_LOG jobs=${PRIOR_JOBS[*]}"
fi

SUBMITTED_JOBS=()
SUBMISSION_RECORDED=0
LAST_SUBMITTED_JOB=""
rollback_unrecorded_submission() {
  local shell_status=$?
  trap - EXIT
  if [[ "$SUBMISSION_RECORDED" != 1 ]] && (( ${#SUBMITTED_JOBS[@]} )); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
    echo "Canceled unrecorded replacement jobs: ${SUBMITTED_JOBS[*]}" >&2
  fi
  exit "$shell_status"
}
trap rollback_unrecorded_submission EXIT

submit_job() {
  local response
  response="$(sbatch --parsable "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job id: $response" >&2; return 2; }
  LAST_SUBMITTED_JOB="$response"
  SUBMITTED_JOBS+=("$response")
}

submit_job "${COMMON[@]}" \
  --output="$LOG_DIR/%x-%j.out" \
  "$HANDOFF/preflight_cpu.sbatch"
preflight_job="$LAST_SUBMITTED_JOB"
submit_job "${COMMON[@]}" \
  --dependency="afterok:$preflight_job" \
  --gpus=h200:1 \
  --output="$LOG_DIR/%x-%j.out" \
  "$HANDOFF/canary_h200.sbatch"
canary_job="$LAST_SUBMITTED_JOB"

array_jobs=()
merge_jobs=()
array_assignments=()
merge_assignments=()
for variant in "${VARIANTS[@]}"; do
  slug="${variant//_/-}"
  submit_job "${COMMON[@]}" \
    --dependency="afterok:$canary_job" \
    --job-name="q06-m5-${slug}" \
    --array=0-3 \
    --gpus=h200:1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=02:00:00 \
    --signal=B:USR1@600 \
    --export=ALL,EVAL_VARIANT="$variant" \
    --output="$LOG_DIR/%x-%A_%a.out" \
    "$HANDOFF/run_h200.sbatch"
  array_job="$LAST_SUBMITTED_JOB"
  submit_job "${COMMON[@]}" \
    --dependency="afterok:$array_job" \
    --job-name="q06-m5-${slug}-final" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:00:00 \
    --export=ALL,EVAL_VARIANT="$variant" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/finalize_cpu.sbatch"
  merge_job="$LAST_SUBMITTED_JOB"
  array_jobs+=("$array_job")
  merge_jobs+=("$merge_job")
  array_assignments+=(--array-job "$variant=$array_job")
  merge_assignments+=(--merge-job "$variant=$merge_job")
done

merge_dependency="afterok:$(IFS=:; echo "${merge_jobs[*]}")"
submit_job "${COMMON[@]}" \
  --dependency="$merge_dependency" \
  --cpus-per-task=8 \
  --mem=64G \
  --time=01:00:00 \
  --output="$LOG_DIR/%x-%j.out" \
  "$HANDOFF/audit_cpu.sbatch"
audit_job="$LAST_SUBMITTED_JOB"

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" \
  --canary-job "$canary_job" \
  "${array_assignments[@]}" \
  "${merge_assignments[@]}" \
  --audit-job "$audit_job"

SUBMISSION_RECORDED=1
trap - EXIT
echo "MATH500_V6_SUBMITTED preflight=$preflight_job canary=$canary_job arrays=${array_jobs[*]} merges=${merge_jobs[*]} audit=$audit_job"

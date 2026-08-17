#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 --test-only|--submit}"
[[ "$MODE" == "--test-only" || "$MODE" == "--submit" ]] || {
  echo "Usage: $0 --test-only|--submit" >&2
  exit 2
}

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/bin/python"
OUTPUT_ROOT="$("$PYTHON_BIN" "$HANDOFF/pipeline.py" path --field output_root)"
LOG_ROOT="$("$PYTHON_BIN" "$HANDOFF/pipeline.py" path --field log_root)"
mkdir -p "$LOG_ROOT"

"$PYTHON_BIN" "$HANDOFF/pipeline.py" verify >/dev/null

COMMON=(
  --account=raivn-ckpt
  --partition=ckpt-all
  --qos=ckpt
  --nodes=1
  --ntasks=1
  --mail-user=suryadv@cs.washington.edu
  --mail-type=END,FAIL
  --export=ALL,REPO_ROOT="$REPO_ROOT"
)

submit_job() {
  local response
  response="$(sbatch --parsable "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || {
    echo "Unexpected sbatch job ID: $response" >&2
    return 2
  }
  printf '%s\n' "$response"
}

if [[ "$MODE" == "--test-only" ]]; then
  sbatch --test-only "${COMMON[@]}" --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/preflight.sbatch"
  sbatch --test-only "${COMMON[@]}" --gpus=h200:1 --cpus-per-task=8 --mem=220G \
    --time=02:00:00 --requeue --signal=B:USR1@600 \
    --export=ALL,REPO_ROOT="$REPO_ROOT",TASK_ID_OVERRIDE=0,CANARY_LIMIT=1 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-63 --gpus=h200:1 --cpus-per-task=8 \
    --mem=220G --time=08:00:00 --requeue --signal=B:USR1@600 \
    --output="$LOG_ROOT/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/audit.sbatch"
  sbatch --test-only "${COMMON[@]}" --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/publish_hf.sbatch"
  echo "QWEN3_AIME_SLURM_SHAPES_VALID array=0-63 completions=7680 output=$OUTPUT_ROOT"
  exit 0
fi

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Commit and push evaluator scripts before submission" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
}
[[ ! -e "$HANDOFF/SUBMISSION.json" ]] || {
  echo "Submission metadata already exists: $HANDOFF/SUBMISSION.json" >&2
  exit 2
}

preflight_job="$(
  submit_job "${COMMON[@]}" --cpus-per-task=8 --mem=64G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/preflight.sbatch"
)"
canary_job="$(
  submit_job "${COMMON[@]}" --dependency="afterok:$preflight_job" \
    --gpus=h200:1 --cpus-per-task=8 --mem=220G --time=02:00:00 \
    --requeue --signal=B:USR1@600 \
    --export=ALL,REPO_ROOT="$REPO_ROOT",TASK_ID_OVERRIDE=0,CANARY_LIMIT=1 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/run_h200.sbatch"
)"
generation_job="$(
  submit_job "${COMMON[@]}" --dependency="afterok:$canary_job" --array=0-63 \
    --gpus=h200:1 --cpus-per-task=8 --mem=220G --time=08:00:00 \
    --requeue --signal=B:USR1@600 \
    --output="$LOG_ROOT/%x-%A_%a.out" "$HANDOFF/run_h200.sbatch"
)"
audit_job="$(
  submit_job "${COMMON[@]}" --dependency="afterok:$generation_job" \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/audit.sbatch"
)"
publish_job="$(
  submit_job "${COMMON[@]}" --dependency="afterok:$audit_job" \
    --cpus-per-task=8 --mem=96G --time=04:00:00 \
    --output="$LOG_ROOT/%x-%j.out" "$HANDOFF/publish_hf.sbatch"
)"

"$PYTHON_BIN" "$HANDOFF/pipeline.py" record-submission \
  --preflight-job "$preflight_job" \
  --canary-job "$canary_job" \
  --generation-job "$generation_job" \
  --audit-job "$audit_job" \
  --publish-job "$publish_job"

echo "QWEN3_AIME_SUBMITTED preflight=$preflight_job canary=$canary_job generation=$generation_job audit=$audit_job publish=$publish_job"

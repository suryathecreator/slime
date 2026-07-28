#!/usr/bin/env bash
# Submit base first, then release the three 40K MATH-500 evaluations.
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$AXOLOTL_ROOT/.venv-vllm/bin/python}"
CONTROL="$HANDOFF/current_math500_eval.py"
EXPECTED_AXOLOTL_COMMIT="6b8f0e3314e3d162260cdc35d84741c3da163f30"
CURRENT_IDS=(
  base_8b
  40k_correct_only
  40k_weighted_tau_0p20
  40k_unmasked
)
FORTY_K_IDS=(
  40k_correct_only
  40k_weighted_tau_0p20
  40k_unmasked
)

[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == "$EXPECTED_AXOLOTL_COMMIT" ]] || {
  echo "Axolotl evaluator commit changed" >&2
  exit 2
}
[[ -z "$(git -C "$AXOLOTL_ROOT" status --porcelain)" ]] || {
  echo "Axolotl evaluator worktree is dirty" >&2
  exit 2
}

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" prepare
"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints

CONTROL_ROOT="$("$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
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

submit_job() {
  local response
  response="$(sbatch --parsable "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || {
    echo "Unexpected sbatch job id: $response" >&2
    return 2
  }
  printf '%s\n' "$response"
}

if [[ "$MODE" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/preflight_current.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --gpus=h200:1 \
    --export=ALL,EVAL_ID=base_8b \
    --output="$LOG_DIR/%x-%A_%a.out" \
    "$HANDOFF/run_current_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" \
    --export=ALL,EVAL_ID=base_8b \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/finalize_current.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --gpus=h200:1 \
    --export=ALL,EVAL_ID=40k_correct_only \
    --output="$LOG_DIR/%x-%A_%a.out" \
    "$HANDOFF/run_current_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/audit_current.sbatch"
  echo "CURRENT_SLURM_SHAPES_VALID checkpoints=4 shards=16 base_first=1"
  exit 0
fi

[[ ! -e "$CONTROL_ROOT/submission.json" ]] || {
  echo "Submission journal already exists: $CONTROL_ROOT/submission.json" >&2
  exit 2
}

preflight_job="$(
  submit_job "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/preflight_current.sbatch"
)"

base_array="$(
  submit_job "${COMMON[@]}" \
    --dependency="afterok:$preflight_job" \
    --job-name=q8-m5-base-current \
    --array=0-3 \
    --gpus=h200:1 \
    --cpus-per-task=8 \
    --mem=220G \
    --time=10-00:00:00 \
    --signal=B:USR1@600 \
    --export=ALL,EVAL_ID=base_8b \
    --output="$LOG_DIR/%x-%A_%a.out" \
    "$HANDOFF/run_current_h200.sbatch"
)"
base_finalize="$(
  submit_job "${COMMON[@]}" \
    --dependency="afterok:$base_array" \
    --job-name=q8-m5-base-current-finalize \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --export=ALL,EVAL_ID=base_8b \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/finalize_current.sbatch"
)"

array_jobs=("$base_array")
finalize_jobs=("$base_finalize")
array_assignments=(--array-job "base_8b=$base_array")
finalize_assignments=(--finalize-job "base_8b=$base_finalize")

for eval_id in "${FORTY_K_IDS[@]}"; do
  slug="${eval_id//_/-}"
  array_job="$(
    submit_job "${COMMON[@]}" \
      --dependency="afterok:$base_finalize" \
      --job-name="q8-m5-${slug}" \
      --array=0-3 \
      --gpus=h200:1 \
      --cpus-per-task=8 \
      --mem=220G \
      --time=10-00:00:00 \
      --signal=B:USR1@600 \
      --export=ALL,EVAL_ID="$eval_id" \
      --output="$LOG_DIR/%x-%A_%a.out" \
      "$HANDOFF/run_current_h200.sbatch"
  )"
  finalize_job="$(
    submit_job "${COMMON[@]}" \
      --dependency="afterok:$array_job" \
      --job-name="q8-m5-${slug}-finalize" \
      --cpus-per-task=8 \
      --mem=64G \
      --time=04:00:00 \
      --export=ALL,EVAL_ID="$eval_id" \
      --output="$LOG_DIR/%x-%j.out" \
      "$HANDOFF/finalize_current.sbatch"
  )"
  array_jobs+=("$array_job")
  finalize_jobs+=("$finalize_job")
  array_assignments+=(--array-job "$eval_id=$array_job")
  finalize_assignments+=(--finalize-job "$eval_id=$finalize_job")
done

audit_dependency="afterok:$(IFS=:; echo "${finalize_jobs[*]:1}")"
audit_job="$(
  submit_job "${COMMON[@]}" \
    --dependency="$audit_dependency" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/audit_current.sbatch"
)"

"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" \
  "${array_assignments[@]}" \
  "${finalize_assignments[@]}" \
  --audit-job "$audit_job"

echo "CURRENT_BASE_40K_SUBMITTED preflight=$preflight_job base_array=$base_array base_finalize=$base_finalize arrays=${array_jobs[*]} finalizers=${finalize_jobs[*]} audit=$audit_job"

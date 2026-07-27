#!/usr/bin/env bash
# Submit the eleven available 2K checkpoints through Axolotl's fast evaluator.
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
  exit 2
}

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
PYTHON_BIN="${PYTHON_BIN:-$AXOLOTL_ROOT/.venv/bin/python}"
CONTROL="$HANDOFF/axolotl_math500_eval.py"
SOURCE_EVAL="${SOURCE_EVAL:-$AXOLOTL_ROOT/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/eval_benchmarks/math500.jsonl}"
VARIANTS=(
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

[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "qwen3-8b-openr1-masked-sft-klone-eval" ]] || {
  echo "Wrong SLIME execution branch" >&2
  exit 2
}
git -C "$REPO_ROOT" merge-base --is-ancestor \
  88abc752730ae6ba1b3b036b99381a5132157aaa HEAD || {
  echo "Execution branch does not contain the approved SLIME handoff commit" >&2
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

"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" prepare \
  --source "$SOURCE_EVAL"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints

CONTROL_ROOT="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
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
    "$HANDOFF/preflight_axolotl.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --gpus=h200:1 \
    --export=ALL,EVAL_VARIANT=correct_only \
    --output="$LOG_DIR/%x-%A_%a.out" \
    "$HANDOFF/run_axolotl_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" \
    --export=ALL,EVAL_VARIANT=correct_only \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/finalize_axolotl.sbatch"
  sbatch --test-only "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/audit_axolotl.sbatch"
  echo "AXOLOTL_SLURM_SHAPES_VALID checkpoints=11 shards=44"
  exit 0
fi

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Commit the local evaluator provenance before submission" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
}
[[ ! -e "$CONTROL_ROOT/submission.json" ]] || {
  echo "Submission journal already exists: $CONTROL_ROOT/submission.json" >&2
  exit 2
}

preflight_job="$(
  submit_job "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/preflight_axolotl.sbatch"
)"

array_jobs=()
merge_jobs=()
array_assignments=()
merge_assignments=()
for variant in "${VARIANTS[@]}"; do
  job_slug="${variant//_/-}"
  array_job="$(
    submit_job "${COMMON[@]}" \
      --dependency="afterok:$preflight_job" \
      --job-name="q8-m5-${job_slug}" \
      --array=0-3 \
      --gpus=h200:1 \
      --cpus-per-task=8 \
      --mem=220G \
      --time=10-00:00:00 \
      --signal=B:USR1@600 \
      --export=ALL,EVAL_VARIANT="$variant" \
      --output="$LOG_DIR/%x-%A_%a.out" \
      "$HANDOFF/run_axolotl_h200.sbatch"
  )"
  merge_job="$(
    submit_job "${COMMON[@]}" \
      --dependency="afterok:$array_job" \
      --job-name="q8-m5-${job_slug}-merge" \
      --cpus-per-task=4 \
      --mem=32G \
      --time=01:00:00 \
      --export=ALL,EVAL_VARIANT="$variant" \
      --output="$LOG_DIR/%x-%j.out" \
      "$HANDOFF/finalize_axolotl.sbatch"
  )"
  array_jobs+=("$array_job")
  merge_jobs+=("$merge_job")
  array_assignments+=("--array-job" "$variant=$array_job")
  merge_assignments+=("--merge-job" "$variant=$merge_job")
done

merge_dependency="afterok:$(IFS=:; echo "${merge_jobs[*]}")"
audit_job="$(
  submit_job "${COMMON[@]}" \
    --dependency="$merge_dependency" \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/audit_axolotl.sbatch"
)"

"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" \
  "${array_assignments[@]}" \
  "${merge_assignments[@]}" \
  --audit-job "$audit_job"

echo "AXOLOTL_AVAILABLE_EVAL_SUBMITTED preflight=$preflight_job arrays=${array_jobs[*]} merges=${merge_jobs[*]} audit=$audit_job"

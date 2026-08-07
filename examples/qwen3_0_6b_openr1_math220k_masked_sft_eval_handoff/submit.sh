#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" ]] || {
  echo "Usage: $0 --test-only-shapes|--submit" >&2
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
  echo "MATH500_V5_SLURM_SHAPES_VALID checkpoints=12 shards=48"
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
[[ ! -e "$CONTROL_ROOT/submission.json" ]] || {
  echo "Submission journal already exists: $CONTROL_ROOT/submission.json" >&2
  exit 2
}

submit_job() {
  local response
  response="$(sbatch --parsable "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job id: $response" >&2; return 2; }
  printf '%s\n' "$response"
}

preflight_job="$(
  submit_job "${COMMON[@]}" \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/preflight_cpu.sbatch"
)"
canary_job="$(
  submit_job "${COMMON[@]}" \
    --dependency="afterok:$preflight_job" \
    --gpus=h200:1 \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/canary_h200.sbatch"
)"

array_jobs=()
merge_jobs=()
array_assignments=()
merge_assignments=()
for variant in "${VARIANTS[@]}"; do
  slug="${variant//_/-}"
  array_job="$(
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
  )"
  merge_job="$(
    submit_job "${COMMON[@]}" \
      --dependency="afterok:$array_job" \
      --job-name="q06-m5-${slug}-final" \
      --cpus-per-task=8 \
      --mem=64G \
      --time=01:00:00 \
      --export=ALL,EVAL_VARIANT="$variant" \
      --output="$LOG_DIR/%x-%j.out" \
      "$HANDOFF/finalize_cpu.sbatch"
  )"
  array_jobs+=("$array_job")
  merge_jobs+=("$merge_job")
  array_assignments+=(--array-job "$variant=$array_job")
  merge_assignments+=(--merge-job "$variant=$merge_job")
done

merge_dependency="afterok:$(IFS=:; echo "${merge_jobs[*]}")"
audit_job="$(
  submit_job "${COMMON[@]}" \
    --dependency="$merge_dependency" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:00:00 \
    --output="$LOG_DIR/%x-%j.out" \
    "$HANDOFF/audit_cpu.sbatch"
)"

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" \
  --canary-job "$canary_job" \
  "${array_assignments[@]}" \
  "${merge_assignments[@]}" \
  --audit-job "$audit_job"

echo "MATH500_V5_SUBMITTED preflight=$preflight_job canary=$canary_job arrays=${array_jobs[*]} merges=${merge_jobs[*]} audit=$audit_job"

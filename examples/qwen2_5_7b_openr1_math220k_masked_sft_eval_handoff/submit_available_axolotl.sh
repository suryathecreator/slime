#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" ]]
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
PYTHON_BIN="${PYTHON_BIN:-$AXOLOTL_ROOT/.venv/bin/python}"
CONTROL="$HANDOFF/axolotl_math500_eval.py"
SOURCE_EVAL="${SOURCE_EVAL:-$AXOLOTL_ROOT/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/eval_benchmarks/math500.jsonl}"
VARIANTS=(base_7b correct_only unmasked random_mask_25 random_mask_50 random_mask_70 random_mask_80 random_mask_90 inverse_tau_0p20 inverse_tau_0p05 margin_mask prob_ratio_mask)
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "qwen2.5-7b-openr1-masked-sft-2k" ]] || {
  echo "Wrong SLIME execution branch" >&2
  exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == "6b8f0e3314e3d162260cdc35d84741c3da163f30" ]]
[[ -z "$(git -C "$AXOLOTL_ROOT" status --porcelain)" ]]
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" prepare --source "$SOURCE_EVAL"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints
CONTROL_ROOT="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
LOG_DIR="$CONTROL_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"
COMMON=(--account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL)
submit_job() {
  local response
  response="$(sbatch --parsable "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]]
  printf '%s\n' "$response"
}
if [[ "$MODE" == "--test-only-shapes" ]]; then
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_axolotl.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --export=ALL,EVAL_VARIANT=base_7b --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_axolotl_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --export=ALL,EVAL_VARIANT=base_7b --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_axolotl.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_axolotl.sbatch"
  echo "QWEN25_AXOLOTL_SLURM_SHAPES_VALID checkpoints=12 shards=48"
  exit 0
fi
[[ ! -e "$CONTROL_ROOT/submission.json" ]]
preflight_job="$(submit_job "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_axolotl.sbatch")"
array_jobs=()
merge_jobs=()
array_assignments=()
merge_assignments=()
for variant in "${VARIANTS[@]}"; do
  slug="${variant//_/-}"
  array_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$preflight_job" --job-name="q25-m5-$slug" --array=0-3 --export=ALL,EVAL_VARIANT="$variant" --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_axolotl_h200.sbatch")"
  merge_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$array_job" --job-name="q25-m5-$slug-merge" --export=ALL,EVAL_VARIANT="$variant" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/finalize_axolotl.sbatch")"
  array_jobs+=("$array_job")
  merge_jobs+=("$merge_job")
  array_assignments+=(--array-job "$variant=$array_job")
  merge_assignments+=(--merge-job "$variant=$merge_job")
done
merge_dependency="afterok:$(IFS=:; echo "${merge_jobs[*]}")"
audit_job="$(submit_job "${COMMON[@]}" --dependency="$merge_dependency" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_axolotl.sbatch")"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" "${array_assignments[@]}" "${merge_assignments[@]}" --audit-job "$audit_job"
echo "QWEN25_AXOLOTL_SUBMITTED preflight=$preflight_job arrays=${array_jobs[*]} merges=${merge_jobs[*]} audit=$audit_job"

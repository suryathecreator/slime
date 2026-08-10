#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?Use --test-only-shapes or --submit}"
[[ "$MODE" == "--test-only-shapes" || "$MODE" == "--submit" ]]
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
LLAMA_GATE_PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$LLAMA_GATE_PYTHON_BIN}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$LLAMA_GATE_PYTHON_BIN}"
CONTROL="$HANDOFF/axolotl_math500_eval.py"
LLAMA_CONTROL="$HANDOFF/llama_math500_eval.py"
SOURCE_EVAL="${SOURCE_EVAL:-$AXOLOTL_ROOT/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm/eval_benchmarks/math500.jsonl}"
VARIANTS=(base_7b correct_only unmasked random_mask_25 random_mask_50 random_mask_70 random_mask_80 random_mask_90 inverse_tau_0p20 inverse_tau_0p05 margin_mask prob_ratio_mask)
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "qwen2.5-7b-openr1-masked-sft-2k" ]] || {
  echo "Wrong SLIME execution branch" >&2
  exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain -- . \
  ":(exclude)examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff")" ]] || {
  echo "SLIME has changes outside the active handoff" >&2
  exit 2
}
[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == "6b8f0e3314e3d162260cdc35d84741c3da163f30" ]]
[[ -z "$(git -C "$AXOLOTL_ROOT" status --porcelain)" ]]
[[ -x "$LLAMA_GATE_PYTHON_BIN" ]] || {
  echo "Missing pinned runtime; run $HANDOFF/setup_llama_gate_runtime.sh" >&2
  exit 2
}
[[ -f "$REPO_ROOT/checkpoints/models/nvidia--Llama-3.3-70B-Instruct-FP8--68579f675008/MODEL_READY.json" ]] || {
  echo "Missing pinned Llama model; run $HANDOFF/setup_llama_gate_runtime.sh" >&2
  exit 2
}
export PYTHON_BIN VLLM_PYTHON_BIN LLAMA_GATE_PYTHON_BIN
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
"$LLAMA_GATE_PYTHON_BIN" "$LLAMA_CONTROL" --repo-root "$REPO_ROOT" check-runtime
"$LLAMA_GATE_PYTHON_BIN" "$LLAMA_CONTROL" --repo-root "$REPO_ROOT" check-model
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
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/prepare_llama_gate.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/run_llama_gate_canary_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --array=0-3 --export=ALL,EVAL_VARIANT=base_7b --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_llama_gate_h200.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/score_llama_gate.sbatch"
  sbatch --test-only "${COMMON[@]}" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_llama_gate.sbatch"
  echo "QWEN25_AXOLOTL_SLURM_SHAPES_VALID checkpoints=12 generation_shards=48 llama_gate_shards=48"
  exit 0
fi
[[ ! -e "$CONTROL_ROOT/submission.json" ]]
GATE_ROOT="$("$LLAMA_GATE_PYTHON_BIN" "$LLAMA_CONTROL" --repo-root "$REPO_ROOT" path --field gate)"
[[ ! -e "$GATE_ROOT/submission.json" ]]
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
gate_prepare_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$audit_job" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/prepare_llama_gate.sbatch")"
gate_canary_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$gate_prepare_job" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/run_llama_gate_canary_h200.sbatch")"
gate_array_jobs=()
gate_array_assignments=()
for variant in "${VARIANTS[@]}"; do
  slug="${variant//_/-}"
  gate_array_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$gate_canary_job" --job-name="q25-gate-$slug" --array=0-3 --export=ALL,EVAL_VARIANT="$variant" --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_llama_gate_h200.sbatch")"
  gate_array_jobs+=("$gate_array_job")
  gate_array_assignments+=(--gate-array-job "$variant=$gate_array_job")
done
gate_dependency="afterok:$(IFS=:; echo "${gate_array_jobs[*]}")"
score_job="$(submit_job "${COMMON[@]}" --dependency="$gate_dependency" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/score_llama_gate.sbatch")"
strict_audit_job="$(submit_job "${COMMON[@]}" --dependency="afterok:$score_job" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_llama_gate.sbatch")"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --preflight-job "$preflight_job" "${array_assignments[@]}" "${merge_assignments[@]}" --audit-job "$audit_job"
"$LLAMA_GATE_PYTHON_BIN" "$LLAMA_CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --prepare-job "$gate_prepare_job" --canary-job "$gate_canary_job" "${gate_array_assignments[@]}" \
  --score-job "$score_job" --audit-job "$strict_audit_job"
echo "QWEN25_AXOLOTL_SUBMITTED preflight=$preflight_job generation_arrays=${array_jobs[*]} merges=${merge_jobs[*]} native_audit=$audit_job gate_prepare=$gate_prepare_job gate_canary=$gate_canary_job gate_arrays=${gate_array_jobs[*]} score=$score_job strict_audit=$strict_audit_job"

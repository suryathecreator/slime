#!/usr/bin/env bash
set -euo pipefail

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
CONTROL="$HANDOFF/llama_math500_eval.py"
PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
export REPO_ROOT PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

GATE_ROOT="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field gate)"
LOG_DIR="$GATE_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"
COMMON=(--parsable --account=raivn-ckpt --partition=ckpt-all --qos=ckpt --requeue --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL)

submit_job() {
  local response
  response="$(sbatch "${COMMON[@]}" "$@")"
  response="${response%%;*}"
  [[ "$response" =~ ^[0-9]+$ ]]
  printf '%s\n' "$response"
}

prepare_job="$(submit_job --output="$LOG_DIR/%x-%j.out" "$HANDOFF/prepare_llama_gate.sbatch")"
canary_job="$(submit_job --dependency="afterok:$prepare_job" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/run_llama_gate_canary_h200.sbatch")"
mapfile -t variants < <("$PYTHON_BIN" -c 'import json,sys; print("\n".join(x["variant"] for x in json.load(open(sys.argv[1]))["checkpoints"]))' "$HANDOFF/available_eval_manifest.json")
[[ "${#variants[@]}" -eq 12 ]]
gate_jobs=()
gate_assignments=()
for variant in "${variants[@]}"; do
  slug="${variant//_/-}"
  gate_job="$(submit_job --dependency="afterok:$canary_job" --job-name="q25-gate-v2-$slug" --array=0-3 --export=ALL,EVAL_VARIANT="$variant" --output="$LOG_DIR/%x-%A_%a.out" "$HANDOFF/run_llama_gate_h200.sbatch")"
  gate_jobs+=("$gate_job")
  gate_assignments+=(--gate-array-job "$variant=$gate_job")
done
gate_dependency="afterok:$(IFS=:; echo "${gate_jobs[*]}")"
score_job="$(submit_job --dependency="$gate_dependency" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/score_llama_gate.sbatch")"
audit_job="$(submit_job --dependency="afterok:$score_job" --output="$LOG_DIR/%x-%j.out" "$HANDOFF/audit_llama_gate.sbatch")"
"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" record-submission \
  --prepare-job "$prepare_job" --canary-job "$canary_job" "${gate_assignments[@]}" \
  --score-job "$score_job" --audit-job "$audit_job"
echo "LLAMA_GATE_V2_SUBMITTED prepare=$prepare_job canary=$canary_job gate_arrays=${gate_jobs[*]} score=$score_job audit=$audit_job"

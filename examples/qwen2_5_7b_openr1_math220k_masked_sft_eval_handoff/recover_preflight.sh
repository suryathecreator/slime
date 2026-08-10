#!/usr/bin/env bash
set -euo pipefail

FAILED_JOB="${1:?Pass the failed preflight job ID}"
HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HANDOFF/../.." && pwd)"
LLAMA_GATE_PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$LLAMA_GATE_PYTHON_BIN}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$LLAMA_GATE_PYTHON_BIN}"
CONTROL="$HANDOFF/llama_math500_eval.py"
NATIVE_CONTROL="$HANDOFF/axolotl_math500_eval.py"
export PYTHON_BIN VLLM_PYTHON_BIN LLAMA_GATE_PYTHON_BIN REPO_ROOT
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONTROL_ROOT="$("$PYTHON_BIN" "$NATIVE_CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
SUBMISSION="$CONTROL_ROOT/submission.json"
LOG_DIR="$CONTROL_ROOT/slurm_logs"
recorded_failed="$(SUBMISSION="$SUBMISSION" "$PYTHON_BIN" -c 'import json, os; print(json.load(open(os.environ["SUBMISSION"]))["preflight_job"])')"
[[ "$recorded_failed" == "$FAILED_JOB" ]]
mapfile -t array_jobs < <(SUBMISSION="$SUBMISSION" "$PYTHON_BIN" -c 'import json, os; value=json.load(open(os.environ["SUBMISSION"])); print("\n".join(value["array_jobs"].values()))')
[[ "${#array_jobs[@]}" -eq 12 ]]

response="$(sbatch --parsable --account=raivn-ckpt --partition=ckpt-all --qos=ckpt \
  --requeue --mail-user=suryadv@cs.washington.edu --mail-type=END,FAIL \
  --output="$LOG_DIR/%x-%j.out" "$HANDOFF/preflight_axolotl.sbatch")"
replacement_job="${response%%;*}"
[[ "$replacement_job" =~ ^[0-9]+$ ]]
for array_job in "${array_jobs[@]}"; do
  scontrol update JobId="$array_job" Dependency="afterok:$replacement_job"
done
"$LLAMA_GATE_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" \
  record-preflight-recovery --failed-job "$FAILED_JOB" --replacement-job "$replacement_job"
echo "QWEN25_PREFLIGHT_RECOVERED failed=$FAILED_JOB replacement=$replacement_job arrays=${array_jobs[*]}"

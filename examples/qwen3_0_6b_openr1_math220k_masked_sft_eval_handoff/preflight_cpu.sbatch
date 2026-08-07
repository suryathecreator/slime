#!/usr/bin/env bash
#SBATCH --job-name=q06-m500-v5-preflight
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/math500_eval.py"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
VERIFY_STATUS="$CONTROL_ROOT/checkpoint_verification.json"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

"$RUNTIME_PYTHON" -c '
import importlib.metadata as m
expected={"vllm":"0.10.2","torch":"2.8.0","transformers":"4.57.6","math-verify":"0.9.0"}
actual={name:m.version(name) for name in expected}
assert actual == expected, (actual, expected)
print("MATH500_V5_RUNTIME_EXACT", actual, flush=True)
'
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints \
  --full --status "$VERIFY_STATUS"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" validate-gold
"$RUNTIME_PYTHON" -m pytest -q \
  examples/qwen3_8b_32b_full_repro/tests/test_math500_last_boxed_scorer_v5.py
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" mark-preflight \
  --job-id "${SLURM_JOB_ID:?}"
echo "MATH500_V5_PREFLIGHT_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"

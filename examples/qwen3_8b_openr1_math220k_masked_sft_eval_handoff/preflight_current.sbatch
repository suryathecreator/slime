#!/usr/bin/env bash
#SBATCH --job-name=q8b-m500-current-preflight
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/current_math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$AXOLOTL_ROOT/.venv-vllm/bin/python}"
CONTROL_ROOT="$("$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
MODEL_DIR="$("$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --eval-id base_8b)"
TOKENIZER_DIR="$("$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer)"
SMOKE_FILE="$("$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field smoke)"
SMOKE_OUTPUT="$CONTROL_ROOT/smoke_base_8b"
VERIFY_STATUS="$CONTROL_ROOT/checkpoint_verification.json"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SCRUBBED_CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/axolotl-masked-sft}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRUBBED_CACHE_ROOT/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRUBBED_CACHE_ROOT/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRUBBED_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRUBBED_CACHE_ROOT/triton}"
export TMPDIR="/tmp/${USER:-suryadv}/q8b-current-${SLURM_JOB_ID}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$REPO_ROOT:$AXOLOTL_ROOT/scripts/openr1_axolotl:${PYTHONPATH:-}"

mkdir -p "$CONTROL_ROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$REPO_ROOT"

[[ "$(git -C "$AXOLOTL_ROOT" rev-parse HEAD)" == "6b8f0e3314e3d162260cdc35d84741c3da163f30" ]] || {
  echo "Axolotl evaluator commit changed" >&2
  exit 2
}

child_pid=""
requeue_job() {
  trap - USR1
  echo "CURRENT_PREFLIGHT_REQUEUE job=$SLURM_JOB_ID at=$(date -Is)"
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" prepare
"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints \
  --full --status "$VERIFY_STATUS" &
child_pid="$!"
wait "$child_pid"
child_pid=""

"$VLLM_PYTHON_BIN" "$AXOLOTL_ROOT/scripts/openr1_axolotl/check_vllm.py" \
  --require-cuda
"$VLLM_PYTHON_BIN" -c '
import importlib.metadata as m
expected={"vllm":"0.10.2","torch":"2.8.0","transformers":"4.57.6","math-verify":"0.9.0"}
actual={name:m.version(name) for name in expected}
assert actual == expected, (actual, expected)
from examples.qwen3_8b_32b_full_repro.math500_scorer_v4 import SCORER_VERSION
assert SCORER_VERSION == "math500_strict_boxed_units_scorer_v4", SCORER_VERSION
print("CURRENT_RUNTIME_EXACT", actual, SCORER_VERSION, flush=True)
'

if [[ ! -f "$SMOKE_OUTPUT/metrics.json" ]]; then
  "$VLLM_PYTHON_BIN" \
    "$AXOLOTL_ROOT/scripts/openr1_axolotl/evaluate_math_vllm.py" \
    --eval_file "$SMOKE_FILE" \
    --output_dir "$SMOKE_OUTPUT" \
    --variant preflight_base_8b \
    --model_name "$TOKENIZER_DIR" \
    --adapter_or_model_dir "$MODEL_DIR" \
    --max_tokens 1 \
    --max_model_len 32768 \
    --chunk_size 1 \
    --max_num_seqs 1 \
    --gpu_memory_utilization 0.92 \
    --require_math_verify &
  child_pid="$!"
  wait "$child_pid"
  child_pid=""
fi

"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" mark-ready \
  --job-id "${SLURM_JOB_ID:?}"
echo "CURRENT_PREFLIGHT_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"

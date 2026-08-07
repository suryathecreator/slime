#!/usr/bin/env bash
#SBATCH --job-name=q06-m500-v5-canary
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
MODEL_DIR="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --variant base_0p6b)"
TOKENIZER_DIR="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer)"
SMOKE_FILE="$CONTROL_ROOT/eval_benchmarks/smoke.jsonl"
CANARY_OUTPUT="$CONTROL_ROOT/canary"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SCRUBBED_CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q06-math500-v5}"
TASK_CACHE_ROOT="$SCRUBBED_CACHE_ROOT/tasks/job_${SLURM_JOB_ID:?}/canary"
export XDG_CACHE_HOME="$TASK_CACHE_ROOT/xdg"
export VLLM_CACHE_ROOT="$TASK_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$TASK_CACHE_ROOT/triton"
TMPDIR="$(mktemp -d "/tmp/q06m5-${SLURM_JOB_ID:?}.XXXXXX")"
export TMPDIR
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$REPO_ROOT:$AXOLOTL_ROOT/scripts/openr1_axolotl:${PYTHONPATH:-}"

mkdir -p "$CANARY_OUTPUT" "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "MATH500_V5_CACHE_NAMESPACE variant=canary_base_0p6b root=$TASK_CACHE_ROOT tmp=$TMPDIR"
cd "$REPO_ROOT"

child_pid=""
requeue_job() {
  trap - USR1
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" check-preflight
if [[ ! -f "$CANARY_OUTPUT/metrics.json" ]]; then
  "$RUNTIME_PYTHON" "$AXOLOTL_ROOT/scripts/openr1_axolotl/evaluate_math_vllm.py" \
    --eval_file "$SMOKE_FILE" \
    --output_dir "$CANARY_OUTPUT" \
    --variant canary_base_0p6b \
    --model_name "$TOKENIZER_DIR" \
    --adapter_or_model_dir "$MODEL_DIR" \
    --max_tokens 32768 \
    --max_model_len 32768 \
    --chunk_size 1 \
    --max_num_seqs 1 \
    --gpu_memory_utilization 0.92 \
    --require_math_verify &
  child_pid="$!"
  wait "$child_pid"
  child_pid=""
fi
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" mark-canary \
  --job-id "${SLURM_JOB_ID:?}"
echo "MATH500_V5_CANARY_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"

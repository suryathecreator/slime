#!/usr/bin/env bash
#SBATCH --job-name=q8b-m500-axo
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/axolotl_math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
PYTHON_BIN="${PYTHON_BIN:-$AXOLOTL_ROOT/.venv/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$AXOLOTL_ROOT/.venv-vllm/bin/python}"
: "${EVAL_VARIANT:?Set EVAL_VARIANT to one approved available variant}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?Run this worker as the 0-3 shard array}"
[[ "$SHARD_INDEX" =~ ^[0-3]$ ]] || {
  echo "Shard index must be 0..3: $SHARD_INDEX" >&2
  exit 2
}

CONTROL_ROOT="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
MODEL_DIR="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --variant "$EVAL_VARIANT")"
MODEL_MANIFEST="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field manifest --variant "$EVAL_VARIANT")"
VARIANT_OUTPUT="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field output --variant "$EVAL_VARIANT")"
EVAL_FILE="$("$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" path --field shard --shard-index "$SHARD_INDEX")"
SHARD_OUTPUT="$VARIANT_OUTPUT/shards/shard_$(printf '%02d' "$SHARD_INDEX")"

"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" check-preflight
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "Missing Full SFT checkpoint config: $MODEL_DIR/config.json" >&2
  exit 2
}
"$PYTHON_BIN" "$HANDOFF/checkpoint_manifest.py" identity \
  --manifest "$MODEL_MANIFEST" >/dev/null

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SCRUBBED_CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/axolotl-masked-sft}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRUBBED_CACHE_ROOT/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRUBBED_CACHE_ROOT/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRUBBED_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRUBBED_CACHE_ROOT/triton}"
export TMPDIR="${TMPDIR:-$CONTROL_ROOT/tmp}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$AXOLOTL_ROOT/scripts/openr1_axolotl:${PYTHONPATH:-}"

mkdir -p "$SHARD_OUTPUT" "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$AXOLOTL_ROOT"

"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" repair-shard \
  --eval-file "$EVAL_FILE" --output-dir "$SHARD_OUTPUT"

child_pid=""
requeue_eval() {
  trap - USR1
  echo "AXOLOTL_EVAL_REQUEUE variant=$EVAL_VARIANT shard=$SHARD_INDEX job=$SLURM_JOB_ID at=$(date -Is)"
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_eval USR1

"$VLLM_PYTHON_BIN" scripts/openr1_axolotl/evaluate_math_vllm.py \
  --eval_file "$EVAL_FILE" \
  --output_dir "$SHARD_OUTPUT" \
  --variant "$EVAL_VARIANT" \
  --model_name "$MODEL_DIR" \
  --adapter_or_model_dir "$MODEL_DIR" \
  --max_tokens 32768 \
  --max_model_len 32768 \
  --chunk_size 32 \
  --max_num_seqs 8 \
  --gpu_memory_utilization 0.92 \
  --require_math_verify &
child_pid="$!"
wait "$child_pid"
child_pid=""

"$PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" repair-shard \
  --eval-file "$EVAL_FILE" --output-dir "$SHARD_OUTPUT"
echo "AXOLOTL_EVAL_SHARD_COMPLETE variant=$EVAL_VARIANT shard=$SHARD_INDEX job=$SLURM_JOB_ID at=$(date -Is)"

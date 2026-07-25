#!/bin/bash
#SBATCH --job-name=q8b-masksft-m500
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gpus=h200:1
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the pulled Slime repository}"
HANDOFF_DIR="${REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
EVALUATOR="${REPO_ROOT}/examples/qwen3_8b_32b_full_repro/evaluate_math500.py"
: "${MODEL_DIR:?Set MODEL_DIR to the transferred Hugging Face checkpoint}"
: "${MODEL_MANIFEST:?Set MODEL_MANIFEST to its checkpoint manifest JSON}"
: "${EVAL_STAGE:?Set EVAL_STAGE to the checkpoint id in checkpoint_sources.json}"
: "${EVAL_OUTPUT_DIR:?Set a stable shared-filesystem EVAL_OUTPUT_DIR}"
: "${EVAL_PYTHON:?Set EVAL_PYTHON to the Python executable with vLLM and eval dependencies}"

[[ -x "${EVAL_PYTHON}" ]] || {
  echo "EVAL_PYTHON is not executable: ${EVAL_PYTHON}" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/config.json" ]] || {
  echo "Transferred model is incomplete: ${MODEL_DIR}" >&2
  exit 2
}
[[ -f "${MODEL_MANIFEST}" ]] || {
  echo "Missing checkpoint manifest: ${MODEL_MANIFEST}" >&2
  exit 2
}
"${EVAL_PYTHON}" -c \
  'import vllm; assert vllm.__version__ == "0.24.0", vllm.__version__'

mkdir -p "${EVAL_OUTPUT_DIR}/problems"
child_pid=""
requeue_eval() {
  trap - USR1
  echo "EVAL_REQUEUE stage=${EVAL_STAGE} job=${SLURM_JOB_ID} at=$(date -Is)"
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_eval USR1

"${EVAL_PYTHON}" "${HANDOFF_DIR}/checkpoint_manifest.py" verify \
  --manifest "${MODEL_MANIFEST}" \
  --checkpoint "${MODEL_DIR}"
CHECKPOINT_ID="$(
  "${EVAL_PYTHON}" "${HANDOFF_DIR}/checkpoint_manifest.py" identity \
    --manifest "${MODEL_MANIFEST}"
)"

"${EVAL_PYTHON}" "${EVALUATOR}" shard \
  --model "${MODEL_DIR}" \
  --checkpoint-manifest-sha256 "${CHECKPOINT_ID}" \
  --stage "${EVAL_STAGE}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --shard-index 0 \
  --num-shards 1 \
  --dataset HuggingFaceH4/MATH-500 \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --temperature 0.0 \
  --top-p 1.0 \
  --top-k -1 \
  --max-new-tokens 32768 \
  --target-context 32768 \
  --safety-margin 64 \
  --seed 1234 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 16384 \
  --stop-token-ids 151643 151645 \
  --resume &
child_pid="$!"
wait "${child_pid}"
child_pid=""

"${EVAL_PYTHON}" "${EVALUATOR}" aggregate \
  --stage "${EVAL_STAGE}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --expected 500 \
  --bootstrap-seed 1234
"${EVAL_PYTHON}" -c \
  'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); (p.parent/"summary.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest()+"\n")' \
  "${EVAL_OUTPUT_DIR}/summary.json"
"${EVAL_PYTHON}" "${HANDOFF_DIR}/build_result_bundle.py" \
  --evaluation-dir "${EVAL_OUTPUT_DIR}" \
  --stage "${EVAL_STAGE}" \
  --checkpoint-manifest "${MODEL_MANIFEST}" \
  --eval-policy "${HANDOFF_DIR}/eval_policy.json" \
  --output-dir "${EVAL_OUTPUT_DIR}/bundle"
echo "EVAL_AND_BUNDLE_COMPLETE stage=${EVAL_STAGE} at=$(date -Is)"

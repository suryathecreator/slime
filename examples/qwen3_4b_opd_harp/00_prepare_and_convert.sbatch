#!/bin/bash
#SBATCH --job-name=q3-4b-harp-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}" "${MODEL_ROOT}" "${HARP_EVAL_SITE}" "${VLLM_CACHE_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"
echo "Preparing Qwen3-4B/HARP V2 experiment on $(hostname) at $(date)"
{
  find examples/qwen3_4b_opd_harp -type f ! -path '*/__pycache__/*' -print
  printf '%s\n' \
    examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch \
    examples/qwen3_8b_opd_tillicum/container_exec.sh \
    examples/qwen3_8b_opd_tillicum/env.sh \
    examples/qwen3_8b_opd_tillicum/write_opd_trained_manifest.py \
    scripts/models/qwen3-4B.sh
} | sort -u | xargs sha256sum >"${SOURCE_MANIFEST}"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0

"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
mkdir -p "${HARP_EVAL_SITE}" "${MODEL_ROOT}" "${DATA_ROOT}" "${VLLM_CACHE_ROOT}"
export PYTHONPATH="${HARP_EVAL_SITE}:${VLLM_EVAL_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(find "${VLLM_EVAL_SITE}" -path "*/nvidia/*/lib" -type d -printf "%p:" 2>/dev/null || true)${LD_LIBRARY_PATH:-}"

if ! python3 -c "import math_verify, antlr4, sympy" >/dev/null 2>&1; then
  python3 -m pip install --upgrade --target "${HARP_EVAL_SITE}" ${HARP_EVAL_INSTALL_SPECS}
fi
python3 - <<"PY"
import importlib.metadata as metadata
expected = {
    "vllm": "0.24.0",
    "transformers": "5.13.0",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
    "sympy": "1.14.0",
}
for package, wanted in expected.items():
    found = metadata.version(package)
    print(package, found)
    if found != wanted:
        raise SystemExit(f"Dependency mismatch for {package}: expected {wanted}, found {found}")
PY

python3 examples/qwen3_4b_opd_harp/prepare_experiment.py \
  --harp-source "${HARP_SOURCE_CACHE}" \
  --cached-harp-source "${HARP_EXISTING_SOURCE}" \
  --harp-output "${HARP_JSONL}" \
  --opd-source-1k "${SOURCE_OPD_1K_JSONL}" \
  --opd-source-4k "${SOURCE_OPD_4K_JSONL}" \
  --opd-output-1k "${CLEANED_OPD_1K_JSONL}" \
  --opd-output-4k "${CLEANED_OPD_4K_JSONL}" \
  --metadata "${HARP_PREP_METADATA}"

if [[ ! -f "${STUDENT_HF_DIR}/config.json" ]]; then
  hf download "${STUDENT_HF_REPO}" --local-dir "${STUDENT_HF_DIR}"
fi
if [[ ! -f "${TEACHER_HF_DIR}/config.json" ]]; then
  hf download "${TEACHER_HF_REPO}" --local-dir "${TEACHER_HF_DIR}"
fi

python3 examples/qwen3_8b_opd_tillicum/validate_qwen3_generation.py \
  --tokenizer "${STUDENT_HF_DIR}" \
  --im-end-token-id "${QWEN3_IM_END_TOKEN_ID}" \
  --endoftext-token-id "${QWEN3_ENDOFTEXT_TOKEN_ID}"
python3 examples/qwen3_8b_opd_tillicum/validate_qwen3_generation.py \
  --tokenizer "${TEACHER_HF_DIR}" \
  --im-end-token-id "${QWEN3_IM_END_TOKEN_ID}" \
  --endoftext-token-id "${QWEN3_ENDOFTEXT_TOKEN_ID}"

if [[ ! -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
  source "${STUDENT_MODEL_ARGS_SCRIPT}"
  mkdir -p "${STUDENT_TORCH_DIST_DIR}"
  PYTHONPATH="/root/Megatron-LM:${SLIME_REPO_ROOT}:${PYTHONPATH:-}" \
  torchrun --nproc_per_node=4 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF_DIR}" \
    --save "${STUDENT_TORCH_DIST_DIR}"
fi

test -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"
test -f "${HARP_PREP_METADATA}"
test -f "${HARP_JSONL}"
'

echo "Preparation complete at $(date)"

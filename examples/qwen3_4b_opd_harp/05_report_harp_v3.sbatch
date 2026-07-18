#!/bin/bash
#SBATCH --job-name=q3-harp-v3-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}" "${HARP_V3_RESCORE_DIR}" "${HARP_V3_CORRECTION_REPORT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${HARP_V3_SOURCE_MANIFEST}"
test -s "${EVAL_4B_BASE_DIR}/predictions.jsonl"
test -s "${EVAL_4B_BASE_DIR}/metrics.json"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0
"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export PYTHONPATH="${HARP_EVAL_SITE}:${HARP_EXAMPLE_DIR}:${PYTHONPATH:-}"
python3 examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py rescore \
  --data "${HARP_JSONL}" \
  --predictions "${EVAL_4B_BASE_DIR}/predictions.jsonl" \
  --output-dir "${HARP_V3_RESCORE_DIR}" \
  --scorer-version harp_answer_v3 \
  --exclusion-manifest "${HARP_EXCLUSION_MANIFEST}" \
  --max-tokens "${VLLM_EVAL_MAX_RESPONSE_LEN}"
python3 examples/qwen3_4b_opd_harp/report_harp_v3.py \
  --v2-dir "${EVAL_4B_BASE_DIR}" \
  --v3-dir "${HARP_V3_RESCORE_DIR}" \
  --output-dir "${HARP_V3_CORRECTION_REPORT_DIR}"
'

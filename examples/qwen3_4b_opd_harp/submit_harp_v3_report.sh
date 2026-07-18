#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}"

for prerequisite in "${HARP_JSONL}" "${EVAL_4B_BASE_DIR}/predictions.jsonl" "${EVAL_4B_BASE_DIR}/metrics.json" "${HARP_EXCLUSION_MANIFEST}"; do
  test -s "${prerequisite}" || { echo "Missing correction-report prerequisite: ${prerequisite}" >&2; exit 1; }
done

printf '%s\n' \
  examples/qwen3_4b_opd_harp/harp_answer_v2.py \
  examples/qwen3_4b_opd_harp/harp_answer_v3.py \
  examples/qwen3_4b_opd_harp/harp_exclusions_v3.json \
  examples/qwen3_4b_opd_harp/harp_official/latex_answer_check.py \
  examples/qwen3_4b_opd_harp/harp_official/parsing_lib.py \
  examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py \
  examples/qwen3_4b_opd_harp/report_harp_v3.py \
  examples/qwen3_4b_opd_harp/05_report_harp_v3.sbatch \
  examples/qwen3_4b_opd_harp/env.sh \
  examples/qwen3_8b_opd_tillicum/container_exec.sh \
  | xargs sha256sum >"${HARP_V3_SOURCE_MANIFEST}"

# Slurm may purge a completed source job before this report is submitted, in
# which case an afterok dependency is rejected. The required merged artifacts
# and hashes above are the completion gate, so this report can be submitted
# directly once those checks pass.
jid=$(sbatch --parsable -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1 --cpus-per-task=8 --mem="${SLURM_MEM_REPORT}" --time=02:00:00 --job-name=q3-harp-v3-report --export=ALL examples/qwen3_4b_opd_harp/05_report_harp_v3.sbatch)
submission_log="${SLURM_LOG_DIR}/submit_harp_v3_report_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=$(git rev-parse HEAD)"
  echo "source_job=177135"
  echo "source_predictions_sha256=$(sha256sum "${EVAL_4B_BASE_DIR}/predictions.jsonl" | awk '{print $1}')"
  echo "report_job=${jid}"
  echo "source_manifest=${HARP_V3_SOURCE_MANIFEST}"
  echo "output_dir=${HARP_V3_CORRECTION_REPORT_DIR}"
} | tee "${submission_log}"

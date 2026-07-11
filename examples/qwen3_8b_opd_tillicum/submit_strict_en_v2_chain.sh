#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/strict_en_v2_env.sh"
cd "${SLIME_REPO_ROOT}"

mkdir -p "${SLURM_LOG_DIR}"
for required_path in \
  "${SLIME_SIF}" \
  "${STUDENT_HF_DIR}/config.json" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing preserved prerequisite for strict-v2 submission: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -d "${VLLM_EVAL_SITE}/vllm" && ! -x "${VLLM_EVAL_VENV}/bin/python" ]]; then
  echo "Missing preserved vLLM eval environment under ${VLLM_EVAL_SITE} or ${VLLM_EVAL_VENV}" >&2
  exit 1
fi
SBATCH_ONE=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1)

jid_setup="$(sbatch --parsable "${SBATCH_ONE[@]}" --time=06:00:00 --job-name=slime-qwen3-strict-lang --cpus-per-task=8 --export=ALL examples/qwen3_8b_opd_tillicum/11_setup_strict_language_env.sbatch)"
jid_clean="$(sbatch --parsable "${SBATCH_ONE[@]}" --dependency=afterok:${jid_setup} --time=24:00:00 --job-name=slime-qwen3-strict-data --cpus-per-task=8 --export=ALL examples/qwen3_8b_opd_tillicum/12_prepare_strict_english_data.sbatch)"
jid_dispatch="$(sbatch --parsable "${SBATCH_ONE[@]}" --dependency=afterok:${jid_clean} --time=01:00:00 --job-name=slime-qwen3-strict-dispatch --cpus-per-task=4 --export=ALL examples/qwen3_8b_opd_tillicum/13_dispatch_strict_en_chain.sbatch)"

submit_log="${SLURM_LOG_DIR}/submit_strict_en_v2_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "experiment=${STRICT_EN_EXPERIMENT_LABEL}"
  echo "implementation_commit=${STRICT_EN_IMPLEMENTATION_COMMIT:-unknown}"
  echo "language_setup=${jid_setup}"
  echo "strict_clean=${jid_clean}"
  echo "dynamic_dispatch=${jid_dispatch}"
  echo "dependency_policy=setup_afterok_clean_afterok_dispatch_then_strict_afterok_training_chain"
  echo "max_h200_at_once=4"
  echo "overall_dataset=${STRICT_EN_DATASET_DIR}"
  echo "sft_data=${SFT_PARQUET}"
  echo "opd_reserve=${CLEANED_OPD_RESERVE_JSONL}"
  echo "opd_1k=${CLEANED_OPD_1K_JSONL}"
  echo "opd_continuation=${CLEANED_OPD_4K_JSONL}"
  echo "metadata=${SPLIT_METADATA}"
  echo "base_eval=reused_165035"
  echo "old_jobs_cleaned=167285,167286,167287,167288,167289,167290"
  echo "old_artifacts_deleted=none"
} | tee "${submit_log}"

echo "Submitted setup=${jid_setup} clean=${jid_clean} dispatch=${jid_dispatch}"

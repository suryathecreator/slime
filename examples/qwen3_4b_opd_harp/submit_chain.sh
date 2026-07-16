#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}"

for prerequisite in \
  "${SLIME_SIF}" \
  "${TEACHER_HF_DIR}/config.json" \
  "${VLLM_EVAL_SITE}/vllm" \
  "${SOURCE_OPD_1K_JSONL}" \
  "${SOURCE_OPD_4K_JSONL}" \
  "${HARP_EXISTING_SOURCE}"; do
  if [[ ! -e "${prerequisite}" ]]; then
    echo "Missing submission prerequisite: ${prerequisite}" >&2
    exit 1
  fi
done

SBATCH_GPU4=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4 --cpus-per-task=32)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1 --cpus-per-task=8)
submission_log="${SLURM_LOG_DIR}/submit_qwen3_4b_harp_v2_$(date +%Y%m%d_%H%M%S).txt"

jid_setup=$(sbatch --parsable "${SBATCH_GPU4[@]}" --time=06:00:00 --job-name=q3-4b-harp-setup --export=ALL examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch)
jid_tune4=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_setup}" --time=24:00:00 --job-name=q3-4b-harp-tune --export="ALL,EVAL_MODE=tune,EVAL_MODEL_DIR=${STUDENT_HF_DIR},EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_4B_BASE_DIR}/tuning,EVAL_TUNING_FILE=${EVAL_4B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)
jid_eval4=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_tune4}" --time=24:00:00 --job-name=q3-4b-harp-base --export="ALL,EVAL_MODE=eval,EVAL_MODEL_DIR=${STUDENT_HF_DIR},EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_4B_BASE_DIR}/shards,EVAL_TUNING_FILE=${EVAL_4B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)
jid_tune32=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_eval4}" --time=24:00:00 --job-name=q3-32b-harp-tune --export="ALL,EVAL_MODE=tune,EVAL_MODEL_DIR=${TEACHER_HF_DIR},EVAL_TOKENIZER_DIR=${TEACHER_HF_DIR},EVAL_STAGE_DIR=${EVAL_32B_BASE_DIR}/tuning,EVAL_TUNING_FILE=${EVAL_32B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)
jid_eval32=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_tune32}" --time=24:00:00 --job-name=q3-32b-harp-base --export="ALL,EVAL_MODE=eval,EVAL_MODEL_DIR=${TEACHER_HF_DIR},EVAL_TOKENIZER_DIR=${TEACHER_HF_DIR},EVAL_STAGE_DIR=${EVAL_32B_BASE_DIR}/shards,EVAL_TUNING_FILE=${EVAL_32B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)

jid_base_report=$(sbatch --parsable "${SBATCH_REPORT[@]}" --dependency="afterok:${jid_eval32}" --time=01:00:00 --job-name=q3-harp-base-report --export=ALL,REPORT_MODE=base examples/qwen3_4b_opd_harp/04_report_harp.sbatch)
jid_opd_tune=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_base_report}" --time=24:00:00 --job-name=q3-4b-opd-tune --export=ALL examples/qwen3_4b_opd_harp/02_tune_opd.sbatch)
jid_opd1=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_opd_tune}" --time=24:00:00 --job-name=q3-4b-opd-001024 --export=ALL,OPD_STAGE=1k examples/qwen3_4b_opd_harp/03_run_opd.sbatch)
jid_eval1=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_opd1}" --time=24:00:00 --job-name=q3-4b-harp-opd1k --export="ALL,EVAL_MODE=eval,EVAL_MODEL_DIR=${OPD1_HF_DIR}/iter_0000007,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD1K_DIR}/shards,EVAL_TUNING_FILE=${EVAL_4B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)
jid_opd5=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_eval1}" --time=24:00:00 --job-name=q3-4b-opd-005120 --export=ALL,OPD_STAGE=5k examples/qwen3_4b_opd_harp/03_run_opd.sbatch)
jid_eval5=$(sbatch --parsable "${SBATCH_GPU4[@]}" --dependency="afterok:${jid_opd5}" --time=24:00:00 --job-name=q3-4b-harp-opd5k --export="ALL,EVAL_MODE=eval,EVAL_MODEL_DIR=${OPD_CONT_HF_DIR}/iter_0000039,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD5120_DIR}/shards,EVAL_TUNING_FILE=${EVAL_4B_BASE_DIR}/tuning/selected.json" examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch)
jid_final_report=$(sbatch --parsable "${SBATCH_REPORT[@]}" --dependency="afterok:${jid_eval5}" --time=01:00:00 --job-name=q3-harp-final-report --export=ALL,REPORT_MODE=final examples/qwen3_4b_opd_harp/04_report_harp.sbatch)

{
  echo "experiment=qwen3_4b_posttrained_harp_v2_opd5120"
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=$(git rev-parse HEAD)"
  echo "setup=${jid_setup}"
  echo "qwen3_4b_eval_tune=${jid_tune4}"
  echo "qwen3_4b_base_eval=${jid_eval4}"
  echo "qwen3_32b_eval_tune=${jid_tune32}"
  echo "qwen3_32b_base_eval=${jid_eval32}"
  echo "base_report=${jid_base_report}"
  echo "opd_throughput_tune=${jid_opd_tune}"
  echo "opd_001024_train=${jid_opd1}"
  echo "opd_001024_eval=${jid_eval1}"
  echo "opd_005120_train=${jid_opd5}"
  echo "opd_005120_eval=${jid_eval5}"
  echo "final_report=${jid_final_report}"
  echo "dependency_policy=strict_serial_gpu_afterok"
  echo "max_concurrent_gpu_count=4"
  echo "harp_seed=42"
  echo "harp_subset_sha256=30a43db8e669d7a6cc8fbb8a93a84018d5440b110632372467183d707a30c5ab"
  echo "prompt_contract=qwen3_harp_boxed_v1"
  echo "output_root=${OUTPUT_ROOT}"
  echo "source_manifest=${SOURCE_MANIFEST}"
} | tee "${submission_log}"

echo "Submitted Qwen3-4B/HARP V2 chain; metadata: ${submission_log}"

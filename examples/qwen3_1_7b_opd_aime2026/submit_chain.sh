#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}"

for prerequisite in \
  "${SLIME_SIF}" \
  "${VLLM_EVAL_SITE}/vllm" \
  "${SOURCE_OPD_1K_JSONL}" \
  "${SOURCE_OPD_4K_JSONL}"; do
  if [[ ! -e "${prerequisite}" ]]; then
    echo "Missing submission prerequisite: ${prerequisite}" >&2
    exit 1
  fi
done

SBATCH_GPU4=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4 --cpus-per-task=32)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1 --cpus-per-task=8)
submission_log="${SLURM_LOG_DIR}/submit_qwen3_1.7b_aime2026_$(date +%Y%m%d_%H%M%S).txt"

jid_setup=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_SETUP}" --time=06:00:00 --job-name=q3-aime26-setup --export=ALL examples/qwen3_1_7b_opd_aime2026/00_prepare_and_convert.sbatch)
jid_eval_student=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_setup}" --time=24:00:00 --job-name=q3-17b-aime-base --export="ALL,EVAL_MODEL_DIR=${STUDENT_HF_DIR},EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_STUDENT_BASE_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch)
jid_eval_teacher=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_TEACHER}" --dependency="afterok:${jid_eval_student}" --time=24:00:00 --job-name=q3-8b-aime-teacher --export="ALL,EVAL_MODEL_DIR=${TEACHER_HF_DIR},EVAL_TOKENIZER_DIR=${TEACHER_HF_DIR},EVAL_STAGE_DIR=${EVAL_TEACHER_BASE_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_TEACHER_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch)
jid_base_report=$(sbatch --parsable "${SBATCH_REPORT[@]}" --mem="${SLURM_MEM_REPORT}" --dependency="afterok:${jid_eval_teacher}" --time=01:00:00 --job-name=q3-aime-base-rpt --export=ALL,REPORT_MODE=base examples/qwen3_1_7b_opd_aime2026/04_report_aime.sbatch)
jid_opd1=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_OPD}" --dependency="afterok:${jid_base_report}" --time=24:00:00 --job-name=q3-17b-opd-1k --export=ALL,OPD_STAGE=1k examples/qwen3_1_7b_opd_aime2026/03_run_opd.sbatch)
jid_eval1=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_opd1}" --time=24:00:00 --job-name=q3-17b-aime-opd1 --export="ALL,EVAL_MODEL_DIR=${OPD1_HF_DIR}/iter_0000007,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD1K_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch)
jid_report1=$(sbatch --parsable "${SBATCH_REPORT[@]}" --mem="${SLURM_MEM_REPORT}" --dependency="afterok:${jid_eval1}" --time=01:00:00 --job-name=q3-aime-opd1-rpt --export=ALL,REPORT_MODE=opd1k examples/qwen3_1_7b_opd_aime2026/04_report_aime.sbatch)
jid_opd5=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_OPD}" --dependency="afterok:${jid_report1}" --time=24:00:00 --job-name=q3-17b-opd-5k --export=ALL,OPD_STAGE=5k examples/qwen3_1_7b_opd_aime2026/03_run_opd.sbatch)
jid_eval5=$(sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_opd5}" --time=24:00:00 --job-name=q3-17b-aime-opd5 --export="ALL,EVAL_MODEL_DIR=${OPD_CONT_HF_DIR}/iter_0000039,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD5120_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch)
jid_final_report=$(sbatch --parsable "${SBATCH_REPORT[@]}" --mem="${SLURM_MEM_REPORT}" --dependency="afterok:${jid_eval5}" --time=01:00:00 --job-name=q3-aime-final-rpt --export=ALL,REPORT_MODE=final examples/qwen3_1_7b_opd_aime2026/04_report_aime.sbatch)

{
  echo "experiment=qwen3_1.7b_student_qwen3_8b_teacher_aime2026_opd5120"
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=$(git rev-parse HEAD)"
  echo "setup=${jid_setup}"
  echo "qwen3_1.7b_base_eval=${jid_eval_student}"
  echo "qwen3_8b_teacher_eval=${jid_eval_teacher}"
  echo "base_report=${jid_base_report}"
  echo "opd_001024_train=${jid_opd1}"
  echo "opd_001024_eval=${jid_eval1}"
  echo "opd_001024_report=${jid_report1}"
  echo "opd_005120_train=${jid_opd5}"
  echo "opd_005120_eval=${jid_eval5}"
  echo "final_report=${jid_final_report}"
  echo "dependency_policy=strict_serial_gpu_afterok"
  echo "max_concurrent_gpu_count=4"
  echo "student_model=${STUDENT_HF_REPO}@${STUDENT_HF_REVISION}"
  echo "teacher_model=${TEACHER_HF_REPO}@${TEACHER_HF_REVISION}"
  echo "dataset=${AIME_DATASET_REPO}@${AIME_DATASET_REVISION}"
  echo "protocol=30_problems_x_16_independent_samples"
  echo "metric=total_correct_divided_by_480_mc_pass_at_1"
  echo "sampling=temperature_0.6_top_p_0.95_top_k_20_min_p_0"
  echo "seed_formula=42_plus_problem_position_times_16_plus_sample_index"
  echo "max_model_len=${VLLM_EVAL_MAX_MODEL_LEN}"
  echo "max_response_len=${VLLM_EVAL_MAX_RESPONSE_LEN}"
  echo "student_max_num_seqs=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}"
  echo "teacher_max_num_seqs=${VLLM_EVAL_TEACHER_MAX_NUM_SEQS}"
  echo "prompt_contract=${OPD_PROMPT_CONTRACT_VERSION}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "source_manifest=${SOURCE_MANIFEST}"
} | tee "${submission_log}"

echo "Submitted Qwen3-1.7B/Qwen3-8B AIME 2026 chain; metadata: ${submission_log}"

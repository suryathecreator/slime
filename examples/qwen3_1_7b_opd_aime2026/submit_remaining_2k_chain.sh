#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --completed-teacher-job JOB_ID [--dry-run]" >&2
}

completed_teacher_job=""
dry_run=0
while (( $# )); do
  case "$1" in
    --completed-teacher-job)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      completed_teacher_job="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done
if [[ ! "${completed_teacher_job}" =~ ^[0-9]+$ ]]; then
  echo "--completed-teacher-job must be a numeric Slurm job ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

REMAINING_DATA_ROOT="${DATA_ROOT}/remaining_2k_chain"
REMAINING_OPD_2K_JSONL="${REMAINING_DATA_ROOT}/opd_002048_qwen3_aime_prompt.jsonl"
REMAINING_OPD_3K_JSONL="${REMAINING_DATA_ROOT}/opd_next_003072_qwen3_aime_prompt.jsonl"
REMAINING_OPD_METADATA="${REMAINING_DATA_ROOT}/repartition_metadata.json"
REMAINING_SOURCE_MANIFEST="${SCRATCH_ROOT}/source_manifest_remaining_2k.sha256"

EVAL_TEACHER_SINGLE_DIR="${OUTPUT_ROOT}/aime2026_qwen3_8b_teacher_single"
EVAL_TEACHER_SINGLE_V2_DIR="${EVAL_TEACHER_SINGLE_DIR}/rescored/aime_answer_v2"
EVAL_OPD2K_SINGLE_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_opd_002048_single"
EVAL_OPD5120_SINGLE_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_opd_005120_single"
EVAL_OPD5120_MC16_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_opd_005120_mc16"
OPD2_SAVE_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_full_optim"
OPD2_HF_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_hf"
OPD2_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_rollouts"
OPD2_SANITY_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_sanity"

mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}" "${REMAINING_DATA_ROOT}"

bash examples/qwen3_1_7b_opd_aime2026/write_source_manifest.sh "${SOURCE_MANIFEST}" >/dev/null
remaining_sources=(
  examples/qwen3_1_7b_opd_aime2026/aime_answer_v2.py
  examples/qwen3_1_7b_opd_aime2026/02_eval_aime_once_vllm.sbatch
  examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
  examples/qwen3_1_7b_opd_aime2026/env.sh
  examples/qwen3_1_7b_opd_aime2026/evaluate_aime_once_vllm.py
  examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
  examples/qwen3_1_7b_opd_aime2026/prepare_remaining_2k_split.py
  examples/qwen3_1_7b_opd_aime2026/submit_remaining_2k_chain.sh
  examples/qwen3_8b_opd_tillicum/container_exec.sh
)
printf '%s\n' "${remaining_sources[@]}" | sort -u | xargs sha256sum >"${REMAINING_SOURCE_MANIFEST}"
sha256sum -c "${SOURCE_MANIFEST}"
sha256sum -c "${REMAINING_SOURCE_MANIFEST}"

for prerequisite in \
  "${SLIME_SIF}" \
  "${VLLM_EVAL_SITE}/vllm" \
  "${AIME_ZIG_CC}" \
  "${AIME_ZIG_INCLUDE_ROOT}/python3.12/Python.h" \
  "${AIME_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h" \
  "${AIME_JSONL}" \
  "${STUDENT_HF_DIR}/config.json" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${TEACHER_HF_DIR}/config.json" \
  "${EVAL_TEACHER_SINGLE_DIR}/predictions.jsonl" \
  "${EVAL_TEACHER_SINGLE_DIR}/metrics.json" \
  "${EVAL_TEACHER_SINGLE_V2_DIR}/predictions.jsonl" \
  "${EVAL_TEACHER_SINGLE_V2_DIR}/metrics.json" \
  "${CLEANED_OPD_1K_JSONL}" \
  "${CLEANED_OPD_4K_JSONL}"; do
  if [[ ! -e "${prerequisite}" ]]; then
    echo "Missing submission prerequisite: ${prerequisite}" >&2
    exit 1
  fi
done

python3 examples/qwen3_1_7b_opd_aime2026/prepare_remaining_2k_split.py \
  --source-1k "${CLEANED_OPD_1K_JSONL}" \
  --source-4k "${CLEANED_OPD_4K_JSONL}" \
  --output-2k "${REMAINING_OPD_2K_JSONL}" \
  --output-3k "${REMAINING_OPD_3K_JSONL}" \
  --metadata "${REMAINING_OPD_METADATA}" >/dev/null

require_fresh_directory() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse nonempty output directory: ${path}" >&2
    exit 1
  fi
}
for path in \
  "${EVAL_OPD2K_SINGLE_DIR}" \
  "${EVAL_OPD5120_SINGLE_DIR}" \
  "${EVAL_OPD5120_MC16_DIR}" \
  "${OPD2_SAVE_DIR}" \
  "${OPD2_HF_DIR}" \
  "${OPD2_ROLLOUT_LOG_DIR}" \
  "${OPD2_SANITY_DIR}" \
  "${OPD_CONT_SAVE_DIR}" \
  "${OPD_CONT_HF_DIR}" \
  "${OPD_CONT_ROLLOUT_LOG_DIR}" \
  "${OPD_CONT_SANITY_DIR}"; do
  require_fresh_directory "${path}"
done

teacher_accounting="$(sacct -n -P -j "${completed_teacher_job}" --format=JobIDRaw,State,ExitCode)"
teacher_record="$(awk -F'|' -v wanted="${completed_teacher_job}" '$1 == wanted {print $2 "|" $3; exit}' <<<"${teacher_accounting}")"
teacher_state="${teacher_record%%|*}"
teacher_exit="${teacher_record#*|}"
if [[ "${teacher_state}" != "COMPLETED" || "${teacher_exit}" != "0:0" ]]; then
  echo "Teacher job ${completed_teacher_job} must be COMPLETED with exit 0:0, found ${teacher_record:-missing}" >&2
  exit 1
fi
python3 - "${EVAL_TEACHER_SINGLE_V2_DIR}/metrics.json" <<'PY'
import json, sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if metrics.get("scorer_version") != "aime_answer_v2":
    raise SystemExit("Completed teacher artifacts have not been rescored with aime_answer_v2")
if metrics.get("num_generations") != 30:
    raise SystemExit("Completed teacher artifact does not contain exactly 30 generations")
PY

submit_job() {
  local dry_id="$1"
  shift
  if (( dry_run )); then
    printf 'DRY-RUN' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf '%s\n' "${dry_id}"
  else
    local result
    result="$("$@")"
    printf '%s\n' "${result%%;*}"
  fi
}

SBATCH_GPU4=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4 --cpus-per-task=32)
submission_log="${SLURM_LOG_DIR}/submit_qwen3_1.7b_aime2026_remaining_2k_$(date +%Y%m%d_%H%M%S).txt"

jid_opd2="$(submit_job DRY_OPD2K sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_OPD}" --time=24:00:00 --job-name=q3-17b-opd-2k --export="ALL,REMAINING_SOURCE_MANIFEST=${REMAINING_SOURCE_MANIFEST},OPD_STAGE=2k" examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch)"
jid_eval2="$(submit_job DRY_EVAL2K sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_opd2}" --time=24:00:00 --job-name=q3-17b-aime-opd2-once --export="ALL,REMAINING_SOURCE_MANIFEST=${REMAINING_SOURCE_MANIFEST},EVAL_MODEL_DIR=${OPD2_HF_DIR}/iter_0000015,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD2K_SINGLE_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/02_eval_aime_once_vllm.sbatch)"
jid_opd5="$(submit_job DRY_OPD3072 sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_OPD}" --dependency="afterok:${jid_eval2}" --time=24:00:00 --job-name=q3-17b-opd-next3k --export="ALL,REMAINING_SOURCE_MANIFEST=${REMAINING_SOURCE_MANIFEST},OPD_STAGE=5k" examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch)"
jid_eval5="$(submit_job DRY_EVAL5120 sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_opd5}" --time=24:00:00 --job-name=q3-17b-aime-opd5-once --export="ALL,REMAINING_SOURCE_MANIFEST=${REMAINING_SOURCE_MANIFEST},EVAL_MODEL_DIR=${OPD_CONT_HF_DIR}/iter_0000039,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD5120_SINGLE_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/02_eval_aime_once_vllm.sbatch)"
jid_mc5="$(submit_job DRY_MC5120 sbatch --parsable "${SBATCH_GPU4[@]}" --mem="${SLURM_MEM_EVAL_STUDENT}" --dependency="afterok:${jid_eval5}" --time=24:00:00 --job-name=q3-17b-aime-opd5-mc16 --export="ALL,EVAL_MODEL_DIR=${OPD_CONT_HF_DIR}/iter_0000039,EVAL_TOKENIZER_DIR=${STUDENT_HF_DIR},EVAL_STAGE_DIR=${EVAL_OPD5120_MC16_DIR}/shards,EVAL_MAX_NUM_SEQS=${VLLM_EVAL_STUDENT_MAX_NUM_SEQS}" examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch)"

metadata="$({
  echo "experiment=qwen3_1.7b_student_qwen3_8b_teacher_aime2026_opd_2048_plus_3072"
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=$(git rev-parse HEAD)"
  echo "replaces_jobs=179070,179071,179072,179073,179074"
  echo "completed_teacher_single_eval=${completed_teacher_job}"
  echo "completed_teacher_single_eval_state=${teacher_state}_${teacher_exit}"
  echo "root_dependency_policy=completed_teacher_verified_in_sacct_then_opd_submitted_without_dependency"
  echo "opd_002048_train=${jid_opd2}"
  echo "opd_002048_single_eval=${jid_eval2}"
  echo "opd_next_003072_train=${jid_opd5}"
  echo "opd_005120_single_eval=${jid_eval5}"
  echo "opd_005120_mc16_eval=${jid_mc5}"
  echo "dependency_policy=strict_serial_gpu_afterok"
  echo "max_concurrent_gpu_count=4"
  echo "scorer_version=aime_answer_v2"
  echo "opd_actor_gpus=${OPD_ACTOR_GPUS}"
  echo "opd_rollout_gpus=${OPD_ROLLOUT_GPUS}"
  echo "student_model=${STUDENT_HF_REPO}@${STUDENT_HF_REVISION}"
  echo "teacher_model=${TEACHER_HF_REPO}@${TEACHER_HF_REVISION}"
  echo "dataset=${AIME_DATASET_REPO}@${AIME_DATASET_REVISION}"
  echo "single_protocol=30_problems_x_1_sample_mc_sample_zero"
  echo "single_seed_formula=42_plus_problem_position_times_16"
  echo "mc_protocol=30_problems_x_16_independent_samples"
  echo "sampling=temperature_0.6_top_p_0.95_top_k_20_min_p_0"
  echo "opd_first_rows=2048"
  echo "opd_continuation_rows=3072"
  echo "opd_total_unique_rows=5120"
  echo "opd_repartition_metadata=${REMAINING_OPD_METADATA}"
  echo "teacher_single_output=${EVAL_TEACHER_SINGLE_DIR}"
  echo "opd_002048_single_output=${EVAL_OPD2K_SINGLE_DIR}"
  echo "opd_005120_single_output=${EVAL_OPD5120_SINGLE_DIR}"
  echo "opd_005120_mc16_output=${EVAL_OPD5120_MC16_DIR}"
  echo "source_manifest=${SOURCE_MANIFEST}"
  echo "remaining_source_manifest=${REMAINING_SOURCE_MANIFEST}"
})"
if (( dry_run )); then
  printf '%s\n' "${metadata}"
  echo "Dry run complete; no jobs submitted." >&2
else
  printf '%s\n' "${metadata}" | tee "${submission_log}"
  echo "Submitted replacement AIME 2026 2K+3,072 chain; metadata: ${submission_log}"
fi

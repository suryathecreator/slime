#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export AIME_EXAMPLE_DIR
AIME_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${AIME_EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"
export SLURM_MEM_SETUP="${SLURM_MEM_SETUP:-64G}"
export SLURM_MEM_EVAL_STUDENT="${SLURM_MEM_EVAL_STUDENT:-96G}"
export SLURM_MEM_EVAL_TEACHER="${SLURM_MEM_EVAL_TEACHER:-256G}"
export SLURM_MEM_OPD="${SLURM_MEM_OPD:-512G}"
export SLURM_MEM_REPORT="${SLURM_MEM_REPORT:-16G}"
export SCRATCH_ROOT="${AIME_SCRATCH_ROOT:-/gpfs/scrubbed/suryadv/slime-qwen3-1.7b-opd-aime2026}"
export DATA_ROOT="${SCRATCH_ROOT}/data"
export MODEL_ROOT="${SCRATCH_ROOT}/models"
export OUTPUT_ROOT="${SCRATCH_ROOT}/outputs"
export SLURM_LOG_DIR="${SCRATCH_ROOT}/slurm_logs"
export SOURCE_MANIFEST="${SCRATCH_ROOT}/source_manifest.sha256"
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${SCRATCH_ROOT}/vllm_cache"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export RAY_TMPDIR="${SCRATCH_ROOT}/ray_tmp"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_ROOT}/wandb"

export APPTAINER_CACHEDIR="${SCRATCH_ROOT}/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH_ROOT}/apptainer_tmp"
export SLIME_SIF="${SLIME_SIF:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/containers/slime_latest.sandbox}"
export SLIME_CONTAINER_FORMAT=sandbox
export CONTAINER_BIND_ROOTS="${CONTAINER_BIND_ROOTS:-/gpfs/scrubbed/suryadv,/tmp}"
export CONTAINER_HOME="${SCRATCH_ROOT}/container_home"
export CONTAINER_HOME_INNER="/home/${USER:-slime}"

export STUDENT_HF_REPO="Qwen/Qwen3-1.7B"
export STUDENT_HF_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
export TEACHER_HF_REPO="Qwen/Qwen3-8B"
export TEACHER_HF_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen3-1.7B.sh"
export OPD_WANDB_GROUP_PREFIX=qwen3-1.7b
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen3-1.7B"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-1.7B_torch_dist"
export TEACHER_HF_DIR="${MODEL_ROOT}/Qwen3-8B"

export AIME_DATASET_REPO="MathArena/aime_2026"
export AIME_DATASET_REVISION="d2de22f3c656b4f56cf8981212186377d1e23bc3"
export AIME_SOURCE_DIR="${DATA_ROOT}/source/aime_2026"
export AIME_PARQUET="${AIME_SOURCE_DIR}/data/train-00000-of-00001.parquet"
export AIME_JSONL="${DATA_ROOT}/aime_2026.jsonl"
export AIME_PREP_METADATA="${DATA_ROOT}/preparation_metadata.json"
export SOURCE_OPD_1K_JSONL="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openr1_math220k_default_sft50k_opd5120/opd_001024.jsonl"
export SOURCE_OPD_4K_JSONL="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openr1_math220k_default_sft50k_opd5120/opd_next_004096.jsonl"
export CLEANED_OPD_1K_JSONL="${DATA_ROOT}/opd_001024_qwen3_aime_prompt.jsonl"
export CLEANED_OPD_4K_JSONL="${DATA_ROOT}/opd_next_004096_qwen3_aime_prompt.jsonl"
export SPLIT_METADATA="${AIME_PREP_METADATA}"

export VLLM_EVAL_SITE="${VLLM_EVAL_SITE:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/vllm_eval_site}"
export AIME_ZIG_CC="${AIME_ZIG_CC:-/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site/bin/zig-cc}"
export AIME_ZIG_INCLUDE_ROOT="${AIME_ZIG_INCLUDE_ROOT:-/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site/python_include}"
export VLLM_EVAL_NUM_GPUS=4
export VLLM_EVAL_MAX_MODEL_LEN=32768
export VLLM_EVAL_MAX_RESPONSE_LEN=31744
export VLLM_EVAL_GPU_MEMORY_UTILIZATION=0.92
export VLLM_EVAL_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_EVAL_STUDENT_MAX_NUM_SEQS=34
export VLLM_EVAL_TEACHER_MAX_NUM_SEQS=21
export VLLM_EVAL_PREFER_CUDA_GRAPH=1
export VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS=900
export EVAL_EXPECTED_PROBLEMS=30
export EVAL_SAMPLES_PER_PROBLEM=16
export EVAL_EXPECTED_GENERATIONS=480
export EVAL_TEMPERATURE=0.6
export EVAL_TOP_P=0.95
export EVAL_TOP_K=20
export EVAL_MIN_P=0
export EVAL_SEED=42

export EVAL_STUDENT_BASE_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_base"
export EVAL_TEACHER_BASE_DIR="${OUTPUT_ROOT}/aime2026_qwen3_8b_teacher"
export EVAL_OPD1K_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_opd_001024"
export EVAL_OPD5120_DIR="${OUTPUT_ROOT}/aime2026_qwen3_1.7b_opd_005120"
export BASE_REPORT_DIR="${OUTPUT_ROOT}/reports/base_student_teacher"
export OPD1K_REPORT_DIR="${OUTPUT_ROOT}/reports/opd_001024"
export FINAL_REPORT_DIR="${OUTPUT_ROOT}/reports/final"

export QWEN3_ENABLE_THINKING=1
export QWEN3_IM_END_TOKEN_ID=151645
export QWEN3_ENDOFTEXT_TOKEN_ID=151643
export QWEN3_STOP_TOKEN_IDS="${QWEN3_IM_END_TOKEN_ID} ${QWEN3_ENDOFTEXT_TOKEN_ID}"
export OPD_ROLLOUT_STOP_TOKEN_IDS="${QWEN3_STOP_TOKEN_IDS}"
export OPD_PROMPT_CONTRACT_VERSION=qwen3_aime2026_boxed_v1

export OPD_ACTOR_GPUS=2
export OPD_ROLLOUT_GPUS=3
export OPD_RAY_GPUS=3
export OPD_TEACHER_GPU=3
export OPD_TENSOR_MODEL_PARALLEL_SIZE=1
export OPD_CONTEXT_PARALLEL_SIZE=1
export OPD_SEQ_LENGTH=32766
export OPD_ROLLOUT_MAX_CONTEXT_LEN=32766
export OPD_MAX_RESPONSE_LEN=31744
export OPD_MAX_TOKENS_PER_GPU=8192
export OPD_LOG_PROBS_CHUNK_SIZE=2048
export OPD_ROLLOUT_BATCH_SIZE=128
export OPD_GLOBAL_BATCH_SIZE=128
export OPD_N_SAMPLES_PER_PROMPT=1
export OPD_TRAIN_MEMORY_MARGIN_BYTES=0
export OPD_LR=1e-6
export OPD_COLOCATE=1
export OPD_OFFLOAD_TRAIN=1
export OPD_OFFLOAD_ROLLOUT=1
export OPD_OPTIMIZER_CPU_OFFLOAD=1
export OPD_RECOMPUTE_LOSS_FUNCTION=1
export OPD_DISABLE_CUDA_GRAPH=1
export OPD_TEACHER_PORT=13141
export OPD_TEACHER_MEM_FRACTION=0.65
export OPD_SGLANG_RL_ON_POLICY_TARGET=""
export SLIME_SGLANG_FORCE_NATIVE_ROPE=1
export SLIME_SGLANG_PATCH_SITE=1
export SLIME_VLLM_PATCH_SITE=0
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0
export OPD_ALREADY_TRAINED_SAMPLES=0
export OPD_START_ROLLOUT_ID=0
export OPD_MANIFEST_FROM_ROLLOUT_LOGS=0
export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR=""
export OPD_RETIRE_PREVIOUS_HF_DIR=""
export OPD_SANITY_CHECK_ENABLED=1
export OPD_SANITY_FAIL_ON_COLLAPSE=0
export OPD_SANITY_MAX_ROLLOUT_ID=1000000
export OPD_SANITY_MAX_CAP_HIT_RATE=1.0
export OPD_SANITY_MAX_AVG_RESPONSE_TOKENS=31744
export OPD_SANITY_MIN_FINAL_ANSWER_RATE=0.0
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=600
export ESTIMATED_FULL_OPTIM_CKPT_BYTES=50000000000
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF="${SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

export OPD1_SAVE_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_001024_full_optim"
export OPD1_HF_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_001024_hf"
export OPD1_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_001024_rollouts"
export OPD1_SANITY_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_001024_sanity"
export OPD_CONT_SAVE_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_005120_full_optim"
export OPD_CONT_HF_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_005120_hf"
export OPD_CONT_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_005120_rollouts"
export OPD_CONT_SANITY_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_005120_sanity"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports"
export TEACHER_LOG_DIR="${OUTPUT_ROOT}/teacher_logs"

export SFT_FINAL_HF_DIR="${STUDENT_HF_DIR}"
export SFT_FINAL_FULL_CKPT_DIR="${STUDENT_TORCH_DIST_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

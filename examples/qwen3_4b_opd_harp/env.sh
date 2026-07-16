#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export HARP_EXAMPLE_DIR
HARP_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${HARP_EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"
export SCRATCH_ROOT="${HARP_SCRATCH_ROOT:-/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2}"
export DATA_ROOT="${SCRATCH_ROOT}/data"
export MODEL_ROOT="${SCRATCH_ROOT}/models"
export OUTPUT_ROOT="${SCRATCH_ROOT}/outputs"
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${SCRATCH_ROOT}/vllm_cache"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export RAY_TMPDIR="${SCRATCH_ROOT}/ray_tmp"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export SLURM_LOG_DIR="${SCRATCH_ROOT}/slurm_logs"
export SOURCE_MANIFEST="${SCRATCH_ROOT}/source_manifest.sha256"

export APPTAINER_CACHEDIR="${SCRATCH_ROOT}/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH_ROOT}/apptainer_tmp"
export SLIME_SIF="${SLIME_SIF:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/containers/slime_latest.sandbox}"
export SLIME_CONTAINER_FORMAT=sandbox
export CONTAINER_BIND_ROOTS="${CONTAINER_BIND_ROOTS:-/gpfs/scrubbed/suryadv,/tmp}"
export CONTAINER_HOME="${SCRATCH_ROOT}/container_home"
export CONTAINER_HOME_INNER="/home/${USER:-slime}"

export STUDENT_HF_REPO="Qwen/Qwen3-4B"
export TEACHER_HF_REPO="Qwen/Qwen3-32B"
export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen3-4B.sh"
export OPD_WANDB_GROUP_PREFIX=qwen3-4b
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen3-4B"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-4B_torch_dist"
export TEACHER_HF_DIR="${TEACHER_HF_DIR:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/models/Qwen3-32B}"

export HARP_SOURCE_CACHE="${DATA_ROOT}/source/HARP.jsonl.zip"
export HARP_EXISTING_SOURCE="/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm_prob_ratio_mask/eval_benchmarks/source/HARP.jsonl.zip"
export HARP_JSONL="${DATA_ROOT}/harp_seed42_500.jsonl"
export HARP_PREP_METADATA="${DATA_ROOT}/preparation_metadata.json"
export SOURCE_OPD_1K_JSONL="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openr1_math220k_default_sft50k_opd5120/opd_001024.jsonl"
export SOURCE_OPD_4K_JSONL="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/data/openr1_math220k_default_sft50k_opd5120/opd_next_004096.jsonl"
export CLEANED_OPD_1K_JSONL="${DATA_ROOT}/opd_001024_qwen3_harp_prompt.jsonl"
export CLEANED_OPD_4K_JSONL="${DATA_ROOT}/opd_next_004096_qwen3_harp_prompt.jsonl"
export SPLIT_METADATA="${HARP_PREP_METADATA}"

export VLLM_EVAL_SITE="${VLLM_EVAL_SITE:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd/vllm_eval_site}"
export HARP_EVAL_SITE="${SCRATCH_ROOT}/harp_eval_site"
export HARP_EVAL_INSTALL_SPECS="math-verify==0.9.0 latex2sympy2-extended==1.11.0 antlr4-python3-runtime==4.13.2 sympy==1.14.0 pyparsing==3.3.2 tqdm==4.68.3"
export VLLM_EVAL_PYTHON=python3
export VLLM_EVAL_NUM_GPUS=4
export VLLM_EVAL_MAX_MODEL_LEN=32768
export VLLM_EVAL_MAX_RESPONSE_LEN=31744
export VLLM_EVAL_GPU_MEMORY_UTILIZATION=0.92
export VLLM_EVAL_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_EVAL_BENCHMARK_SIZE=64
export VLLM_EVAL_BENCHMARK_MAX_TOKENS=4096
export EVAL_EXPECTED_SAMPLES=500

export EVAL_4B_BASE_DIR="${OUTPUT_ROOT}/harp_v2_qwen3_4b_base"
export EVAL_32B_BASE_DIR="${OUTPUT_ROOT}/harp_v2_qwen3_32b_base"
export EVAL_OPD1K_DIR="${OUTPUT_ROOT}/harp_v2_qwen3_4b_opd_001024"
export EVAL_OPD5120_DIR="${OUTPUT_ROOT}/harp_v2_qwen3_4b_opd_005120"
export BASE_REPORT_DIR="${OUTPUT_ROOT}/reports/base_4b_32b"
export FINAL_REPORT_DIR="${OUTPUT_ROOT}/reports/final"

export QWEN3_ENABLE_THINKING=1
export QWEN3_IM_END_TOKEN_ID=151645
export QWEN3_ENDOFTEXT_TOKEN_ID=151643
export QWEN3_STOP_TOKEN_IDS="${QWEN3_IM_END_TOKEN_ID} ${QWEN3_ENDOFTEXT_TOKEN_ID}"
export OPD_ROLLOUT_STOP_TOKEN_IDS="${QWEN3_STOP_TOKEN_IDS}"
export OPD_PROMPT_CONTRACT_VERSION=qwen3_harp_boxed_v1

export OPD_ACTOR_GPUS=3
export OPD_ROLLOUT_GPUS=3
export OPD_RAY_GPUS=3
export OPD_TEACHER_GPU=3
export OPD_TENSOR_MODEL_PARALLEL_SIZE=1
export OPD_CONTEXT_PARALLEL_SIZE=3
export OPD_SEQ_LENGTH=32766
export OPD_ROLLOUT_MAX_CONTEXT_LEN=32766
export OPD_MAX_RESPONSE_LEN=31744
export OPD_MAX_TOKENS_PER_GPU=2048
export OPD_LOG_PROBS_CHUNK_SIZE=512
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
export OPD_TEACHER_MEM_FRACTION=0.6
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
export ESTIMATED_FULL_OPTIM_CKPT_BYTES=80000000000
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF="${SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

export OPD1_SAVE_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_001024_full_optim"
export OPD1_HF_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_001024_hf"
export OPD1_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_001024_rollouts"
export OPD1_SANITY_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_001024_sanity"
export OPD_CONT_SAVE_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_005120_full_optim"
export OPD_CONT_HF_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_005120_hf"
export OPD_CONT_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_005120_rollouts"
export OPD_CONT_SANITY_DIR="${OUTPUT_ROOT}/qwen3_4b_opd_005120_sanity"
export OPD_TUNING_ROOT="${OUTPUT_ROOT}/opd_throughput_tuning"
export OPD_TUNING_FILE="${OPD_TUNING_ROOT}/selected.env"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports"
export TEACHER_LOG_DIR="${OUTPUT_ROOT}/teacher_logs"

# Compatibility names used by the shared OPD launcher. No SFT is run.
export SFT_FINAL_HF_DIR="${STUDENT_HF_DIR}"
export SFT_FINAL_FULL_CKPT_DIR="${STUDENT_TORCH_DIST_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

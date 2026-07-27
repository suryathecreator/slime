#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export MASKED_SFT_EXAMPLE_DIR
MASKED_SFT_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${MASKED_SFT_EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CONTRACT_FILE="${MASKED_SFT_EXAMPLE_DIR}/config/experiment_contract.json"
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export EXPERIMENT_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-openr1-masked-sft-40k/v1/${CONTRACT_HASH}"
export SCRATCH_ROOT="${EXPERIMENT_ROOT}"
export SOURCE_40K_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-openr1-masked-sft-40k/v1/18b67d8ec9f3ff18"
export DATA_ROOT="${SOURCE_40K_ROOT}/data"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export TMPDIR="${TMPDIR:-${EXPERIMENT_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${EXPERIMENT_ROOT}/ray_tmp}"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Immutable model, dataset, container, and evaluator caches from the OPD path.
export LEGACY_CACHE_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd"
export MODEL_ROOT="${LEGACY_CACHE_ROOT}/models"
export HF_HOME="${LEGACY_CACHE_ROOT}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SLIME_SIF="${LEGACY_CACHE_ROOT}/containers/slime_latest.sandbox"
export VLLM_EVAL_SITE="${LEGACY_CACHE_ROOT}/vllm_eval_site"
export MATH_VERIFY_SITE="/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site"
export EVAL_ZIG_CC="${EVAL_ZIG_CC:-${MATH_VERIFY_SITE}/bin/zig-cc}"
export EVAL_ZIG_INCLUDE_ROOT="${EVAL_ZIG_INCLUDE_ROOT:-${MATH_VERIFY_SITE}/python_include}"
export SFT_TORCH_COMPILE_CC="${SFT_TORCH_COMPILE_CC:-${EVAL_ZIG_CC}}"
export SFT_TORCH_COMPILE_CPATH="${SFT_TORCH_COMPILE_CPATH:-${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}}"

export STUDENT_HF_REPO="Qwen/Qwen3-8B-Base"
export STUDENT_HF_REVISION="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen3-8B-Base"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"
export OPENR1_DATASET="open-r1/OpenR1-Math-220k"
export OPENR1_CONFIG="default"
export OPENR1_SPLIT="train"
export OPENR1_REVISION="dc748648036c1ed619b020e056dc4b603eb39817"
export DATA_SEED=42
export MAX_SEQUENCE_LENGTH=32768
export CORRECT_ONLY_ROWS=40000
export MIXED_CORRECT_ROWS=20000
export MIXED_WRONG_ROWS=20000
export WEIGHT_TAU=0.20

export SELECTED_TRACES_JSONL="${DATA_ROOT}/selected_traces.jsonl"
export PREP_STATS_JSON="${DATA_ROOT}/selection_and_tokenization_stats.json"
export CORRECT_ONLY_JSONL="${DATA_ROOT}/correct_only_40000.jsonl"
export UNMASKED_JSONL="${DATA_ROOT}/correct_wrong_unmasked_40000.jsonl"
export MARGIN_SHARD_DIR="${DATA_ROOT}/margin_shards"
export MARGIN_MERGED_JSONL="${DATA_ROOT}/wrong_margins_merged.jsonl"
export MARGIN_STATS_JSON="${DATA_ROOT}/wrong_margins_stats.json"
export WEIGHTED_JSONL="${DATA_ROOT}/correct_wrong_weighted_tau_0p20_40000.jsonl"
export WEIGHTED_STATS_JSON="${DATA_ROOT}/weighted_tau_0p20_stats.json"
export SPLIT_METADATA="${PREP_STATS_JSON}"

export SFT_SIZE=40000
export SFT_NUM_EPOCH=1
export SFT_ROLLOUT_BATCH_SIZE=200
export SFT_GLOBAL_BATCH_SIZE=200
export SFT_NUM_ROLLOUT=200
export SFT_FINAL_ROLLOUT_ID=199
export SFT_MILESTONE_ROLLOUT_IDS="49 99 149 199"
export SFT_SAVE_INTERVAL=50
export SFT_ACTOR_GPUS=4
export SFT_TENSOR_MODEL_PARALLEL_SIZE=2
export SFT_CONTEXT_PARALLEL_SIZE=1
export SFT_PIPELINE_MODEL_PARALLEL_SIZE=1
export SFT_ZERO_STAGE=1
export SFT_CKPT_FORMAT=torch_dist
export SFT_INPUT_KEY=input_ids
export SFT_ROLLOUT_FUNCTION_PATH="examples.qwen3_8b_openr1_math220k_masked_sft_40k.weighted_sft_rollout.generate_rollout"
export SFT_LOSS_MASK_TYPE=qwen3
export SFT_LOSS_MASK_PREFLIGHT_ENABLED=0
export SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK=0
export SFT_LOSS_MASK_PREFLIGHT_SAMPLES=0
export SFT_MAX_TOKENS_PER_GPU=16384
export SFT_SEQ_LENGTH=32768
export SFT_GRAD_CLIP=1.0
export SFT_LOG_PROBS_CHUNK_SIZE=-1
export SFT_OPTIMIZER_CPU_OFFLOAD=1
export SFT_RECOMPUTE_LOSS_FUNCTION=1
export SFT_LR=1e-6
export SFT_WANDB_PROJECT="slime-openr1-masked-sft"
export SFT_WANDB_GROUP="qwen3-8b-full-sft-40k"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports"
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=60

export QWEN3_ENABLE_THINKING=1
export QWEN3_ENDOFTEXT_TOKEN_ID=151643
export QWEN3_IM_END_TOKEN_ID=151645
export QWEN3_STOP_TOKEN_IDS="151645 151643"
export MATH500_DATASET="HuggingFaceH4/MATH-500"
export MATH500_REVISION="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
export EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}/math500"
export EVAL_TEMPERATURE=0.0
export EVAL_TOP_P=1.0
export EVAL_TOP_K=-1
export EVAL_MAX_RESPONSE_LEN=32768
export EVAL_TARGET_CONTEXT=32768
export EVAL_SAFETY_MARGIN=64
export EVAL_SEED=1234
export EVAL_STOP_TOKEN_IDS="151645 151643"
export EVAL_EXPECTED_SAMPLES=500
export SHARED_2K_CONTRACT_FILE="${SLIME_REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_2k/config/experiment_contract.json"
export SHARED_2K_CONTRACT_HASH
SHARED_2K_CONTRACT_HASH="$(sha256sum "${SHARED_2K_CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SHARED_BASE_EVAL_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-openr1-masked-sft-2k/v1/${SHARED_2K_CONTRACT_HASH}/outputs/math500/base_8b"
export BASE_EVAL_SUMMARY="${SHARED_BASE_EVAL_ROOT}/summary.json"
export BASE_EVAL_SUMMARY_SHA256_FILE="${SHARED_BASE_EVAL_ROOT}/summary.sha256"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export MASKED_SFT_2K_DIR
MASKED_SFT_2K_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${MASKED_SFT_2K_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CONTRACT_FILE="${MASKED_SFT_2K_DIR}/config/experiment_contract.json"
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen2_5-7b-openr1-masked-sft-2k"
export EXPERIMENT_ROOT="${SCRATCH_ROOT}/v1/${CONTRACT_HASH}"
export DATA_ROOT="${EXPERIMENT_ROOT}/data"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export TMPDIR="${TMPDIR:-${EXPERIMENT_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${EXPERIMENT_ROOT}/ray_tmp}"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export AXOLOTL_SOURCE_ROOT="/gpfs/scrubbed/suryadv/repos/axolotl-masked-sft/runs/2026-06-23_openr1_220k_2k_balanced_full_sft_axolotl_harp_vllm_prob_ratio_mask/unpaired/seed_42/datasets"
export AXOLOTL_CORRECT_ONLY="${AXOLOTL_SOURCE_ROOT}/correct_only.jsonl"
export AXOLOTL_MIXED_UNMASKED="${AXOLOTL_SOURCE_ROOT}/correct_wrong_unmasked.jsonl"
export AXOLOTL_SOURCE_STATS="${AXOLOTL_SOURCE_ROOT}/stats.json"

export LEGACY_CACHE_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd"
export MODEL_ROOT="${SCRATCH_ROOT}/models"
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SLIME_SIF="${LEGACY_CACHE_ROOT}/containers/slime_latest.sandbox"
export APPTAINER_CACHEDIR="${SCRATCH_ROOT}/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH_ROOT}/apptainer_tmp"
export CONTAINER_BIND_ROOTS="/gpfs/scrubbed/suryadv,/tmp"
export CONTAINER_HOME="${SCRATCH_ROOT}/container_home"
export CONTAINER_HOME_INNER="/home/${USER:-suryadv}"
export VLLM_EVAL_SITE="${LEGACY_CACHE_ROOT}/vllm_eval_site"
export MATH_VERIFY_SITE="/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site"
export EVAL_ZIG_CC="${EVAL_ZIG_CC:-${MATH_VERIFY_SITE}/bin/zig-cc}"
export EVAL_ZIG_INCLUDE_ROOT="${EVAL_ZIG_INCLUDE_ROOT:-${MATH_VERIFY_SITE}/python_include}"
export SFT_TORCH_COMPILE_CC="${SFT_TORCH_COMPILE_CC:-${EVAL_ZIG_CC}}"
export SFT_TORCH_COMPILE_CPATH="${SFT_TORCH_COMPILE_CPATH:-${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}}"

export STUDENT_HF_REPO="Qwen/Qwen2.5-7B"
export STUDENT_HF_REVISION="d149729398750b98c0af14eb82c78cfe92750796"
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen2.5-7B"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen2.5-7B_torch_dist"
export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen2.5-7B.sh"
export SFT_MODEL_ARGS_SCRIPT="${STUDENT_MODEL_ARGS_SCRIPT}"
export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON="{}"

export DATA_SEED=42
export MASK_SEED=42
export MAX_SEQUENCE_LENGTH=32768
export CORRECT_ONLY_ROWS=2000
export MIXED_CORRECT_ROWS=1000
export MIXED_WRONG_ROWS=1000

export SELECTED_TRACES_JSONL="${DATA_ROOT}/selected_trace_union.jsonl"
export PREP_STATS_JSON="${DATA_ROOT}/selection_and_tokenization_stats.json"
export CORRECT_ONLY_JSONL="${DATA_ROOT}/correct_only_2000.jsonl"
export UNMASKED_JSONL="${DATA_ROOT}/correct_wrong_unmasked_2000.jsonl"
export MARGIN_SHARD_DIR="${DATA_ROOT}/margin_shards"
export VARIANT_STATS_JSON="${DATA_ROOT}/variant_stats.json"
export TOKENIZER_INVENTORY_JSON="${DATA_ROOT}/tokenizer_control_inventory.json"

export RANDOM_MASK_25_JSONL="${DATA_ROOT}/random_mask_25_2000.jsonl"
export RANDOM_MASK_50_JSONL="${DATA_ROOT}/random_mask_50_2000.jsonl"
export RANDOM_MASK_70_JSONL="${DATA_ROOT}/random_mask_70_2000.jsonl"
export RANDOM_MASK_80_JSONL="${DATA_ROOT}/random_mask_80_2000.jsonl"
export RANDOM_MASK_90_JSONL="${DATA_ROOT}/random_mask_90_2000.jsonl"
export INVERSE_TAU_0P20_JSONL="${DATA_ROOT}/inverse_tau_0p20_2000.jsonl"
export INVERSE_TAU_0P05_JSONL="${DATA_ROOT}/inverse_tau_0p05_2000.jsonl"
export MARGIN_MASK_JSONL="${DATA_ROOT}/margin_mask_2000.jsonl"
export PROB_RATIO_MASK_JSONL="${DATA_ROOT}/prob_ratio_mask_2000.jsonl"

export SFT_SIZE=2000
export SFT_NUM_EPOCH=1
export SFT_ROLLOUT_BATCH_SIZE=200
export SFT_GLOBAL_BATCH_SIZE=200
export SFT_NUM_ROLLOUT=10
export SFT_FINAL_ROLLOUT_ID=9
export SFT_MILESTONE_ROLLOUT_IDS="9"
export SFT_SAVE_INTERVAL=10
export SFT_ACTOR_GPUS=4
export SFT_TENSOR_MODEL_PARALLEL_SIZE=2
export SFT_CONTEXT_PARALLEL_SIZE=1
export SFT_PIPELINE_MODEL_PARALLEL_SIZE=1
export SFT_ZERO_STAGE=1
export SFT_CKPT_FORMAT=torch_dist
export SFT_INPUT_KEY=input_ids
export SFT_ROLLOUT_FUNCTION_PATH="examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout.generate_rollout"
export SFT_LOSS_MASK_TYPE=qwen
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
export SFT_WANDB_GROUP="qwen2.5-7b-full-sft-2k"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports"
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=60
export SPLIT_METADATA="${PREP_STATS_JSON}"

export QWEN_ENDOFTEXT_TOKEN_ID=151643
export QWEN_IM_START_TOKEN_ID=151644
export QWEN_IM_END_TOKEN_ID=151645
export MATH500_DATASET="HuggingFaceH4/MATH-500"
export MATH500_REVISION="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

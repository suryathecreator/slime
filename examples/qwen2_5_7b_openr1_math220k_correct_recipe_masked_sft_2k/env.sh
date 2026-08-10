#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export CORRECT_RECIPE_MASKED_DIR
CORRECT_RECIPE_MASKED_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${CORRECT_RECIPE_MASKED_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CONTRACT_FILE="${CORRECT_RECIPE_MASKED_DIR}/config/experiment_contract.json"
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen2-5-7b-correct-recipe-masked-sft-2k"
export EXPERIMENT_ROOT="${SCRATCH_ROOT}/v1/${CONTRACT_HASH}"
export DATA_ROOT="${EXPERIMENT_ROOT}/data"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export HANDOFF_ROOT="${EXPERIMENT_ROOT}/handoff"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export SOURCE_CONTRACT_HASH="3568736a3743c319"
export SOURCE_EXPERIMENT_ROOT="/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/v1/${SOURCE_CONTRACT_HASH}"
export SOURCE_DATA_ROOT="${SOURCE_EXPERIMENT_ROOT}/data"
export SOURCE_SELECTED="${SOURCE_DATA_ROOT}/selected/8k_shared.jsonl"
export SOURCE_PRETOKENIZED="${SOURCE_DATA_ROOT}/datasets/qwen2_5_7b_8k.jsonl"
export SOURCE_PREP_STATS="${SOURCE_DATA_ROOT}/selection_and_tokenization_stats.json"
export SOURCE_BASE_MANIFEST="${SOURCE_EXPERIMENT_ROOT}/handoff/checkpoints/base_qwen2_5_7b.json"
export SOURCE_CORRECT_ONLY_MANIFEST="${SOURCE_EXPERIMENT_ROOT}/handoff/checkpoints/qwen2_5_7b_8k.json"
export SOURCE_TRAINING_STATUS="${SOURCE_EXPERIMENT_ROOT}/TRAINING_STATUS.json"

export MODEL_ROOT="${SOURCE_EXPERIMENT_ROOT}/models"
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen2.5-7B"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen2.5-7B_torch_dist"
export STUDENT_HF_REPO="Qwen/Qwen2.5-7B"
export STUDENT_HF_REVISION="d149729398750b98c0af14eb82c78cfe92750796"
export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen2.5-7B.sh"
export SFT_MODEL_ARGS_SCRIPT="${STUDENT_MODEL_ARGS_SCRIPT}"

export SOURCE_SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k"
export HF_HOME="${SOURCE_SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LEGACY_CACHE_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd"
export SLIME_SIF="${LEGACY_CACHE_ROOT}/containers/slime_latest.sandbox"
export APPTAINER_CACHEDIR="${SCRATCH_ROOT}/apptainer_cache"
export APPTAINER_TMPDIR="${SCRATCH_ROOT}/apptainer_tmp"
export CONTAINER_BIND_ROOTS="/gpfs/scrubbed/suryadv,/tmp"
export CONTAINER_HOME="${SCRATCH_ROOT}/container_home"
export CONTAINER_HOME_INNER="/home/${USER:-suryadv}"
export EVAL_ZIG_INCLUDE_ROOT="/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site/python_include"
export SFT_TORCH_COMPILE_CC="/gpfs/scrubbed/suryadv/slime-qwen3-4b-opd-harp-v2/harp_eval_site/bin/zig-cc"
export SFT_TORCH_COMPILE_CPATH="${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}"

export OPENR1_DATASET="open-r1/OpenR1-Math-220k"
export OPENR1_CONFIG="default"
export OPENR1_SPLIT="train"
export OPENR1_REVISION="dc748648036c1ed619b020e056dc4b603eb39817"
export DATA_SEED=42
export MASK_SEED=42
export MIXED_CORRECT_ROWS=1000
export MIXED_WRONG_ROWS=1000
export MAX_SEQUENCE_LENGTH=10240
export TRACE_TOKEN_CAP=8192

export SELECTED_TRACES_JSONL="${DATA_ROOT}/selected_mixed_traces.jsonl"
export UNMASKED_JSONL="${DATA_ROOT}/unmasked_2000.jsonl"
export PREP_STATS_JSON="${DATA_ROOT}/selection_and_tokenization_stats.json"
export TOKENIZER_INVENTORY_JSON="${DATA_ROOT}/tokenizer_control_inventory.json"
export MARGIN_SHARD_DIR="${DATA_ROOT}/margin_shards"
export VARIANT_STATS_JSON="${DATA_ROOT}/variant_stats.json"
export SCHEDULE_AUDIT_JSON="${DATA_ROOT}/schedule_audit.json"
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
export SFT_NUM_EPOCH=4
export SFT_ROLLOUT_BATCH_SIZE=64
export SFT_GLOBAL_BATCH_SIZE=64
export SFT_NUM_ROLLOUT=125
export SFT_FINAL_ROLLOUT_ID=124
export SFT_MILESTONE_ROLLOUT_IDS=124
export SFT_SAVE_INTERVAL=125
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
export SFT_SEQ_LENGTH=10240
export SFT_GRAD_CLIP=1.0
export SFT_LOG_PROBS_CHUNK_SIZE=-1
export SFT_OPTIMIZER_CPU_OFFLOAD=1
export SFT_RECOMPUTE_LOSS_FUNCTION=1
export SFT_OPTIMIZER=adam
export SFT_LR=5e-6
export SFT_MIN_LR=1e-6
export SFT_LR_DECAY_STYLE=cosine
export SFT_LR_WARMUP_FRACTION=0.03
export SFT_LR_WARMUP_INIT=0.0
export SFT_LR_DECAY_ITERS=125
export SFT_WEIGHT_DECAY=1e-4
export SFT_ADAM_BETA1=0.9
export SFT_ADAM_BETA2=0.95
export SFT_ADAM_EPS=1.0e-8
export SFT_WANDB_PROJECT="slime-openr1-correct-recipe-masked-sft"
export SFT_WANDB_GROUP="qwen2.5-7b-correct-recipe-masked-sft-2k"
export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON="{}"
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=60
export SPLIT_METADATA="${PREP_STATS_JSON}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"

VARIANTS=(
  unmasked random_mask_70 inverse_tau_0p20 inverse_tau_0p05
  random_mask_25 random_mask_50 random_mask_80 random_mask_90
  margin_mask prob_ratio_mask
)

variant_data_path() {
  case "${1:?variant required}" in
    unmasked) echo "${UNMASKED_JSONL}" ;;
    random_mask_25) echo "${RANDOM_MASK_25_JSONL}" ;;
    random_mask_50) echo "${RANDOM_MASK_50_JSONL}" ;;
    random_mask_70) echo "${RANDOM_MASK_70_JSONL}" ;;
    random_mask_80) echo "${RANDOM_MASK_80_JSONL}" ;;
    random_mask_90) echo "${RANDOM_MASK_90_JSONL}" ;;
    inverse_tau_0p20) echo "${INVERSE_TAU_0P20_JSONL}" ;;
    inverse_tau_0p05) echo "${INVERSE_TAU_0P05_JSONL}" ;;
    margin_mask) echo "${MARGIN_MASK_JSONL}" ;;
    prob_ratio_mask) echo "${PROB_RATIO_MASK_JSONL}" ;;
    *) echo "Unknown variant: $1" >&2; return 2 ;;
  esac
}

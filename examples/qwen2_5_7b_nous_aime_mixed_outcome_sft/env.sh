#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export NOUS_AIME_DIR
NOUS_AIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export AIME_GENERALIZATION_DIR="${NOUS_AIME_DIR}"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${NOUS_AIME_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NOUS_AIME_RECIPE="${NOUS_AIME_RECIPE:-one_epoch}"
case "${NOUS_AIME_RECIPE}" in
  one_epoch) export CONTRACT_FILE="${NOUS_AIME_DIR}/config/experiment_contract.json" ;;
  four_epoch) export CONTRACT_FILE="${NOUS_AIME_DIR}/config/experiment_contract_4epoch.json" ;;
  *) echo "Unknown NOUS_AIME_RECIPE: ${NOUS_AIME_RECIPE}" >&2; return 2 ;;
esac
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen2-5-7b-nous-aime-mixed-outcome-sft"
export EXPERIMENT_ROOT="${SCRATCH_ROOT}/v1/${CONTRACT_HASH}"
export DATA_ROOT="${EXPERIMENT_ROOT}/data"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export HANDOFF_ROOT="${EXPERIMENT_ROOT}/handoff"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export NOUS_SOURCE_ROOT="/gpfs/scrubbed/suryadv/datasets/NousResearch_eval-Qwen3-235B-A22B-reasoning/67bba1eb0642bc9a39d6971fd6c149e6bbbbb0c8"
export AIME24_SOURCE="${NOUS_SOURCE_ROOT}/aime24/conversations.parquet"
export AIME25_SOURCE="${NOUS_SOURCE_ROOT}/aime25/conversations.parquet"
export AIME24_SOURCE_SHA256="1df735b01ae4b479a027d58dc53f79d7b4d38a8b6207c00cd9208779cba0cdb7"
export AIME25_SOURCE_SHA256="4d41cb204087ffa313d14fbdde9cfb40f937623ff250f1e5aed664810826f004"
export DATA_SEED=42

export CORRECT_ONLY_ROOT="/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k/v1/3568736a3743c319"
export MODEL_ROOT="${CORRECT_ONLY_ROOT}/models"
export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen2.5-7B"
export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen2.5-7B_torch_dist"
export STUDENT_HF_REPO="Qwen/Qwen2.5-7B"
export STUDENT_HF_REVISION="d149729398750b98c0af14eb82c78cfe92750796"
export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen2.5-7B.sh"
export SFT_MODEL_ARGS_SCRIPT="${STUDENT_MODEL_ARGS_SCRIPT}"
export BASE_CHECKPOINT_MANIFEST="${CORRECT_ONLY_ROOT}/handoff/checkpoints/base_qwen2_5_7b.json"

export SOURCE_SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k"
export HF_HOME="${SOURCE_SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}"
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

export PREP_STATS_JSON="${DATA_ROOT}/selection_and_tokenization_stats.json"
export TOKENIZER_INVENTORY_JSON="${DATA_ROOT}/tokenizer_control_inventory.json"
export SOURCE_ARTIFACTS_JSON="${DATA_ROOT}/source_artifacts.json"
export SCHEDULE_AUDIT_JSON="${DATA_ROOT}/schedule_audit.json"
export AIME24_EVAL_JSONL="${DATA_ROOT}/eval/aime24_30.jsonl"
export AIME25_EVAL_JSONL="${DATA_ROOT}/eval/aime25_30.jsonl"
export HELD_IN_EVAL_JSONL="${AIME24_EVAL_JSONL}"

export SFT_NUM_EPOCH
SFT_NUM_EPOCH="$(jq -er '.training.epochs' "${CONTRACT_FILE}")"
export NOUS_AIME_OPTIMIZER_UPDATES
NOUS_AIME_OPTIMIZER_UPDATES="$(jq -er '.training.optimizer_updates' "${CONTRACT_FILE}")"
export NOUS_AIME_FINAL_ITERATION
NOUS_AIME_FINAL_ITERATION="$(jq -er '.training.final_iteration' "${CONTRACT_FILE}")"
export SFT_ROLLOUT_BATCH_SIZE=64
export SFT_GLOBAL_BATCH_SIZE=64
export SFT_ACTOR_GPUS=4
export SFT_TENSOR_MODEL_PARALLEL_SIZE=4
export SFT_CONTEXT_PARALLEL_SIZE=1
export SFT_PIPELINE_MODEL_PARALLEL_SIZE=1
export SFT_ZERO_STAGE=0
export SFT_CKPT_FORMAT=torch_dist
export SFT_TRAIN_ENTRYPOINT=train.py
export SFT_INPUT_KEY=input_ids
export SFT_ROLLOUT_FUNCTION_PATH="examples.qwen2_5_7b_openr1_math220k_masked_sft_2k.weighted_sft_rollout.generate_rollout"
export SFT_LOSS_MASK_TYPE=qwen
export SFT_LOSS_MASK_PREFLIGHT_ENABLED=0
export SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK=0
export SFT_LOSS_MASK_PREFLIGHT_SAMPLES=0
export SFT_MAX_TOKENS_PER_GPU=16384
export SFT_SEQ_LENGTH=32768
export SFT_GRAD_CLIP=1.0
export SFT_LOG_PROBS_CHUNK_SIZE=2048
export SFT_OPTIMIZER_CPU_OFFLOAD=1
export SFT_RECOMPUTE_LOSS_FUNCTION=1
export SFT_OPTIMIZER=adam
export SFT_LR=5e-6
export SFT_MIN_LR=1e-6
export SFT_LR_DECAY_STYLE=cosine
export SFT_LR_WARMUP_FRACTION=0.03
export SFT_LR_WARMUP_INIT=0.0
export SFT_WEIGHT_DECAY=1e-4
export SFT_ADAM_BETA1=0.9
export SFT_ADAM_BETA2=0.95
export SFT_ADAM_EPS=1.0e-8
export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON="{}"
export SFT_WANDB_PROJECT="slime-nous-aime-mixed-outcome-sft"
export SFT_WANDB_GROUP_PREFIX="qwen2.5-7b-nous-aime"
export SFT_MANIFEST_FAMILY="nous_aime_mixed_outcome"
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=60
export CHECKPOINT_REQUIRE_ROLLOUT_STATE=1
export SPLIT_METADATA="${PREP_STATS_JSON}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"

ALL_TRAINED_VARIANTS=(
  aime24_correct_3000 aime24_incorrect_3000
  aime25_correct_3000 aime25_incorrect_3000
)

variant_data_path() {
  local variant="${1:?variant required}"
  case "${variant}" in
    aime24_correct_3000|aime24_incorrect_3000|aime25_correct_3000|aime25_incorrect_3000)
      echo "${DATA_ROOT}/datasets/${variant}.jsonl"
      ;;
    *) echo "Unknown training variant: ${variant}" >&2; return 2 ;;
  esac
}

configure_sft_variant() {
  local variant="${1:?variant required}"
  export SFT_VARIANT="${variant}"
  export SFT_PARQUET
  SFT_PARQUET="$(variant_data_path "${variant}")"
  unset SFT_INITIAL_HF_DIR
  export SFT_SIZE=3000
  export SFT_NUM_ROLLOUT="${NOUS_AIME_OPTIMIZER_UPDATES}"
  export SFT_FINAL_ROLLOUT_ID="${NOUS_AIME_FINAL_ITERATION}"
  export SFT_MILESTONE_ROLLOUT_IDS="${NOUS_AIME_FINAL_ITERATION}"
  export SFT_SAVE_INTERVAL=24
  export SFT_LR_DECAY_ITERS="${NOUS_AIME_OPTIMIZER_UPDATES}"
}

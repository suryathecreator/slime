#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export AIME_SPLIT_6K_DIR
AIME_SPLIT_6K_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export AIME_GENERALIZATION_DIR="${AIME_SPLIT_6K_DIR}"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${AIME_SPLIT_6K_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CONTRACT_FILE="${AIME_SPLIT_6K_DIR}/config/experiment_contract.json"
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen2-5-7b-aime-2009-2024-split-sft-6k"
export EXPERIMENT_ROOT="${SCRATCH_ROOT}/v1/${CONTRACT_HASH}"
export DATA_ROOT="${EXPERIMENT_ROOT}/data"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export HANDOFF_ROOT="${EXPERIMENT_ROOT}/handoff"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

export SOURCE_DATASET_ROOT="/gpfs/scrubbed/suryadv/datasets/suryadv_qwen3-8b-aime-2009-2024-16x/c34143c92856d9b90fdb9c808c7e66070dcab9a1/data"
export SOURCE_DATASET_REVISION="c34143c92856d9b90fdb9c808c7e66070dcab9a1"
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
export HELD_IN_EVAL_JSONL="${DATA_ROOT}/eval/held_in_106.jsonl"
export HELD_OUT_EVAL_JSONL="${DATA_ROOT}/eval/held_out_106.jsonl"

export SFT_NUM_EPOCH=2
export AIME_SPLIT_6K_OPTIMIZER_UPDATES=188
export AIME_SPLIT_6K_FINAL_ITERATION=187
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
export SFT_SEQ_LENGTH=32896
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
export SFT_WANDB_PROJECT="slime-aime-2009-2024-split-sft-6k"
export SFT_WANDB_GROUP_PREFIX="qwen2.5-7b-aime-split-6k"
export SFT_MANIFEST_FAMILY="aime_2009_2024_split"
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
  held_in_correct_6000 held_in_incorrect_6000
  held_out_correct_6000 held_out_incorrect_6000
)

variant_data_path() {
  local variant="${1:?variant required}"
  case "${variant}" in
    held_in_correct_6000|held_in_incorrect_6000|held_out_correct_6000|held_out_incorrect_6000)
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
  export SFT_SIZE=6000
  export SFT_NUM_ROLLOUT="${AIME_SPLIT_6K_OPTIMIZER_UPDATES}"
  export SFT_FINAL_ROLLOUT_ID="${AIME_SPLIT_6K_FINAL_ITERATION}"
  export SFT_MILESTONE_ROLLOUT_IDS="${AIME_SPLIT_6K_FINAL_ITERATION}"
  export SFT_SAVE_INTERVAL=24
  export SFT_LR_DECAY_ITERS="${AIME_SPLIT_6K_OPTIMIZER_UPDATES}"
}

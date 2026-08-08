#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it." >&2
  exit 2
fi

export CORRECT_ONLY_EXAMPLE_DIR
CORRECT_ONLY_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${CORRECT_ONLY_EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${SLIME_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CONTRACT_FILE="${CORRECT_ONLY_EXAMPLE_DIR}/config/experiment_contract.json"
export CONTRACT_HASH
CONTRACT_HASH="$(sha256sum "${CONTRACT_FILE}" | awk '{print substr($1,1,16)}')"
export SCRATCH_ROOT="/gpfs/scrubbed/suryadv/slime-qwen-correct-only-openr1-sft-2k"
export EXPERIMENT_ROOT="${SCRATCH_ROOT}/v1/${CONTRACT_HASH}"
export DATA_ROOT="${EXPERIMENT_ROOT}/data"
export MODEL_ROOT="${EXPERIMENT_ROOT}/models"
export OUTPUT_ROOT="${EXPERIMENT_ROOT}/outputs"
export MANIFEST_ROOT="${EXPERIMENT_ROOT}/manifests"
export SLURM_LOG_DIR="${EXPERIMENT_ROOT}/slurm_logs"
export HANDOFF_ROOT="${EXPERIMENT_ROOT}/handoff"
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DIR="${EXPERIMENT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"

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
export CORRECT_ONLY_ROWS=2000
export SHARED_8K_SELECTED="${DATA_ROOT}/selected/8k_shared.jsonl"
export QWEN25_3B_16K_SELECTED="${DATA_ROOT}/selected/16k_qwen2_5_3b.jsonl"
export PREP_STATS_JSON="${DATA_ROOT}/selection_and_tokenization_stats.json"
export SFT_MEMORY_PROFILE=memory_r2
export SCHEDULE_AUDIT_DIR="${DATA_ROOT}/schedule_audits/${SFT_MEMORY_PROFILE}"

export SFT_SIZE=2000
export SFT_NUM_EPOCH=4
export SFT_ROLLOUT_BATCH_SIZE=64
export SFT_GLOBAL_BATCH_SIZE=64
export SFT_NUM_ROLLOUT=125
export SFT_FINAL_ROLLOUT_ID=124
export SFT_MILESTONE_ROLLOUT_IDS=124
export SFT_SAVE_INTERVAL=125
export SFT_ACTOR_GPUS=4
export SFT_CONTEXT_PARALLEL_SIZE=1
export SFT_PIPELINE_MODEL_PARALLEL_SIZE=1
export SFT_ZERO_STAGE=1
export SFT_CKPT_FORMAT=torch_dist
export SFT_INPUT_KEY=input_ids
export SFT_ROLLOUT_FUNCTION_PATH="examples.qwen_correct_only_openr1_math220k_sft_2k.correct_only_sft_rollout.generate_rollout"
export SFT_LOSS_MASK_PREFLIGHT_ENABLED=0
export SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK=0
export SFT_LOSS_MASK_PREFLIGHT_SAMPLES=0
export SFT_GRAD_CLIP=1.0
export SFT_LOG_PROBS_CHUNK_SIZE=-1
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
export SFT_WANDB_PROJECT="slime-openr1-correct-only-sft"
export CHECKPOINT_PRUNE_INTERVAL_SECONDS=60
export SPLIT_METADATA="${PREP_STATS_JSON}"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resolve_model() {
  local model_key="${1:?model key required}"
  export MODEL_KEY="${model_key}"
  case "${model_key}" in
    qwen2_5_3b)
      export STUDENT_HF_REPO="Qwen/Qwen2.5-3B"
      export STUDENT_HF_REVISION="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
      export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen2.5-3B"
      export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen2.5-3B_torch_dist"
      export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen2.5-3B.sh"
      export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON='{}'
      export SFT_LOSS_MASK_TYPE=qwen
      ;;
    qwen2_5_7b)
      export STUDENT_HF_REPO="Qwen/Qwen2.5-7B"
      export STUDENT_HF_REVISION="d149729398750b98c0af14eb82c78cfe92750796"
      export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen2.5-7B"
      export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen2.5-7B_torch_dist"
      export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen2.5-7B.sh"
      export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON='{}'
      export SFT_LOSS_MASK_TYPE=qwen
      ;;
    qwen3_4b)
      export STUDENT_HF_REPO="Qwen/Qwen3-4B-Base"
      export STUDENT_HF_REVISION="906bfd4b4dc7f14ee4320094d8b41684abff8539"
      export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen3-4B-Base"
      export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-4B-Base_torch_dist"
      export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen3-4B.sh"
      export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":true}'
      export SFT_LOSS_MASK_TYPE=qwen3
      ;;
    qwen3_8b)
      export STUDENT_HF_REPO="Qwen/Qwen3-8B-Base"
      export STUDENT_HF_REVISION="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
      export STUDENT_HF_DIR="${MODEL_ROOT}/Qwen3-8B-Base"
      export STUDENT_TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"
      export STUDENT_MODEL_ARGS_SCRIPT="scripts/models/qwen3-8B.sh"
      export SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":true}'
      export SFT_LOSS_MASK_TYPE=qwen3
      ;;
    *) echo "Unknown MODEL_KEY=${model_key}" >&2; return 2 ;;
  esac
  export SFT_MODEL_ARGS_SCRIPT="${STUDENT_MODEL_ARGS_SCRIPT}"
}

resolve_run() {
  local run_key="${1:?run key required}"
  export RUN_KEY="${run_key}"
  case "${run_key}" in
    qwen2_5_3b_8k)
      resolve_model qwen2_5_3b
      export TRACE_TOKEN_CAP=8192 MAX_SEQUENCE_LENGTH=10240
      export SFT_TENSOR_MODEL_PARALLEL_SIZE=1 SFT_MAX_TOKENS_PER_GPU=16384
      export SFT_OPTIMIZER_CPU_OFFLOAD=0
      ;;
    qwen2_5_3b_16k)
      resolve_model qwen2_5_3b
      export TRACE_TOKEN_CAP=16384 MAX_SEQUENCE_LENGTH=18432
      export SFT_TENSOR_MODEL_PARALLEL_SIZE=1 SFT_MAX_TOKENS_PER_GPU=16384
      export SFT_OPTIMIZER_CPU_OFFLOAD=0
      ;;
    qwen2_5_7b_8k)
      resolve_model qwen2_5_7b
      export TRACE_TOKEN_CAP=8192 MAX_SEQUENCE_LENGTH=10240
      export SFT_TENSOR_MODEL_PARALLEL_SIZE=2 SFT_MAX_TOKENS_PER_GPU=16384
      export SFT_OPTIMIZER_CPU_OFFLOAD=1
      ;;
    qwen3_4b_8k)
      resolve_model qwen3_4b
      export TRACE_TOKEN_CAP=8192 MAX_SEQUENCE_LENGTH=10240
      export SFT_TENSOR_MODEL_PARALLEL_SIZE=1 SFT_MAX_TOKENS_PER_GPU=9216
      export SFT_OPTIMIZER_CPU_OFFLOAD=0
      ;;
    qwen3_8b_8k)
      resolve_model qwen3_8b
      export TRACE_TOKEN_CAP=8192 MAX_SEQUENCE_LENGTH=10240
      export SFT_TENSOR_MODEL_PARALLEL_SIZE=2 SFT_MAX_TOKENS_PER_GPU=16384
      export SFT_OPTIMIZER_CPU_OFFLOAD=1
      ;;
    *) echo "Unknown RUN_KEY=${run_key}" >&2; return 2 ;;
  esac
  export SFT_PARQUET="${DATA_ROOT}/datasets/${RUN_KEY}.jsonl"
  export STOCK_MESSAGES_JSONL="${DATA_ROOT}/messages/${RUN_KEY}.jsonl"
  export SFT_SEQ_LENGTH="${MAX_SEQUENCE_LENGTH}"
  export SFT_WANDB_GROUP="${RUN_KEY}-correct-only"
}

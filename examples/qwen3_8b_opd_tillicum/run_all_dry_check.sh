#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"

echo "Dry check environment"
echo "  repo: ${SLIME_REPO_ROOT}"
echo "  account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "  scratch: ${SCRATCH_ROOT}"
echo "  container (${SLIME_CONTAINER_FORMAT}): ${SLIME_SIF}"

if [[ "${OPD_CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
  opd_seq_modulus=$((2 * OPD_CONTEXT_PARALLEL_SIZE))
  if (( OPD_SEQ_LENGTH % opd_seq_modulus != 0 )); then
    echo "OPD_SEQ_LENGTH must be divisible by 2 * OPD_CONTEXT_PARALLEL_SIZE when CP > 1." >&2
    echo "  OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH}" >&2
    echo "  OPD_CONTEXT_PARALLEL_SIZE=${OPD_CONTEXT_PARALLEL_SIZE}" >&2
    echo "  required divisor=${opd_seq_modulus}" >&2
    exit 1
  fi
fi
case "${OPD_INITIAL_LOAD_MODE}" in
  megatron|hf) ;;
  *)
    echo "OPD_INITIAL_LOAD_MODE must be 'megatron' or 'hf', got '${OPD_INITIAL_LOAD_MODE}'" >&2
    exit 1
    ;;
esac
if [[ "${OPD_COLOCATE}" == "1" ]]; then
  if [[ "${OPD_ACTOR_GPUS}" -ne "${OPD_ROLLOUT_GPUS}" ]]; then
    echo "Colocated OPD expects actor and rollout GPU counts to match." >&2
    echo "  OPD_ACTOR_GPUS=${OPD_ACTOR_GPUS}" >&2
    echo "  OPD_ROLLOUT_GPUS=${OPD_ROLLOUT_GPUS}" >&2
    exit 1
  fi
  if [[ "${OPD_RAY_GPUS}" -lt "${OPD_ACTOR_GPUS}" ]]; then
    echo "Colocated OPD needs OPD_RAY_GPUS >= OPD_ACTOR_GPUS." >&2
    echo "  OPD_RAY_GPUS=${OPD_RAY_GPUS}" >&2
    echo "  OPD_ACTOR_GPUS=${OPD_ACTOR_GPUS}" >&2
    exit 1
  fi
fi

SHELL_FILES=(
  examples/qwen3_8b_opd_tillicum/env.sh
  examples/qwen3_8b_opd_tillicum/checkpoint_utils.sh
  examples/qwen3_8b_opd_tillicum/container_exec.sh
  examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
  examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
  examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
  examples/qwen3_8b_opd_tillicum/submit_25k_10k_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_resume_sft_eval_then_opd_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_sft_colocate4_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_sft_offload4_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_opd_continue_1k_to_25k_val100_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_cleaned_sft_opd_vllm_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_cleanup_base_opd_2gpu.sh
  examples/qwen3_8b_opd_tillicum/02_prepare_data_25k_10k.sbatch
  examples/qwen3_8b_opd_tillicum/02_prepare_cleaned_data.sbatch
  examples/qwen3_8b_opd_tillicum/02_prepare_opd_continuation_25k_val100.sbatch
  examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
  examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/04_smoke_sft_zero_stages.sbatch
  examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_inner.sh
  examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
  examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
  examples/qwen3_8b_opd_tillicum/08_maybe_base_eval_math500.sbatch
  examples/qwen3_8b_opd_tillicum/09_cleanup_base_opd_2gpu.sbatch
  examples/qwen3_8b_opd_tillicum/10_setup_vllm_eval_env.sbatch
)

PYTHON_FILES=(
  examples/qwen3_8b_opd_tillicum/02_prepare_cleaned_openthoughts3.py
  examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py
  examples/qwen3_8b_opd_tillicum/eval_math500_vllm.py
  examples/qwen3_8b_opd_tillicum/sglang_launch_native_rope.py
  examples/qwen3_8b_opd_tillicum/summarize_opd_sanity.py
  examples/qwen3_8b_opd_tillicum/summarize_eval.py
  examples/qwen3_8b_opd_tillicum/prepare_math500_subset.py
  examples/qwen3_8b_opd_tillicum/prepare_opd_continuation_pool.py
  examples/qwen3_8b_opd_tillicum/write_opd_trained_manifest.py
  sitecustomize.py
  slime/backends/sglang_utils/native_rope.py
  slime/backends/sglang_utils/sglang_engine.py
  slime/utils/arguments.py
  slime/ray/placement_group.py
  slime/ray/rollout.py
  slime/ray/utils.py
)

echo "Checking shell syntax"
for file in "${SHELL_FILES[@]}"; do
  bash -n "${file}"
done

echo "Checking required container env forwarding"
REQUIRED_CONTAINER_ENV=(
  OPD_INITIAL_LOAD_MODE
  OPD_INITIAL_LOAD_DIR
  OPD_COLOCATE
  OPD_OFFLOAD_TRAIN
  OPD_OFFLOAD_ROLLOUT
  OPD_RECOMPUTE_LOSS_FUNCTION
  OPD_OPTIMIZER_CPU_OFFLOAD
  OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH
  OPD_SKIP_ROLLOUT_DATA_STATE_LOAD
  OPD_ALREADY_TRAINED_SAMPLES
  OPD_START_ROLLOUT_ID
  OPD_PREVIOUS_RUN_LABEL
  OPD_PREVIOUS_SAVE_DIR
  OPD_PREVIOUS_HF_SNAPSHOT_DIR
  OPD_PREVIOUS_ROLLOUT_LOG_DIR
  OPD_PREVIOUS_TRAINED_MANIFEST
  OPD_CONTINUATION_METADATA
  OPD_MANIFEST_FROM_ROLLOUT_LOGS
  OPD_REF_LOAD_DIR
  OPD_ROLLOUT_MAX_CONTEXT_LEN
  OPD_LOG_PROBS_CHUNK_SIZE
  OPD_TRAIN_MEMORY_MARGIN_BYTES
  OPD_LR
  OPD_DISABLE_CUDA_GRAPH
  SLIME_SGLANG_FORCE_NATIVE_ROPE
  SLIME_SGLANG_PATCH_SITE
  SLIME_SGLANG_PYTORCH_CUDA_ALLOC_CONF
  OPD_SGLANG_RL_ON_POLICY_TARGET
  OPD_SANITY_CHECK_ENABLED
  OPD_SANITY_REPORT_DIR
  OPD_SANITY_SUMMARY_DIR
  EVAL_DISABLE_CUDA_GRAPH
  EVAL_SGLANG_RL_ON_POLICY_TARGET
  MATH500_VAL100_JSONL
  MATH500_VAL100_CONFIG
  MATH500_VAL100_METADATA
  CLEANED_EXPERIMENT_LABEL
  CLEANED_OPD_FIRST_SIZE
  CLEANED_OPD_NEXT_SIZE
  CLEANED_METADATA
  CLEANED_OPD_RESERVE_JSONL
  CLEANED_OPD_1K_JSONL
  CLEANED_OPD_4K_JSONL
  SFT_ACTOR_GPUS
  SFT_TENSOR_MODEL_PARALLEL_SIZE
  SFT_CONTEXT_PARALLEL_SIZE
  SFT_PIPELINE_MODEL_PARALLEL_SIZE
  SFT_ZERO_STAGE
  SFT_LR
  SFT_CKPT_FORMAT
  EVAL_MAX_CONTEXT_LEN
  EVAL_BACKEND
  REPORT_SFT_FINAL_ONLY
  REPORT_OPD_FINAL_ONLY
  VLLM_EVAL_VENV
  VLLM_EVAL_INSTALL_SPEC
  VLLM_EVAL_PIP_EXTRA_INDEX_URL
  VLLM_EVAL_NUM_GPUS
  VLLM_EVAL_MAX_MODEL_LEN
  VLLM_EVAL_GPU_MEMORY_UTILIZATION
  VLLM_EVAL_MAX_NUM_SEQS
  VLLM_EVAL_MAX_NUM_BATCHED_TOKENS
  VLLM_EVAL_DTYPE
  VLLM_EVAL_TRUST_REMOTE_CODE
)
for name in "${REQUIRED_CONTAINER_ENV[@]}"; do
  if ! grep -Eq "^[[:space:]]+${name}$" examples/qwen3_8b_opd_tillicum/container_exec.sh; then
    echo "container_exec.sh must forward ${name} into Apptainer." >&2
    exit 1
  fi
done

echo "Checking Python syntax"
python3 -m py_compile "${PYTHON_FILES[@]}"

SBATCH_FILES=(
  examples/qwen3_8b_opd_tillicum/02_prepare_data_25k_10k.sbatch
  examples/qwen3_8b_opd_tillicum/02_prepare_cleaned_data.sbatch
  examples/qwen3_8b_opd_tillicum/02_prepare_opd_continuation_25k_val100.sbatch
  examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
  examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/04_smoke_sft_zero_stages.sbatch
  examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
  examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
  examples/qwen3_8b_opd_tillicum/08_maybe_base_eval_math500.sbatch
  examples/qwen3_8b_opd_tillicum/09_cleanup_base_opd_2gpu.sbatch
  examples/qwen3_8b_opd_tillicum/10_setup_vllm_eval_env.sbatch
)

echo "Checking Slurm scripts with sbatch --test-only"
for file in "${SBATCH_FILES[@]}"; do
  sbatch --test-only -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" "${file}"
done

if [[ "${RUN_CONTAINER_CHECKS:-0}" == "1" ]]; then
  if [[ -e "${SLIME_SIF}" ]]; then
    echo "Checking imports inside container"
    "${SCRIPT_DIR}/container_exec.sh" python3 -c "import encodings, slime, sglang, torch, transformers, datasets; import slime.backends.sglang_utils.native_rope; print('container imports ok')"
    echo "Checking comma-safe container env forwarding"
    REPORT_EXPERIMENT_NOTE="comma, spaces -> ok" \
      "${SCRIPT_DIR}/container_exec.sh" python3 -c "import os, sys; expected = 'comma, spaces -> ok'; actual = os.environ.get('REPORT_EXPERIMENT_NOTE'); sys.exit(0 if actual == expected else f'bad REPORT_EXPERIMENT_NOTE: {actual!r}')"
  else
    echo "RUN_CONTAINER_CHECKS=1 but SLIME_SIF does not exist; skipping import check."
  fi
else
  echo "Skipping container import check. Set RUN_CONTAINER_CHECKS=1 after the SIF exists."
fi

echo "git diff --stat"
git diff --stat

echo "git diff --name-only"
git diff --name-only

echo "git status --short"
git status --short

echo "Dry checks completed. No real jobs were submitted."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]"
  exit 2
fi

if [[ ! -e "${SLIME_SIF}" ]]; then
  cat >&2 <<EOF
Missing container image/sandbox:
  ${SLIME_SIF}

Create it with:
  bash ${SCRIPT_DIR}/00_pull_or_load_container.sh
EOF
  exit 1
fi

APPTAINER_BIN="${APPTAINER_BIN:-}"
if [[ -z "${APPTAINER_BIN}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_BIN=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    APPTAINER_BIN=singularity
  else
    echo "Neither apptainer nor singularity is available on PATH." >&2
    exit 1
  fi
fi

mkdir -p \
  "${APPTAINER_CACHEDIR}" \
  "${APPTAINER_TMPDIR}" \
  "${CONTAINER_HOME}" \
  "${TMPDIR}" \
  "${RAY_TMPDIR}"

export APPTAINER_CACHEDIR APPTAINER_TMPDIR

IFS=',' read -r -a BIND_ROOTS <<< "${CONTAINER_BIND_ROOTS}"
APPTAINER_ARGS=(
  exec
  --nv
  --cleanenv
  --ipc
  --writable-tmpfs
  --cwd "${SLIME_REPO_ROOT}"
  --home "${CONTAINER_HOME}:${CONTAINER_HOME_INNER}"
)

for bind_root in "${BIND_ROOTS[@]}"; do
  if [[ -e "${bind_root}" ]]; then
    APPTAINER_ARGS+=(--bind "${bind_root}:${bind_root}")
  fi
done

PASS_ENV=(
  ACCOUNT
  PARTITION
  QOS
  SCRATCH_ROOT
  DATA_ROOT
  MODEL_ROOT
  OUTPUT_ROOT
  HF_HOME
  HF_DATASETS_CACHE
  TRANSFORMERS_CACHE
  WANDB_MODE
  WANDB_DIR
  TMPDIR
  RAY_TMPDIR
  SLIME_REPO_ROOT
  TILLICUM_EXAMPLE_DIR
  STUDENT_HF_REPO
  TEACHER_HF_REPO
  OT3_DATASET
  OT3_SPLIT
  MATH500_DATASET
  SFT_SIZE
  OPD_POOL_SIZE
  OPD_TRAIN_SIZE
  OPD_SIZE
  OPD_RUN_LABEL
  DATA_SEED
  DATA_MATH_FIELD
  DATA_MATH_VALUE
  STUDENT_HF_DIR
  TEACHER_HF_DIR
  STUDENT_TORCH_DIST_DIR
  SFT_PARQUET
  OPD_JSONL
  SPLIT_METADATA
  MATH500_JSONL
  MATH500_CONFIG
  SFT_SAVE_DIR
  OPD_SAVE_DIR
  SFT_HF_SNAPSHOT_DIR
  OPD_HF_SNAPSHOT_DIR
  SFT_HF_SNAPSHOT_TEMPLATE
  OPD_HF_SNAPSHOT_TEMPLATE
  SFT_FINAL_HF_DIR
  OPD_FINAL_HF_DIR
  SFT_DETAILS_DIR
  OPD_ROLLOUT_LOG_DIR
  TEACHER_LOG_DIR
  EVAL_OUTPUT_DIR
  BASE_EVAL_OUTPUT_DIR
  SFT_EVAL_OUTPUT_DIR
  OPD_EVAL_OUTPUT_DIR
  COMBINED_EVAL_OUTPUT_DIR
  CHECKPOINT_REPORT_DIR
  SLURM_LOG_DIR
  SFT_NUM_EPOCH
  SFT_NUM_ROLLOUT
  SFT_FINAL_ROLLOUT_ID
  SFT_MILESTONE_ROLLOUT_IDS
  SFT_ROLLOUT_BATCH_SIZE
  SFT_GLOBAL_BATCH_SIZE
  SFT_MAX_TOKENS_PER_GPU
  SFT_SAVE_INTERVAL
  OPD_NUM_ROLLOUT
  OPD_FINAL_ROLLOUT_ID
  OPD_EFFECTIVE_TRAIN_SAMPLES
  OPD_MILESTONE_ROLLOUT_IDS
  OPD_ROLLOUT_BATCH_SIZE
  OPD_GLOBAL_BATCH_SIZE
  OPD_N_SAMPLES_PER_PROMPT
  OPD_MAX_RESPONSE_LEN
  OPD_SEQ_LENGTH
  OPD_MAX_TOKENS_PER_GPU
  OPD_SAVE_INTERVAL
  OPD_ACTOR_GPUS
  OPD_ROLLOUT_GPUS
  OPD_RAY_GPUS
  OPD_TEACHER_GPU
  OPD_TENSOR_MODEL_PARALLEL_SIZE
  OPD_CONTEXT_PARALLEL_SIZE
  OPD_TEACHER_PORT
  OPD_TEACHER_MEM_FRACTION
  OPD_TRAINED_MANIFEST
  CHECKPOINT_PRUNE_INTERVAL_SECONDS
  ESTIMATED_FULL_OPTIM_CKPT_BYTES
  EVAL_MAX_RESPONSE_LEN
  EVAL_ROLLOUT_BATCH_SIZE
  EVAL_ROLLOUT_NUM_GPUS
  EVAL_NUM_REPEATS
  EVAL_SGLANG_SERVER_CONCURRENCY
  EVAL_SKIP_COMPLETED
  EVAL_EXPECTED_SAMPLES
  EVAL_TARGETS
  PYTHONUNBUFFERED
  TOKENIZERS_PARALLELISM
  NCCL_DEBUG
  PYTORCH_CUDA_ALLOC_CONF
)

for name in "${PASS_ENV[@]}"; do
  if [[ -n "${!name-}" ]]; then
    APPTAINER_ARGS+=(--env "${name}=${!name}")
  fi
done

APPTAINER_ARGS+=(
  --env "HOME=${CONTAINER_HOME_INNER}"
  --env "PYTHONPATH=${SLIME_REPO_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}"
  --env "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
)

container_path_exists() {
  local path="$1"
  if [[ -d "${SLIME_SIF}" ]]; then
    [[ -e "${SLIME_SIF}${path}" ]]
  else
    "${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" "${SLIME_SIF}" test -e "${path}"
  fi
}

container_has_py_or_legacy_pyc() {
  local module_path="$1"
  container_path_exists "${module_path}.py" || container_path_exists "${module_path}.pyc"
}

check_stdlib_files() {
  local missing=()
  for module_path in \
    /usr/lib/python3.12/encodings/__init__ \
    /usr/lib/python3.12/os \
    /usr/lib/python3.12/site; do
    if ! container_has_py_or_legacy_pyc "${module_path}"; then
      missing+=("${module_path}.py or ${module_path}.pyc")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    cat >&2 <<EOF
Container Python stdlib preflight failed before launching the requested command.
The sandbox appears to be pycache-only and not sourceless-import materialized:
  ${SLIME_SIF}

Missing source or legacy sourceless .pyc paths:
EOF
    printf "  %s\n" "${missing[@]}" >&2
    cat >&2 <<EOF

Repair or rebuild it with:
  bash ${SCRIPT_DIR}/00_pull_or_load_container.sh
EOF
    return 1
  fi
}

if [[ "${CONTAINER_PYTHON_PREFLIGHT:-1}" != "0" ]]; then
  check_stdlib_files
  preflight_log="$(mktemp "${TMPDIR%/}/container_python_preflight.XXXXXX")"
  if ! "${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" "${SLIME_SIF}" \
    python3 -c "import encodings, os, site" >/dev/null 2>"${preflight_log}"; then
    cat >&2 <<EOF
Container Python preflight failed before launching the requested command.
The container may be incomplete or corrupt:
  ${SLIME_SIF}

Python could not import required stdlib modules. The sandbox may be pycache-only
and not sourceless-import materialized. Repair or rebuild it with:
  bash ${SCRIPT_DIR}/00_pull_or_load_container.sh

Preflight stderr:
EOF
    sed -n '1,120p' "${preflight_log}" >&2 || true
    rm -f "${preflight_log}"
    exit 1
  fi
  rm -f "${preflight_log}"
fi

"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" "${SLIME_SIF}" "$@"

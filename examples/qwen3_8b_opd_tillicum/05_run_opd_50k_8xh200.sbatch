#!/bin/bash
#SBATCH --job-name=slime-qwen3-opd1k32k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=18:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/checkpoint_utils.sh"

mkdir -p "${SLURM_LOG_DIR}" "${OPD_SAVE_DIR}" "${OPD_HF_SNAPSHOT_DIR}" "${OPD_ROLLOUT_LOG_DIR}" "${TEACHER_LOG_DIR}" "${CHECKPOINT_REPORT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

for required_path in \
  "${OPD_JSONL}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${TEACHER_HF_DIR}" \
  "${SFT_FINAL_HF_DIR}" \
  "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" \
  "${SFT_FINAL_FULL_CKPT_DIR}/.metadata" \
  "${SFT_FINAL_FULL_CKPT_DIR}/common.pt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

echo "Starting OPD on $(hostname) at $(date)"
echo "OPD data pool: ${OPD_JSONL}"
echo "OPD pool rows: ${OPD_POOL_SIZE}"
echo "OPD train rows requested: ${OPD_TRAIN_SIZE}"
echo "OPD effective samples: ${OPD_EFFECTIVE_TRAIN_SAMPLES}"
echo "OPD rollout batch/global batch/samples per prompt: ${OPD_ROLLOUT_BATCH_SIZE}/${OPD_GLOBAL_BATCH_SIZE}/${OPD_N_SAMPLES_PER_PROMPT}"
echo "OPD rollout ids: 0-${OPD_FINAL_ROLLOUT_ID}"
echo "OPD max response len: ${OPD_MAX_RESPONSE_LEN}"
echo "OPD actor/rollout/ray GPUs: ${OPD_ACTOR_GPUS}/${OPD_ROLLOUT_GPUS}/${OPD_RAY_GPUS}"
echo "OPD TP/CP/seq length/max tokens per GPU: ${OPD_TENSOR_MODEL_PARALLEL_SIZE}/${OPD_CONTEXT_PARALLEL_SIZE}/${OPD_SEQ_LENGTH}/${OPD_MAX_TOKENS_PER_GPU}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "Teacher: ${TEACHER_HF_DIR} on physical GPU ${OPD_TEACHER_GPU}"
echo "Ray GPUs: 0-$((OPD_RAY_GPUS - 1))"
echo "Save dir: ${OPD_SAVE_DIR}"
echo "HF eval snapshots: ${OPD_HF_SNAPSHOT_DIR}"
echo "Expected final OPD rollout id: ${OPD_FINAL_ROLLOUT_ID}"
echo "Initial OPD load mode: ${OPD_INITIAL_LOAD_MODE}"
echo "Initial OPD load dir: ${OPD_INITIAL_LOAD_DIR}"
echo "SFT full checkpoint dir: ${SFT_FINAL_FULL_CKPT_DIR}"
echo "OPD optimizer CPU offload: ${OPD_OPTIMIZER_CPU_OFFLOAD}"

checkpoint_start_pruner opd "${OPD_SAVE_DIR}" "${CHECKPOINT_PRUNE_INTERVAL_SECONDS}" "${CHECKPOINT_REPORT_DIR}"
cleanup_pruner() {
  checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
}
trap cleanup_pruner EXIT

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

ray stop --force >/dev/null 2>&1 || true

TEACHER_IP="127.0.0.1"
TEACHER_PORT="${OPD_TEACHER_PORT}"
TEACHER_LOG="${TEACHER_LOG_DIR}/sglang_teacher_${SLURM_JOB_ID:-manual}.log"

CUDA_VISIBLE_DEVICES="${OPD_TEACHER_GPU}" python3 -m sglang.launch_server \
  --model-path "${TEACHER_HF_DIR}" \
  --host 0.0.0.0 \
  --port "${TEACHER_PORT}" \
  --tp 1 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static "${OPD_TEACHER_MEM_FRACTION}" \
  >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!

cleanup() {
  set +e
  if [[ -n "${TEACHER_PID:-}" ]]; then
    kill "${TEACHER_PID}" >/dev/null 2>&1 || true
  fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for teacher server on ${TEACHER_IP}:${TEACHER_PORT}"
until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
  tail -n 20 "${TEACHER_LOG}" || true
  sleep 10
done
curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info" || true
sleep 10

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source scripts/models/qwen3-8B.sh

comma_range() {
  local n="$1"
  local out=""
  local i
  if [[ "${n}" -lt 1 ]]; then
    echo "comma_range requires n >= 1, got ${n}" >&2
    return 1
  fi
  for ((i = 0; i < n; i++)); do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${i}"
  done
  printf "%s" "${out}"
}

validate_megatron_load_dir() {
  local label="$1"
  local load_dir="$2"
  local latest_file="${load_dir}/latest_checkpointed_iteration.txt"
  local latest_iter
  local iter_dir

  if [[ "${load_dir}" == "${SFT_FINAL_HF_DIR}" ]] || [[ -f "${load_dir}/config.json" && ! -f "${load_dir}/.metadata" ]]; then
    echo "${label} load dir appears to be an HF snapshot, not a Megatron checkpoint: ${load_dir}" >&2
    exit 1
  fi
  if [[ ! -s "${latest_file}" ]]; then
    echo "${label} load dir is missing latest checkpoint marker: ${latest_file}" >&2
    exit 1
  fi
  latest_iter="$(tr -d "[:space:]" <"${latest_file}")"
  if [[ ! "${latest_iter}" =~ ^[0-9]+$ ]]; then
    echo "${label} latest checkpoint marker is not numeric: ${latest_file}=${latest_iter}" >&2
    exit 1
  fi
  printf -v iter_dir "%s/iter_%07d" "${load_dir}" "${latest_iter}"
  if [[ ! -f "${iter_dir}/.metadata" || ! -f "${iter_dir}/common.pt" ]]; then
    echo "${label} latest checkpoint is incomplete or not Megatron torch_dist: ${iter_dir}" >&2
    exit 1
  fi
  echo "${label} validated Megatron load dir: ${load_dir} iteration ${latest_iter} (${iter_dir})"
}

validate_hf_load_dir() {
  local label="$1"
  local load_dir="$2"
  local -a weight_files=()

  if [[ ! -d "${load_dir}" ]]; then
    echo "${label} HF load dir is missing: ${load_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${load_dir}/config.json" ]]; then
    echo "${label} HF load dir is missing config.json: ${load_dir}" >&2
    exit 1
  fi
  shopt -s nullglob
  weight_files=("${load_dir}"/model*.safetensors "${load_dir}"/pytorch_model*.bin)
  shopt -u nullglob
  if [[ ! -f "${load_dir}/model.safetensors.index.json" && "${#weight_files[@]}" -eq 0 ]]; then
    echo "${label} HF load dir is missing model weight files: ${load_dir}" >&2
    exit 1
  fi
  echo "${label} validated HF load dir: ${load_dir}"
}

CKPT_HF_CHECKPOINT="${STUDENT_HF_DIR}"
if [[ -f "${OPD_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  OPD_LOAD_DIR="${OPD_SAVE_DIR}"
  OPD_LOAD_KIND="opd_resume"
  echo "Resuming OPD from full optimizer checkpoint ${OPD_LOAD_DIR}"
  validate_megatron_load_dir "${OPD_LOAD_KIND}" "${OPD_LOAD_DIR}"
else
  case "${OPD_INITIAL_LOAD_MODE}" in
    megatron)
      OPD_LOAD_DIR="${OPD_INITIAL_LOAD_DIR}"
      OPD_LOAD_KIND="sft_full_optim_initial"
      echo "Starting OPD from full SFT optimizer checkpoint ${OPD_LOAD_DIR}"
      validate_megatron_load_dir "${OPD_LOAD_KIND}" "${OPD_LOAD_DIR}"
      ;;
    hf)
      OPD_LOAD_DIR="${OPD_INITIAL_LOAD_DIR}"
      OPD_LOAD_KIND="sft_hf_weights_initial"
      CKPT_HF_CHECKPOINT="${OPD_LOAD_DIR}"
      echo "Starting OPD from SFT HF weights snapshot ${OPD_LOAD_DIR}"
      echo "Fresh OPD optimizer will be initialized because SFT full optimizer checkpoint TP layout is incompatible with this actor layout."
      validate_hf_load_dir "${OPD_LOAD_KIND}" "${OPD_LOAD_DIR}"
      ;;
    *)
      echo "Unsupported OPD_INITIAL_LOAD_MODE=${OPD_INITIAL_LOAD_MODE}" >&2
      exit 1
      ;;
  esac
fi

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${CKPT_HF_CHECKPOINT}"
  --ref-load "${STUDENT_TORCH_DIST_DIR}"
  --load "${OPD_LOAD_DIR}"
  --save "${OPD_SAVE_DIR}"
  --save-interval "${OPD_SAVE_INTERVAL}"
  --save-hf "${OPD_HF_SNAPSHOT_TEMPLATE}"
)

ROLLOUT_ARGS=(
  --prompt-data "${OPD_JSONL}"
  --input-key prompt
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${OPD_NUM_ROLLOUT}"
  --rollout-batch-size "${OPD_ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${OPD_N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${OPD_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --rollout-top-p 1
  --global-batch-size "${OPD_GLOBAL_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${OPD_ROLLOUT_LOG_DIR}/rollout_{rollout_id}.pt"
)

RM_ARGS=(
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

PERF_ARGS=(
  --tensor-model-parallel-size "${OPD_TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${OPD_CONTEXT_PARALLEL_SIZE}"
  --seq-length "${OPD_SEQ_LENGTH}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${OPD_MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-opd
  --opd-type sglang
  --opd-kl-coef 1.0
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)
if [[ "${OPD_OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

WANDB_ARGS=()
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  WANDB_ARGS+=(--wandb-project slime-tillicum --wandb-group "qwen3-8b-opd-${OPD_RUN_LABEL}")
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
RAY_VISIBLE_DEVICES="$(comma_range "${OPD_RAY_GPUS}")"
echo "Ray CUDA_VISIBLE_DEVICES=${RAY_VISIBLE_DEVICES}"
CUDA_VISIBLE_DEVICES="${RAY_VISIBLE_DEVICES}" ray start --head --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${OPD_RAY_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SLIME_REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"HF_HOME\": \"${HF_HOME}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\"
  }
}"

CUDA_VISIBLE_DEVICES="${RAY_VISIBLE_DEVICES}" ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${OPD_ACTOR_GPUS}" \
  --rollout-num-gpus "${OPD_ROLLOUT_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}"
'

checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
checkpoint_prune_old opd "${OPD_SAVE_DIR}" final
checkpoint_report opd "${OPD_SAVE_DIR}" "${CHECKPOINT_REPORT_DIR}"
checkpoint_verify_single_latest opd "${OPD_SAVE_DIR}"

if [[ ! -d "${OPD_FINAL_HF_DIR}" ]]; then
  echo "Missing final OPD HF eval snapshot: ${OPD_FINAL_HF_DIR}" >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
from pathlib import Path

opd_jsonl = Path(os.environ["OPD_JSONL"])
save_dir = Path(os.environ["OPD_SAVE_DIR"])
hf_dir = Path(os.environ["OPD_FINAL_HF_DIR"])
manifest_path = Path(os.environ["OPD_TRAINED_MANIFEST"])
train_size = int(os.environ["OPD_TRAIN_SIZE"])
effective_samples = int(os.environ["OPD_EFFECTIVE_TRAIN_SAMPLES"])
rollout_batch_size = int(os.environ["OPD_ROLLOUT_BATCH_SIZE"])
global_batch_size = int(os.environ["OPD_GLOBAL_BATCH_SIZE"])
num_rollout = int(os.environ["OPD_NUM_ROLLOUT"])
final_rollout_id = int(os.environ["OPD_FINAL_ROLLOUT_ID"])

if effective_samples != num_rollout * rollout_batch_size:
    raise SystemExit(
        f"Effective sample mismatch: {effective_samples} != {num_rollout} * {rollout_batch_size}"
    )
if effective_samples > train_size:
    raise SystemExit(f"Effective samples {effective_samples} exceed OPD_TRAIN_SIZE {train_size}")

rows = []
with opd_jsonl.open("r", encoding="utf-8") as f:
    for row_offset, line in enumerate(f):
        if row_offset >= effective_samples:
            break
        item = json.loads(line)
        metadata = item.get("metadata") or {}
        rows.append(
            {
                "row_offset": row_offset,
                "source_row_id": metadata.get("source_row_id"),
                "prompt_sha256": metadata.get("prompt_sha256"),
            }
        )

if len(rows) != effective_samples:
    raise SystemExit(f"OPD pool has only {len(rows)} rows, expected {effective_samples}")

dataset_state = save_dir / "rollout" / f"global_dataset_state_dict_{final_rollout_id}.pt"
if not dataset_state.exists():
    raise SystemExit(f"Missing rollout dataset state for continuation: {dataset_state}")

manifest = {
    "run_label": os.environ["OPD_RUN_LABEL"],
    "opd_data_pool": str(opd_jsonl),
    "split_metadata": os.environ.get("SPLIT_METADATA"),
    "opd_pool_size": int(os.environ["OPD_POOL_SIZE"]),
    "opd_train_size": train_size,
    "opd_effective_samples": effective_samples,
    "rollout_start_id": 0,
    "rollout_final_id": final_rollout_id,
    "num_rollout": num_rollout,
    "rollout_batch_size": rollout_batch_size,
    "global_batch_size": global_batch_size,
    "n_samples_per_prompt": int(os.environ["OPD_N_SAMPLES_PER_PROMPT"]),
    "resume_from_full_optim_checkpoint": str(save_dir / f"iter_{final_rollout_id:07d}"),
    "initial_load_mode": os.environ.get("OPD_INITIAL_LOAD_MODE"),
    "initial_load_dir": os.environ.get("OPD_INITIAL_LOAD_DIR"),
    "sft_full_checkpoint_dir": os.environ.get("SFT_FINAL_FULL_CKPT_DIR"),
    "resume_from_hf_snapshot": str(hf_dir),
    "rollout_dataset_state": str(dataset_state),
    "trained_rows": rows,
}

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
manifest_path.write_text(manifest_text, encoding="utf-8")
(hf_dir / "opd_trained_manifest.json").write_text(manifest_text, encoding="utf-8")
print(f"Wrote OPD trained manifest: {manifest_path}")
print(f"Copied OPD trained manifest to: {hf_dir / 'opd_trained_manifest.json'}")
PY

if [[ -f "${SPLIT_METADATA}" ]]; then
  cp -f "${SPLIT_METADATA}" "${OPD_SAVE_DIR}/split_metadata.json"
  cp -f "${SPLIT_METADATA}" "${OPD_FINAL_HF_DIR}/split_metadata.json"
fi

echo "Finished OPD at $(date)"

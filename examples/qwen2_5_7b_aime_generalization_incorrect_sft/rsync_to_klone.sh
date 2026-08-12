#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --check, --transfer, or --verify}"
[[ "${MODE}" == "--check" || "${MODE}" == "--transfer" || "${MODE}" == "--verify" ]] || {
  echo "Usage: $0 --check|--transfer|--verify" >&2
  exit 2
}
[[ $# -eq 1 ]] || { echo "Usage: $0 --check|--transfer|--verify" >&2; exit 2; }
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${PACKAGE}/env.sh"
EVAL_DIR="${PACKAGE}/eval"
CHECKPOINT_INVENTORY="${HANDOFF_ROOT}/checkpoint_sources.json"
COMPARISON_INVENTORY="${HANDOFF_ROOT}/comparison_sources.json"
MANIFEST_TOOL="${SLIME_REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
HF_GATE="${SLIME_REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/hf_checkpoint_gate.py"
REMOTE_HOST="${REMOTE_HOST:-suryadv@klone.hyak.uw.edu}"
REMOTE_REPO="${REMOTE_REPO:-/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$(dirname "${REMOTE_REPO}")/Axolotl-Masked-SFT/.venv/bin/python}"
REMOTE_EXPERIMENT_RELATIVE="checkpoints/qwen2_5_7b_aime_generalization_incorrect_sft/v1/${CONTRACT_HASH}"
REMOTE_EXPERIMENT="${REMOTE_REPO}/${REMOTE_EXPERIMENT_RELATIVE}"
REMOTE_HF_GATE="${REMOTE_REPO}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/hf_checkpoint_gate.py"
REMOTE_HELD_IN="${REMOTE_EXPERIMENT}/data/eval/held_in.jsonl"
REMOTE_HELD_OUT="${REMOTE_EXPERIMENT}/data/eval/held_out_problem.jsonl"
SOURCE_HELD_IN="${HELD_IN_EVAL_JSONL}"
SOURCE_HELD_OUT="${HELD_OUT_PROBLEM_EVAL_JSONL}"
BASE_SOURCE_CHECKPOINT="${STUDENT_HF_DIR}"
BASE_SOURCE_MANIFEST="${BASE_CHECKPOINT_MANIFEST}"
BASE_REMOTE_CHECKPOINT="${REMOTE_EXPERIMENT}/models/Qwen2.5-7B"
BASE_REMOTE_MANIFEST="${REMOTE_EXPERIMENT}/handoff/checkpoints/base_qwen2_5_7b.json"

for required in \
  "${CHECKPOINT_INVENTORY}" "${COMPARISON_INVENTORY}" \
  "${SOURCE_HELD_IN}" "${SOURCE_HELD_OUT}" \
  "${BASE_SOURCE_MANIFEST}" "${BASE_SOURCE_CHECKPOINT}/config.json"; do
  [[ -e "${required}" ]] || { echo "Missing finalized handoff artifact: ${required}" >&2; exit 1; }
done

python3 "${EVAL_DIR}/verify_transfer.py" \
  --held-in "${SOURCE_HELD_IN}" --held-out "${SOURCE_HELD_OUT}" \
  --checkpoint-inventory "${CHECKPOINT_INVENTORY}" \
  --comparison-inventory "${COMPARISON_INVENTORY}"

mapfile -t records < <(
  jq -r '.checkpoints[] | [.id,.source_checkpoint,.source_manifest,.remote_checkpoint_relative,.remote_manifest_relative] | @tsv' "${CHECKPOINT_INVENTORY}"
)
[[ "${#records[@]}" -eq 12 ]] || { echo "Expected exactly 12 trained checkpoint records" >&2; exit 1; }

verify_remote_hash() {
  local source="${1:?source required}" remote="${2:?remote required}"
  local expected actual
  expected="$(sha256sum "${source}" | awk '{print $1}')"
  actual="$(ssh "${REMOTE_HOST}" sha256sum -- "${remote}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "Remote hash mismatch: ${remote} expected=${expected} actual=${actual}" >&2
    return 1
  }
}

for record in "${records[@]}"; do
  IFS=$'\t' read -r artifact_id source_checkpoint source_manifest remote_checkpoint_relative remote_manifest_relative <<<"${record}"
  remote_checkpoint="${REMOTE_EXPERIMENT}/${remote_checkpoint_relative}"
  remote_manifest="${REMOTE_EXPERIMENT}/${remote_manifest_relative}"
  case "${MODE}" in
    --check)
      python3 "${HF_GATE}" --checkpoint "${source_checkpoint}"
      python3 "${MANIFEST_TOOL}" verify --manifest "${source_manifest}" --checkpoint "${source_checkpoint}"
      ;;
    --transfer)
      ssh "${REMOTE_HOST}" mkdir -p "${remote_checkpoint}" "$(dirname "${remote_manifest}")"
      rsync -a --partial --append-verify --info=progress2 \
        "${source_checkpoint}/" "${REMOTE_HOST}:${remote_checkpoint}/"
      rsync -a --partial --append-verify --info=progress2 \
        "${source_manifest}" "${REMOTE_HOST}:${remote_manifest}"
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" \
        --checkpoint "${remote_checkpoint}"
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
        "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "${remote_manifest}" --checkpoint "${remote_checkpoint}"
      ;;
    --verify)
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" \
        --checkpoint "${remote_checkpoint}"
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
        "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "${remote_manifest}" --checkpoint "${remote_checkpoint}"
      ;;
  esac
  echo "AIME_RSYNC_${MODE#--} artifact=${artifact_id}"
done

case "${MODE}" in
  --check)
    python3 "${HF_GATE}" --checkpoint "${BASE_SOURCE_CHECKPOINT}"
    python3 "${MANIFEST_TOOL}" verify \
      --manifest "${BASE_SOURCE_MANIFEST}" --checkpoint "${BASE_SOURCE_CHECKPOINT}"
    ;;
  --transfer)
    ssh "${REMOTE_HOST}" mkdir -p "${REMOTE_EXPERIMENT}/models" "$(dirname "${BASE_REMOTE_MANIFEST}")"
    rsync -a --partial --append-verify --info=progress2 \
      "${BASE_SOURCE_MANIFEST}" "${REMOTE_HOST}:${BASE_REMOTE_MANIFEST}"
    if ssh "${REMOTE_HOST}" test -f "${BASE_REMOTE_CHECKPOINT}/config.json" && \
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" \
        --checkpoint "${BASE_REMOTE_CHECKPOINT}" && \
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
        "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "${BASE_REMOTE_MANIFEST}" --checkpoint "${BASE_REMOTE_CHECKPOINT}"; then
      echo "AIME_RSYNC_BASE_REUSED remote=${BASE_REMOTE_CHECKPOINT}"
    else
      ssh "${REMOTE_HOST}" mkdir -p "${BASE_REMOTE_CHECKPOINT}"
      rsync -a --delete-delay --partial --append-verify --info=progress2 \
        "${BASE_SOURCE_CHECKPOINT}/" "${REMOTE_HOST}:${BASE_REMOTE_CHECKPOINT}/"
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" \
        --checkpoint "${BASE_REMOTE_CHECKPOINT}"
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
        "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "${BASE_REMOTE_MANIFEST}" --checkpoint "${BASE_REMOTE_CHECKPOINT}"
      echo "AIME_RSYNC_BASE_TRANSFERRED remote=${BASE_REMOTE_CHECKPOINT}"
    fi
    ;;
  --verify)
    ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" \
      --checkpoint "${BASE_REMOTE_CHECKPOINT}"
    ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
      "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
      verify --manifest "${BASE_REMOTE_MANIFEST}" --checkpoint "${BASE_REMOTE_CHECKPOINT}"
    ;;
esac

compact_records=(
  "${EXPERIMENT_ROOT}/TRAINING_STATUS.json|TRAINING_STATUS.json"
  "${CONTRACT_FILE}|config/experiment_contract.json"
  "${SOURCE_ARTIFACTS_JSON}|data/source_artifacts.json"
  "${PREP_STATS_JSON}|data/selection_and_tokenization_stats.json"
  "${VARIANT_STATS_JSON}|data/variant_stats.json"
  "${TOKENIZER_INVENTORY_JSON}|data/tokenizer_control_inventory.json"
  "${SCHEDULE_AUDIT_JSON}|data/schedule_audit.json"
  "${MANIFEST_ROOT}/training_chain.json|manifests/training_chain.json"
  "${CHECKPOINT_INVENTORY}|handoff/checkpoint_sources.json"
  "${COMPARISON_INVENTORY}|handoff/comparison_sources.json"
)
for record in "${compact_records[@]}"; do
  IFS='|' read -r source relative <<<"${record}"
  [[ -f "${source}" ]] || { echo "Missing compact provenance: ${source}" >&2; exit 1; }
  remote="${REMOTE_EXPERIMENT}/${relative}"
  case "${MODE}" in
    --check) sha256sum "${source}" >/dev/null ;;
    --transfer)
      ssh "${REMOTE_HOST}" mkdir -p "$(dirname "${remote}")"
      rsync -a --partial --append-verify --info=progress2 \
        "${source}" "${REMOTE_HOST}:${remote}"
      ;;
    --verify) verify_remote_hash "${source}" "${remote}" ;;
  esac
done

case "${MODE}" in
  --check)
    sha256sum "${SOURCE_HELD_IN}" "${SOURCE_HELD_OUT}"
    ;;
  --transfer)
    ssh "${REMOTE_HOST}" mkdir -p "${REMOTE_EXPERIMENT}/data/eval"
    rsync -a --partial --append-verify --info=progress2 \
      "${SOURCE_HELD_IN}" "${REMOTE_HOST}:${REMOTE_HELD_IN}"
    rsync -a --partial --append-verify --info=progress2 \
      "${SOURCE_HELD_OUT}" "${REMOTE_HOST}:${REMOTE_HELD_OUT}"
    ;;
  --verify)
    verify_remote_hash "${SOURCE_HELD_IN}" "${REMOTE_HELD_IN}"
    verify_remote_hash "${SOURCE_HELD_OUT}" "${REMOTE_HELD_OUT}"
    ;;
esac

echo "AIME_RSYNC_COMPLETE mode=${MODE} trained_checkpoints=12 eval_jsonl=2 base_policy=transfer_only_if_missing training_jsonl=0 optimizer_state=0"

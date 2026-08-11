#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --check, --transfer, or --verify}"
[[ "${MODE}" == "--check" || "${MODE}" == "--transfer" || "${MODE}" == "--verify" ]] || {
  echo "Usage: $0 --check|--transfer|--verify" >&2
  exit 2
}
[[ $# -eq 1 ]] || { echo "Usage: $0 --check|--transfer|--verify" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
INVENTORY="${HANDOFF_ROOT}/checkpoint_sources.json"
MANIFEST_TOOL="${SLIME_REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
REMOTE_HOST="${REMOTE_HOST:-suryadv@klone.hyak.uw.edu}"
REMOTE_REPO="${REMOTE_REPO:-/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$(dirname "${REMOTE_REPO}")/Axolotl-Masked-SFT/.venv/bin/python}"
REMOTE_EXPERIMENT="${REMOTE_REPO}/checkpoints/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/v1/${CONTRACT_HASH}"

[[ -f "${INVENTORY}" ]] || {
  echo "Missing finalized checkpoint inventory ${INVENTORY}" >&2
  exit 1
}
mapfile -t records < <(
  jq -r '.checkpoints[] | [.id,.source_checkpoint,.source_manifest,.remote_checkpoint_relative,.remote_manifest_relative] | @tsv' "${INVENTORY}"
)
[[ "${#records[@]}" -eq 6 ]] || { echo "Expected 6 new handoff records" >&2; exit 1; }

for record in "${records[@]}"; do
  IFS=$'\t' read -r artifact_id source_checkpoint source_manifest remote_checkpoint_relative remote_manifest_relative <<<"${record}"
  remote_checkpoint="${REMOTE_EXPERIMENT}/${remote_checkpoint_relative}"
  remote_manifest="${REMOTE_EXPERIMENT}/${remote_manifest_relative}"
  case "${MODE}" in
    --check)
      python3 "${MANIFEST_TOOL}" verify \
        --manifest "${source_manifest}" --checkpoint "${source_checkpoint}"
      ;;
    --transfer)
      ssh "${REMOTE_HOST}" mkdir -p "${remote_checkpoint}" "$(dirname "${remote_manifest}")"
      rsync -a --partial --append-verify --info=progress2 \
        "${source_checkpoint}/" "${REMOTE_HOST}:${remote_checkpoint}/"
      rsync -a --partial --append-verify --info=progress2 \
        "${source_manifest}" "${REMOTE_HOST}:${remote_manifest}"
      ;;
    --verify)
      ssh "${REMOTE_HOST}" "${REMOTE_PYTHON}" \
        "${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "${remote_manifest}" --checkpoint "${remote_checkpoint}"
      ;;
  esac
  echo "RSYNC_REPLICATE_${MODE#--} artifact=${artifact_id}"
done

if [[ "${MODE}" == "--transfer" ]]; then
  ssh "${REMOTE_HOST}" mkdir -p \
    "${REMOTE_EXPERIMENT}/config" "${REMOTE_EXPERIMENT}/data" \
    "${REMOTE_EXPERIMENT}/handoff"
  rsync -a --partial --append-verify --info=progress2 \
    "${EXPERIMENT_ROOT}/TRAINING_STATUS.json" \
    "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/TRAINING_STATUS.json"
  rsync -a --partial --append-verify --info=progress2 \
    "${CONTRACT_FILE}" \
    "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/config/experiment_contract.json"
  for relative in \
    data/schedule_audit.json data/selection_and_tokenization_stats.json \
    data/source_artifacts.json data/tokenizer_control_inventory.json \
    data/variant_stats.json handoff/checkpoint_sources.json \
    handoff/comparison_sources.json; do
    rsync -a --partial --append-verify --info=progress2 \
      "${EXPERIMENT_ROOT}/${relative}" \
      "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/${relative}"
  done
  echo "RSYNC_REPLICATE_METADATA_TRANSFERRED files=9"
fi

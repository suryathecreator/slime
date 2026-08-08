#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --check, --transfer, or --verify}"
[[ "${MODE}" == "--check" || "${MODE}" == "--transfer" || "${MODE}" == "--verify" ]] || {
  echo "Usage: $0 --check|--transfer|--verify" >&2
  exit 2
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
INVENTORY="${HANDOFF_ROOT}/checkpoint_sources.json"
MANIFEST_TOOL="${SLIME_REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
REMOTE_HOST="${REMOTE_HOST:-suryadv@klone.hyak.uw.edu}"
REMOTE_REPO="${REMOTE_REPO:-/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$(dirname "${REMOTE_REPO}")/Axolotl-Masked-SFT/.venv/bin/python}"
REMOTE_EXPERIMENT="${REMOTE_REPO}/checkpoints/qwen_correct_only_openr1_math220k_sft_2k/v1/${CONTRACT_HASH}"

[[ -f "${INVENTORY}" ]] || {
  echo "Missing finalized checkpoint inventory ${INVENTORY}" >&2
  exit 1
}
mapfile -t records < <(
  jq -r '.checkpoints[] | [.id,.source_checkpoint,.source_manifest,.remote_checkpoint_relative,.remote_manifest_relative] | @tsv' "${INVENTORY}"
)
[[ "${#records[@]}" -eq 9 ]] || { echo "Expected 9 handoff records" >&2; exit 1; }

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
  echo "RSYNC_HANDOFF_${MODE#--} artifact=${artifact_id}"
done

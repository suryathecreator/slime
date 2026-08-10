#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Use --check, --transfer, or --verify}"
[[ "$MODE" == "--check" || "$MODE" == "--transfer" || "$MODE" == "--verify" ]] || {
  echo "Usage: $0 --check|--transfer|--verify" >&2
  exit 2
}
HANDOFF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$HANDOFF/../.." >/dev/null 2>&1 && pwd)"
INVENTORY="$HANDOFF/checkpoint_sources.json"
MANIFEST_TOOL="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
REMOTE_HOST="${REMOTE_HOST:-suryadv@klone.hyak.uw.edu}"
REMOTE_REPO="${REMOTE_REPO:-/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$(dirname "$REMOTE_REPO")/Axolotl-Masked-SFT/.venv/bin/python}"
REMOTE_CHECKPOINT_ROOT="$REMOTE_REPO/checkpoints"
REMOTE_EXPERIMENT="$REMOTE_CHECKPOINT_ROOT/qwen2_5_7b_openr1_math220k_masked_sft_2k/v1/6cfd7bd235e078b8"

mapfile -t records < <(
  jq -r '.checkpoints[] | [.variant,.source_checkpoint,.source_manifest] | @tsv' "$INVENTORY"
)
for record in "${records[@]}"; do
  IFS=$'\t' read -r variant source_checkpoint source_manifest <<<"$record"
  if [[ "$variant" == "base_7b" ]]; then
    remote_checkpoint="$REMOTE_CHECKPOINT_ROOT/models/Qwen2.5-7B"
  else
    remote_checkpoint="$REMOTE_EXPERIMENT/outputs/training/$variant/weights/iter_0000009"
  fi
  remote_manifest="$REMOTE_EXPERIMENT/handoff/checkpoints/$variant.json"
  case "$MODE" in
    --check)
      python3 "$MANIFEST_TOOL" verify \
        --manifest "$source_manifest" --checkpoint "$source_checkpoint"
      ;;
    --transfer)
      ssh "$REMOTE_HOST" mkdir -p "$remote_checkpoint" "$(dirname "$remote_manifest")"
      rsync -a --partial --append-verify --info=progress2 \
        "$source_checkpoint/" "$REMOTE_HOST:$remote_checkpoint/"
      rsync -a --partial --append-verify --info=progress2 \
        "$source_manifest" "$REMOTE_HOST:$remote_manifest"
      ;;
    --verify)
      ssh "$REMOTE_HOST" "$REMOTE_PYTHON" \
        "$REMOTE_REPO/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py" \
        verify --manifest "$remote_manifest" --checkpoint "$remote_checkpoint"
      ;;
  esac
  echo "RSYNC_HANDOFF_${MODE#--} variant=$variant"
done

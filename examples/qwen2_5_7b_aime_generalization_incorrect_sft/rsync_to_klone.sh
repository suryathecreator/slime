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
REMOTE_MANIFEST_TOOL="${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
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
  "${PACKAGE}/provenance/v1/${CONTRACT_HASH}/PROVENANCE_PUBLICATION.json|provenance/PROVENANCE_PUBLICATION.json"
  "${PACKAGE}/provenance/v1/${CONTRACT_HASH}/TRAINING_RESULTS.md|provenance/TRAINING_RESULTS.md"
  "${PACKAGE}/provenance/v1/${CONTRACT_HASH}/training_metrics.json|provenance/training_metrics.json"
)
for record in "${compact_records[@]}"; do
  IFS='|' read -r source _relative <<<"${record}"
  [[ -f "${source}" ]] || { echo "Missing compact provenance: ${source}" >&2; exit 1; }
done

if [[ "${MODE}" == "--check" ]]; then
  for record in "${records[@]}"; do
    IFS=$'\t' read -r artifact_id source_checkpoint source_manifest _remote_checkpoint _remote_manifest <<<"${record}"
    python3 "${HF_GATE}" --checkpoint "${source_checkpoint}"
    python3 "${MANIFEST_TOOL}" verify --manifest "${source_manifest}" --checkpoint "${source_checkpoint}"
    echo "AIME_RSYNC_check artifact=${artifact_id}"
  done
  python3 "${HF_GATE}" --checkpoint "${BASE_SOURCE_CHECKPOINT}"
  python3 "${MANIFEST_TOOL}" verify \
    --manifest "${BASE_SOURCE_MANIFEST}" --checkpoint "${BASE_SOURCE_CHECKPOINT}"
  for record in "${compact_records[@]}"; do
    IFS='|' read -r source _relative <<<"${record}"
    sha256sum "${source}" >/dev/null
  done
  sha256sum "${SOURCE_HELD_IN}" "${SOURCE_HELD_OUT}"
  echo "AIME_RSYNC_COMPLETE mode=--check trained_checkpoints=12 eval_jsonl=2 training_jsonl=0 optimizer_state=0"
  exit 0
fi

if [[ "${MODE}" == "--transfer" ]]; then
  SSH_CONTROL_DIR="$(mktemp -d /tmp/q25a-rsync-ssh.XXXXXX)"
  SSH_CONTROL_SOCKET="${SSH_CONTROL_DIR}/control"
  cleanup_transfer_session() {
    if [[ -S "${SSH_CONTROL_SOCKET}" ]]; then
      ssh -S "${SSH_CONTROL_SOCKET}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
    fi
    rmdir "${SSH_CONTROL_DIR}" 2>/dev/null || true
  }
  trap cleanup_transfer_session EXIT
  ssh -M -S "${SSH_CONTROL_SOCKET}" -o ControlPersist=no \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -fN "${REMOTE_HOST}"
  SSH_SESSION=(ssh -S "${SSH_CONTROL_SOCKET}")
  RSYNC_RSH="ssh -S ${SSH_CONTROL_SOCKET} -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
  echo "AIME_RSYNC_TRANSFER_SESSION_OPEN authentication_prompts=1 resume_partial=1"

  remote_directories=(
    "${REMOTE_EXPERIMENT}/config"
    "${REMOTE_EXPERIMENT}/data/eval"
    "${REMOTE_EXPERIMENT}/handoff/checkpoints"
    "${REMOTE_EXPERIMENT}/manifests"
    "${REMOTE_EXPERIMENT}/models/Qwen2.5-7B"
    "${REMOTE_EXPERIMENT}/provenance"
  )
  for record in "${records[@]}"; do
    IFS=$'\t' read -r _artifact_id _source_checkpoint _source_manifest remote_checkpoint_relative remote_manifest_relative <<<"${record}"
    remote_directories+=(
      "${REMOTE_EXPERIMENT}/${remote_checkpoint_relative}"
      "$(dirname "${REMOTE_EXPERIMENT}/${remote_manifest_relative}")"
    )
  done
  "${SSH_SESSION[@]}" "${REMOTE_HOST}" mkdir -p "${remote_directories[@]}"

  for record in "${records[@]}"; do
    IFS=$'\t' read -r artifact_id source_checkpoint source_manifest remote_checkpoint_relative remote_manifest_relative <<<"${record}"
    remote_checkpoint="${REMOTE_EXPERIMENT}/${remote_checkpoint_relative}"
    remote_manifest="${REMOTE_EXPERIMENT}/${remote_manifest_relative}"
    rsync -a --delete-delay --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
      "${source_checkpoint}/" "${REMOTE_HOST}:${remote_checkpoint}/"
    rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
      "${source_manifest}" "${REMOTE_HOST}:${remote_manifest}"
    echo "AIME_RSYNC_transfer artifact=${artifact_id} remote_verification=pending"
  done

  rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
    "${BASE_SOURCE_MANIFEST}" "${REMOTE_HOST}:${BASE_REMOTE_MANIFEST}"
  rsync -a --delete-delay --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
    "${BASE_SOURCE_CHECKPOINT}/" "${REMOTE_HOST}:${BASE_REMOTE_CHECKPOINT}/"
  echo "AIME_RSYNC_transfer artifact=base_qwen2_5_7b remote_verification=pending"

  for record in "${compact_records[@]}"; do
    IFS='|' read -r source relative <<<"${record}"
    rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
      "${source}" "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/${relative}"
  done
  rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
    "${SOURCE_HELD_IN}" "${REMOTE_HOST}:${REMOTE_HELD_IN}"
  rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}" \
    "${SOURCE_HELD_OUT}" "${REMOTE_HOST}:${REMOTE_HELD_OUT}"

  cleanup_transfer_session
  trap - EXIT
  echo "AIME_RSYNC_COMPLETE mode=--transfer authenticated_sessions=1 trained_checkpoints=12 eval_jsonl=2 training_jsonl=0 optimizer_state=0 remote_verification=required_separate_command"
  exit 0
fi

hash_records=(
  "${compact_records[@]}"
  "${SOURCE_HELD_IN}|data/eval/held_in.jsonl"
  "${SOURCE_HELD_OUT}|data/eval/held_out_problem.jsonl"
)
remote_hash_relatives=()
for record in "${hash_records[@]}"; do
  IFS='|' read -r _source relative <<<"${record}"
  remote_hash_relatives+=("${relative}")
done

VERIFY_OUTPUT="$(mktemp /tmp/q25a-rsync-verify.XXXXXX)"
cleanup_verify_output() {
  rm -f -- "${VERIFY_OUTPUT}"
}
trap cleanup_verify_output EXIT
ssh "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" "${REMOTE_MANIFEST_TOOL}" "${REMOTE_EXPERIMENT}" \
  "${remote_hash_relatives[@]}" <<'REMOTE_VERIFY' | tee "${VERIFY_OUTPUT}"
set -euo pipefail
remote_python="$1"
remote_hf_gate="$2"
remote_manifest_tool="$3"
remote_experiment="$4"
shift 4

for required in "${remote_python}" "${remote_hf_gate}" "${remote_manifest_tool}" \
  "${remote_experiment}/handoff/checkpoint_sources.json"; do
  [[ -e "${required}" ]] || { echo "Missing remote verification prerequisite: ${required}" >&2; exit 1; }
done

mapfile -t checkpoint_rows < <(
  "${remote_python}" -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for item in value["checkpoints"]:
    print("\t".join((item["id"], item["remote_checkpoint_relative"], item["remote_manifest_relative"])))
' "${remote_experiment}/handoff/checkpoint_sources.json"
)
[[ "${#checkpoint_rows[@]}" -eq 12 ]] || { echo "Remote inventory does not contain 12 checkpoints" >&2; exit 1; }
for row in "${checkpoint_rows[@]}"; do
  IFS=$'\t' read -r artifact_id checkpoint_relative manifest_relative <<<"${row}"
  checkpoint="${remote_experiment}/${checkpoint_relative}"
  manifest="${remote_experiment}/${manifest_relative}"
  "${remote_python}" "${remote_hf_gate}" --checkpoint "${checkpoint}"
  "${remote_python}" "${remote_manifest_tool}" verify --manifest "${manifest}" --checkpoint "${checkpoint}"
  echo "AIME_RSYNC_verify artifact=${artifact_id}"
done

base_checkpoint="${remote_experiment}/models/Qwen2.5-7B"
base_manifest="${remote_experiment}/handoff/checkpoints/base_qwen2_5_7b.json"
"${remote_python}" "${remote_hf_gate}" --checkpoint "${base_checkpoint}"
"${remote_python}" "${remote_manifest_tool}" verify --manifest "${base_manifest}" --checkpoint "${base_checkpoint}"
echo "AIME_RSYNC_verify artifact=base_qwen2_5_7b"

for relative in "$@"; do
  remote_file="${remote_experiment}/${relative}"
  [[ -f "${remote_file}" ]] || { echo "Missing transferred file: ${remote_file}" >&2; exit 1; }
  actual="$(sha256sum -- "${remote_file}" | awk '{print $1}')"
  printf 'AIME_REMOTE_SHA256\t%s\t%s\n' "${actual}" "${relative}"
done
REMOTE_VERIFY

for record in "${hash_records[@]}"; do
  IFS='|' read -r source relative <<<"${record}"
  expected="$(sha256sum "${source}" | awk '{print $1}')"
  expected_line="$(printf 'AIME_REMOTE_SHA256\t%s\t%s' "${expected}" "${relative}")"
  grep -Fqx -- "${expected_line}" "${VERIFY_OUTPUT}" || {
    echo "Remote hash mismatch: ${relative}" >&2
    exit 1
  }
done
cleanup_verify_output
trap - EXIT
echo "AIME_RSYNC_COMPLETE mode=--verify authenticated_sessions=1 trained_checkpoints=12 eval_jsonl=2 training_jsonl=0 optimizer_state=0"

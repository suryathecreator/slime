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
CHECKPOINT_INVENTORY="${HANDOFF_ROOT}/checkpoint_sources.json"
COMPARISON_INVENTORY="${HANDOFF_ROOT}/comparison_sources.json"
MANIFEST_TOOL="${SLIME_REPO_ROOT}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
HF_GATE="${SLIME_REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/hf_checkpoint_gate.py"
REMOTE_HOST="${REMOTE_HOST:-suryadv@klone.hyak.uw.edu}"
REMOTE_REPO="${REMOTE_REPO:-/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$(dirname "${REMOTE_REPO}")/Axolotl-Masked-SFT/.venv/bin/python}"
REMOTE_EXPERIMENT_RELATIVE="checkpoints/qwen2_5_7b_nous_aime_mixed_outcome_sft/v1/${CONTRACT_HASH}"
REMOTE_EXPERIMENT="${REMOTE_REPO}/${REMOTE_EXPERIMENT_RELATIVE}"
REMOTE_HF_GATE="${REMOTE_REPO}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/hf_checkpoint_gate.py"
REMOTE_MANIFEST_TOOL="${REMOTE_REPO}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
REMOTE_SHARED_BASE="${REMOTE_REPO}/checkpoints/shared/models/Qwen2.5-7B"
REMOTE_SHARED_BASE_MANIFEST="${REMOTE_REPO}/checkpoints/shared/manifests/base_qwen2_5_7b.json"

for required in \
  "${CHECKPOINT_INVENTORY}" "${COMPARISON_INVENTORY}" \
  "${EXPERIMENT_ROOT}/TRAINING_STATUS.json" "${HANDOFF_ROOT}/eval_contract.json" \
  "${AIME24_EVAL_JSONL}" "${AIME25_EVAL_JSONL}" \
  "${BASE_CHECKPOINT_MANIFEST}" "${STUDENT_HF_DIR}/config.json"; do
  [[ -e "${required}" ]] || { echo "Missing finalized handoff artifact: ${required}" >&2; exit 1; }
done

mapfile -t records < <(
  jq -r '.checkpoints[] | [.id,.source_checkpoint,.source_manifest,.remote_checkpoint_relative,.remote_manifest_relative] | @tsv' "${CHECKPOINT_INVENTORY}"
)
[[ "${#records[@]}" -eq 4 ]] || { echo "Expected exactly four trained checkpoints" >&2; exit 1; }

compact_records=(
  "${EXPERIMENT_ROOT}/TRAINING_STATUS.json|TRAINING_STATUS.json"
  "${CONTRACT_FILE}|config/experiment_contract.json"
  "${AIME24_EVAL_JSONL}|data/eval/aime24_30.jsonl"
  "${AIME25_EVAL_JSONL}|data/eval/aime25_30.jsonl"
  "${SCHEDULE_AUDIT_JSON}|data/schedule_audit.json"
  "${PREP_STATS_JSON}|data/selection_and_tokenization_stats.json"
  "${SOURCE_ARTIFACTS_JSON}|data/source_artifacts.json"
  "${TOKENIZER_INVENTORY_JSON}|data/tokenizer_control_inventory.json"
  "${HANDOFF_ROOT}/EVAL_HANDOFF.md|handoff/EVAL_HANDOFF.md"
  "${CHECKPOINT_INVENTORY}|handoff/checkpoint_sources.json"
  "${COMPARISON_INVENTORY}|handoff/comparison_sources.json"
  "${HANDOFF_ROOT}/eval_contract.json|handoff/eval_contract.json"
  "${HANDOFF_ROOT}/training_metrics.json|handoff/training_metrics.json"
  "${MANIFEST_ROOT}/training_chain.json|manifests/training_chain.json"
)
for record in "${compact_records[@]}"; do
  IFS='|' read -r source _relative <<<"${record}"
  [[ -f "${source}" ]] || { echo "Missing compact provenance: ${source}" >&2; exit 1; }
done

local_verify() {
  for record in "${records[@]}"; do
    IFS=$'\t' read -r artifact_id checkpoint manifest _remote_checkpoint _remote_manifest <<<"${record}"
    python3 "${HF_GATE}" --checkpoint "${checkpoint}"
    python3 "${MANIFEST_TOOL}" verify --manifest "${manifest}" --checkpoint "${checkpoint}"
    echo "NOUS_AIME_RSYNC_check artifact=${artifact_id}"
  done
  python3 "${HF_GATE}" --checkpoint "${STUDENT_HF_DIR}"
  python3 "${MANIFEST_TOOL}" verify \
    --manifest "${BASE_CHECKPOINT_MANIFEST}" --checkpoint "${STUDENT_HF_DIR}"
  for record in "${compact_records[@]}"; do
    IFS='|' read -r source _relative <<<"${record}"
    sha256sum "${source}" >/dev/null
  done
}

if [[ "${MODE}" == "--check" ]]; then
  local_verify
  echo "NOUS_AIME_RSYNC_COMPLETE mode=--check trained=4 eval_jsonl=2 training_jsonl=0 optimizer_state=0"
  exit 0
fi

if [[ "${MODE}" == "--transfer" ]]; then
  local_verify
  control_dir="$(mktemp -d /tmp/q25nam-rsync-ssh.XXXXXX)"
  control_socket="${control_dir}/control"
  cleanup_session() {
    if [[ -S "${control_socket}" ]]; then
      ssh -S "${control_socket}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
    fi
    rmdir "${control_dir}" 2>/dev/null || true
  }
  trap cleanup_session EXIT
  ssh -M -S "${control_socket}" -o ControlPersist=no \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -fN "${REMOTE_HOST}"
  ssh_session=(ssh -S "${control_socket}")
  rsync_rsh="ssh -S ${control_socket} -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
  echo "NOUS_AIME_RSYNC_TRANSFER_SESSION_OPEN authentication_prompts=1"
  remote_directories=(
    "${REMOTE_EXPERIMENT}/config" "${REMOTE_EXPERIMENT}/data/eval"
    "${REMOTE_EXPERIMENT}/handoff/checkpoints" "${REMOTE_EXPERIMENT}/manifests"
    "${REMOTE_REPO}/checkpoints/shared/models/Qwen2.5-7B"
    "${REMOTE_REPO}/checkpoints/shared/manifests"
  )
  for record in "${records[@]}"; do
    IFS=$'\t' read -r _id _checkpoint _manifest checkpoint_relative manifest_relative <<<"${record}"
    remote_directories+=(
      "${REMOTE_EXPERIMENT}/${checkpoint_relative}"
      "$(dirname "${REMOTE_EXPERIMENT}/${manifest_relative}")"
    )
  done
  "${ssh_session[@]}" "${REMOTE_HOST}" mkdir -p "${remote_directories[@]}"
  for record in "${records[@]}"; do
    IFS=$'\t' read -r artifact_id checkpoint manifest checkpoint_relative manifest_relative <<<"${record}"
    rsync -a --delete-delay --partial --append-verify --info=progress2 -e "${rsync_rsh}" \
      "${checkpoint}/" "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/${checkpoint_relative}/"
    rsync -a --partial --append-verify --info=progress2 -e "${rsync_rsh}" \
      "${manifest}" "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/${manifest_relative}"
    echo "NOUS_AIME_RSYNC_transfer artifact=${artifact_id}"
  done
  rsync -a --partial --append-verify --info=progress2 -e "${rsync_rsh}" \
    "${BASE_CHECKPOINT_MANIFEST}" "${REMOTE_HOST}:${REMOTE_SHARED_BASE_MANIFEST}"
  if "${ssh_session[@]}" "${REMOTE_HOST}" \
      "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" --checkpoint "${REMOTE_SHARED_BASE}" >/dev/null 2>&1 && \
     "${ssh_session[@]}" "${REMOTE_HOST}" \
      "${REMOTE_PYTHON}" "${REMOTE_MANIFEST_TOOL}" verify \
      --manifest "${REMOTE_SHARED_BASE_MANIFEST}" --checkpoint "${REMOTE_SHARED_BASE}" >/dev/null 2>&1; then
    echo "NOUS_AIME_RSYNC_reuse artifact=base_qwen2_5_7b verified=1"
  else
    rsync -a --delete-delay --partial --append-verify --info=progress2 -e "${rsync_rsh}" \
      "${STUDENT_HF_DIR}/" "${REMOTE_HOST}:${REMOTE_SHARED_BASE}/"
    echo "NOUS_AIME_RSYNC_transfer artifact=base_qwen2_5_7b"
  fi
  for record in "${compact_records[@]}"; do
    IFS='|' read -r source relative <<<"${record}"
    rsync -a --partial --append-verify --info=progress2 -e "${rsync_rsh}" \
      "${source}" "${REMOTE_HOST}:${REMOTE_EXPERIMENT}/${relative}"
  done
  cleanup_session
  trap - EXIT
  echo "NOUS_AIME_RSYNC_COMPLETE mode=--transfer authenticated_sessions=1 trained=4 eval_jsonl=2 training_jsonl=0 optimizer_state=0 verification=run_--verify"
  exit 0
fi

verify_output="$(mktemp /tmp/q25nam-rsync-verify.XXXXXX)"
trap 'rm -f -- "${verify_output}"' EXIT
remote_relatives=()
for record in "${compact_records[@]}"; do
  IFS='|' read -r _source relative <<<"${record}"
  remote_relatives+=("${relative}")
done
ssh "${REMOTE_HOST}" bash -s -- \
  "${REMOTE_PYTHON}" "${REMOTE_HF_GATE}" "${REMOTE_MANIFEST_TOOL}" \
  "${REMOTE_EXPERIMENT}" "${REMOTE_SHARED_BASE}" "${REMOTE_SHARED_BASE_MANIFEST}" \
  "${remote_relatives[@]}" <<'REMOTE_VERIFY' | tee "${verify_output}"
set -euo pipefail
remote_python="$1"
remote_hf_gate="$2"
remote_manifest_tool="$3"
remote_experiment="$4"
remote_base="$5"
remote_base_manifest="$6"
shift 6
mapfile -t rows < <(
  "${remote_python}" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for item in value["checkpoints"]:
    print("\t".join((item["id"], item["remote_checkpoint_relative"], item["remote_manifest_relative"])))
' "${remote_experiment}/handoff/checkpoint_sources.json"
)
[[ "${#rows[@]}" -eq 4 ]] || { echo "Remote inventory does not contain four checkpoints" >&2; exit 1; }
for row in "${rows[@]}"; do
  IFS=$'\t' read -r artifact_id checkpoint_relative manifest_relative <<<"${row}"
  checkpoint="${remote_experiment}/${checkpoint_relative}"
  manifest="${remote_experiment}/${manifest_relative}"
  "${remote_python}" "${remote_hf_gate}" --checkpoint "${checkpoint}"
  "${remote_python}" "${remote_manifest_tool}" verify --manifest "${manifest}" --checkpoint "${checkpoint}"
  echo "NOUS_AIME_RSYNC_verify artifact=${artifact_id}"
done
"${remote_python}" "${remote_hf_gate}" --checkpoint "${remote_base}"
"${remote_python}" "${remote_manifest_tool}" verify \
  --manifest "${remote_base_manifest}" --checkpoint "${remote_base}"
echo "NOUS_AIME_RSYNC_verify artifact=base_qwen2_5_7b"
for relative in "$@"; do
  remote_file="${remote_experiment}/${relative}"
  [[ -f "${remote_file}" ]] || { echo "Missing transferred file: ${remote_file}" >&2; exit 1; }
  printf 'NOUS_AIME_REMOTE_SHA256\t%s\t%s\n' \
    "$(sha256sum -- "${remote_file}" | awk '{print $1}')" "${relative}"
done
REMOTE_VERIFY

for record in "${compact_records[@]}"; do
  IFS='|' read -r source relative <<<"${record}"
  expected="$(sha256sum "${source}" | awk '{print $1}')"
  grep -Fqx -- "$(printf 'NOUS_AIME_REMOTE_SHA256\t%s\t%s' "${expected}" "${relative}")" \
    "${verify_output}" || { echo "Remote hash mismatch: ${relative}" >&2; exit 1; }
done
rm -f -- "${verify_output}"
trap - EXIT
echo "NOUS_AIME_RSYNC_COMPLETE mode=--verify authenticated_sessions=1 trained=4 eval_jsonl=2 training_jsonl=0 optimizer_state=0"

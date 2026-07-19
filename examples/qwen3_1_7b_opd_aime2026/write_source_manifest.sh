#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

output_path="${1:-${SOURCE_MANIFEST}}"
mkdir -p "$(dirname -- "${output_path}")"
{
  find examples/qwen3_1_7b_opd_aime2026 -type f ! -path '*/__pycache__/*' -print
  printf '%s\n' \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch \
    examples/qwen3_8b_opd_tillicum/checkpoint_utils.sh \
    examples/qwen3_8b_opd_tillicum/container_exec.sh \
    examples/qwen3_8b_opd_tillicum/env.sh \
    examples/qwen3_8b_opd_tillicum/sglang_launch_native_rope.py \
    examples/qwen3_8b_opd_tillicum/summarize_opd_sanity.py \
    examples/qwen3_8b_opd_tillicum/validate_qwen3_generation.py \
    examples/qwen3_8b_opd_tillicum/write_opd_trained_manifest.py \
    scripts/models/qwen3-1.7B.sh
} | sort -u | xargs sha256sum >"${output_path}"
sha256sum -c "${output_path}"
echo "Wrote verified AIME source manifest: ${output_path}"

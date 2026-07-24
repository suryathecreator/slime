#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
exec "${SCRIPT_DIR}/../qwen3_8b_openr1_math220k_masked_sft_2k/submit_completion_chain.sh" "$@"

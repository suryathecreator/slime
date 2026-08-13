#!/usr/bin/env bash
set -euo pipefail

HANDOFF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HANDOFF/../qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff/runtime_python.sh" "$@"

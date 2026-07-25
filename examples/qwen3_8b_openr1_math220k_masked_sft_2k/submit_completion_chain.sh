#!/usr/bin/env bash
set -euo pipefail

echo "This train/eval launcher is retired. No eval jobs may be submitted here." >&2
echo "Use submit_training_only.sh; evaluation is deferred to the portable handoff." >&2
exit 2

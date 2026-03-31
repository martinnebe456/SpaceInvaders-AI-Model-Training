#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

# Silent evaluation is the lightweight way to compare checkpoints numerically.
activate_venv
ensure_mplconfig

# These environment variables let callers reuse the script for quick smoke
# tests or larger evaluation batches.
MODEL_PATH="${MODEL_PATH:-models/space_invaders_dqn.zip}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-20000}"

python -m space_invaders_rl.evaluate \
  --model-path "$MODEL_PATH" \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --render-mode none

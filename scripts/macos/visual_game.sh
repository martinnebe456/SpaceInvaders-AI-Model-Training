#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

activate_venv
ensure_mplconfig

MODEL_PATH="${MODEL_PATH:-models/space_invaders_dqn.zip}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"

python -m space_invaders_rl.evaluate \
  --model-path "$MODEL_PATH" \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --human-max-steps "$MAX_STEPS" \
  --render-mode human \
  --full-game

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

# Load the virtual environment before invoking the training module.
activate_venv
ensure_mplconfig

# Every setting can be overridden through environment variables so this script
# stays easy to customize without editing the file itself.
TIMESTEPS="${TIMESTEPS:-100000000}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20000}"
N_ENVS="${N_ENVS:-10}"
VEC_ENV="${VEC_ENV:-subproc}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
MODEL_PATH="${MODEL_PATH:-models/space_invaders_dqn}"
REPLAY_BUFFER_PATH="${REPLAY_BUFFER_PATH:-models/space_invaders_dqn_replay_buffer.pkl}"
RESET_REPLAY_BUFFER="${RESET_REPLAY_BUFFER:-0}"
LOG_DIR="${LOG_DIR:-runs/automation}"

mkdir -p "$LOG_DIR"
STAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/long_training_${STAMP}.log"

trap 'notify "Space Invaders RL" "Long training failed. Check log ${LOG_FILE}."' ERR

notify "Space Invaders RL" "Starting a long resumable training run for ${TIMESTEPS} timesteps."

if [[ "$RESET_REPLAY_BUFFER" == "1" ]]; then
  # Keep the model weights but intentionally discard old replay memory when
  # the reward function or environment semantics have changed.
  python -m space_invaders_rl.train \
    --timesteps "$TIMESTEPS" \
    --max-steps "$TRAIN_MAX_STEPS" \
    --n-envs "$N_ENVS" \
    --vec-env "$VEC_ENV" \
    --device "$TRAIN_DEVICE" \
    --model-path "$MODEL_PATH" \
    --replay-buffer-path "$REPLAY_BUFFER_PATH" \
    --skip-check-env \
    --reset-replay-buffer \
    2>&1 | tee "$LOG_FILE"
else
  # This is the normal path: continue training from the current checkpoint
  # and keep the replay buffer so learning can build on prior experience.
  python -m space_invaders_rl.train \
    --timesteps "$TIMESTEPS" \
    --max-steps "$TRAIN_MAX_STEPS" \
    --n-envs "$N_ENVS" \
    --vec-env "$VEC_ENV" \
    --device "$TRAIN_DEVICE" \
    --model-path "$MODEL_PATH" \
    --replay-buffer-path "$REPLAY_BUFFER_PATH" \
    --skip-check-env \
    2>&1 | tee "$LOG_FILE"
fi

notify "Space Invaders RL" "Long training finished."
echo "Log saved to $LOG_FILE"

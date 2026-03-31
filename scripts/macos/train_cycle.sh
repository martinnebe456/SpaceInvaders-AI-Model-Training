#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

# Load the local Python environment and keep Matplotlib cache files predictable.
activate_venv
ensure_mplconfig

# Environment variables make it easy to tweak the cycle without editing the
# script, which is useful for both local experimentation and automation.
TIMESTEPS="${TIMESTEPS:-1000000}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-20000}"
N_ENVS="${N_ENVS:-6}"
VEC_ENV="${VEC_ENV:-subproc}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
SILENT_EVAL_EPISODES="${SILENT_EVAL_EPISODES:-5}"
SILENT_EVAL_MAX_STEPS="${SILENT_EVAL_MAX_STEPS:-20000}"
HUMAN_EVAL_EPISODES="${HUMAN_EVAL_EPISODES:-1}"
HUMAN_EVAL_MAX_STEPS="${HUMAN_EVAL_MAX_STEPS:-20000}"
MODEL_PATH="${MODEL_PATH:-models/space_invaders_dqn}"
REPLAY_BUFFER_PATH="${REPLAY_BUFFER_PATH:-models/space_invaders_dqn_replay_buffer.pkl}"
RESET_REPLAY_BUFFER="${RESET_REPLAY_BUFFER:-0}"
LOG_DIR="${LOG_DIR:-runs/automation}"

mkdir -p "$LOG_DIR"
STAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/training_cycle_${STAMP}.log"

trap 'notify "Space Invaders RL" "Training cycle failed. Check log ${LOG_FILE}."' ERR

notify "Space Invaders RL" "Starting a resumable training run for ${TIMESTEPS} timesteps."

if [[ "$RESET_REPLAY_BUFFER" == "1" ]]; then
  # Reuse the model weights but intentionally drop replay memory when the
  # stored experience is no longer trustworthy.
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
  # The default path resumes both the model and the replay buffer so repeated
  # cycle runs keep improving the same agent.
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

notify "Space Invaders RL" "Training finished. Starting silent evaluation."

# Silent evaluation gives a quick numeric snapshot without opening a game window.
python -m space_invaders_rl.evaluate \
  --model-path "${MODEL_PATH}.zip" \
  --episodes "$SILENT_EVAL_EPISODES" \
  --max-steps "$SILENT_EVAL_MAX_STEPS" \
  --render-mode none \
  2>&1 | tee -a "$LOG_FILE"

notify "Space Invaders RL" "Silent evaluation finished. Launching one visual game."

# Human rendering is useful as a final sanity check that the agent's behavior
# still looks sensible and not only numerically good.
python -m space_invaders_rl.evaluate \
  --model-path "${MODEL_PATH}.zip" \
  --episodes "$HUMAN_EVAL_EPISODES" \
  --max-steps "$HUMAN_EVAL_MAX_STEPS" \
  --render-mode human \
  --full-game \
  2>&1 | tee -a "$LOG_FILE"

notify "Space Invaders RL" "The full training cycle is complete."
echo "Log saved to $LOG_FILE"

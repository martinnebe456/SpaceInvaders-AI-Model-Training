#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

activate_venv
ensure_mplconfig

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

trap 'notify "Space Invaders RL" "Skript skoncil chybou. Zkontroluj log ${LOG_FILE}."' ERR

notify "Space Invaders RL" "Startuji navazujici trening na ${TIMESTEPS} kroku."

if [[ "$RESET_REPLAY_BUFFER" == "1" ]]; then
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

notify "Space Invaders RL" "Trening dokonceny. Spoustim tichou evaluaci."

python -m space_invaders_rl.evaluate \
  --model-path "${MODEL_PATH}.zip" \
  --episodes "$SILENT_EVAL_EPISODES" \
  --max-steps "$SILENT_EVAL_MAX_STEPS" \
  --render-mode none \
  2>&1 | tee -a "$LOG_FILE"

notify "Space Invaders RL" "Ticha evaluace hotova. Spoustim jednu vizualni hru."

python -m space_invaders_rl.evaluate \
  --model-path "${MODEL_PATH}.zip" \
  --episodes "$HUMAN_EVAL_EPISODES" \
  --max-steps "$HUMAN_EVAL_MAX_STEPS" \
  --render-mode human \
  --full-game \
  2>&1 | tee -a "$LOG_FILE"

notify "Space Invaders RL" "Cely treninkovy cyklus je hotovy."
echo "Log ulozen do $LOG_FILE"

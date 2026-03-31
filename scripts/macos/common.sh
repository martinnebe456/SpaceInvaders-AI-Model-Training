#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

activate_venv() {
  cd "$REPO_ROOT"
  if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Virtualni prostredi .venv nebylo nalezeno."
    exit 1
  fi
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
}

ensure_mplconfig() {
  MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.mplconfig}"
  mkdir -p "$MPLCONFIGDIR"
  export MPLCONFIGDIR
}

notify() {
  local title="$1"
  local message="$2"
  echo "[$(date +"%H:%M:%S")] $title - $message"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi
}

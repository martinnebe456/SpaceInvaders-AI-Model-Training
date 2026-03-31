#!/usr/bin/env bash
set -euo pipefail

# Resolve the shared repo root once so every macOS wrapper can reuse it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

activate_venv() {
  # Always switch to the repo root first so relative paths behave the same way
  # no matter where the user launched the script from.
  cd "$REPO_ROOT"
  if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Virtual environment .venv was not found."
    exit 1
  fi
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
}

ensure_mplconfig() {
  # Matplotlib writes cache files on first import. Keeping them inside the repo
  # avoids surprises with per-user system directories.
  MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.mplconfig}"
  mkdir -p "$MPLCONFIGDIR"
  export MPLCONFIGDIR
}

notify() {
  # Print every status update to stdout first so logs remain useful even on
  # machines without macOS notification support.
  local title="$1"
  local message="$2"
  echo "[$(date +"%H:%M:%S")] $title - $message"
  if command -v osascript >/dev/null 2>&1; then
    # Desktop notifications are best-effort only. Script success must never
    # depend on them working.
    osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi
}

#!/usr/bin/env bash
set -euo pipefail

# This thin wrapper picks a usable Python interpreter and delegates the real
# setup work to the cross-platform bootstrap script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BOOTSTRAP_PY="$REPO_ROOT/scripts/bootstrap/init_env.py"

find_python() {
  # Respect PYTHON_BIN when the caller wants a specific interpreter, then fall
  # back to a reasonable search order for common local installs.
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=("python3.12" "python3.13" "python3" "python")

  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_CMD="$(find_python || true)"
if [[ -z "$PYTHON_CMD" ]]; then
  echo "Python 3.12 or 3.13 was not found. Install one of them and run init again." >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_CMD" "$BOOTSTRAP_PY" --platform macos "$@"

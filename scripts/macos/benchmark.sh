#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

# Benchmarking only needs the Python environment and the Matplotlib cache path.
activate_venv
ensure_mplconfig

python -m space_invaders_rl.benchmark "$@"

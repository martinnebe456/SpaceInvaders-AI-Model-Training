Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
# Benchmarking needs the same Python environment as training, plus the local
# Matplotlib cache path for predictable startup behavior.
Activate-Venv
Ensure-MplConfig

python -m space_invaders_rl.benchmark @args

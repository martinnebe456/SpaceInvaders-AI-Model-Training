Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
Activate-Venv
Ensure-MplConfig

python -m space_invaders_rl.benchmark @args
